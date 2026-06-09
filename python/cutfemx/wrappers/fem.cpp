// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <cstdint>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <array.h>

#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/EntityMap.h>
#include <dolfinx/mesh/Mesh.h>

#include <dolfinx_custom_data/fem/Form.h>
#include <dolfinx_custom_data/fem/assembler.h>

#include <cutfemx/extensions/cell_aggregation.h>
#include <cutfemx/extensions/extension_penalty.h>
#include <cutfemx/fem/deactivate.h>

namespace nb = nanobind;

namespace
{
template <typename T>
using kernel_fn_t
    = void (*)(T*, const T*, const T*, const typename dolfinx::scalar_value_t<T>*,
               const int*, const std::uint8_t*, void*);

template <typename T>
std::function<void(T*, const T*, const T*, const dolfinx::scalar_value_t<T>*,
                   const int*, const std::uint8_t*, void*)>
kernel_from_pointer(std::uintptr_t kernel_ptr)
{
  if (kernel_ptr == 0)
    throw std::runtime_error("Cannot create CutFEMx Form with a null kernel.");

  auto fn = reinterpret_cast<kernel_fn_t<T>>(kernel_ptr);
  return [fn](T* A, const T* w, const T* c,
              const dolfinx::scalar_value_t<T>* coordinate_dofs,
              const int* entity_local_index, const std::uint8_t* permutation,
              void* custom_data)
  { fn(A, w, c, coordinate_dofs, entity_local_index, permutation, custom_data); };
}

template <typename T>
void declare_runtime_fem(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_t<T>;
  using Form = dolfinx_custom_data::fem::Form<T, U>;
  using FunctionSpace = dolfinx::fem::FunctionSpace<U>;
  using Function = dolfinx::fem::Function<T, U>;
  using Constant = dolfinx::fem::Constant<T>;
  using DirichletBC = dolfinx::fem::DirichletBC<T, U>;
  using Vector = dolfinx::la::Vector<T>;
  using Mesh = dolfinx::mesh::Mesh<U>;
  using ActiveDomain = cutfemx::fem::ActiveDomain<T, U>;
  using CellAggregation = cutfemx::extensions::CellAggregation<U>;
  using integral_spec_t
      = std::tuple<std::uintptr_t,
                   nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>,
                   std::vector<int>, std::optional<std::uintptr_t>>;

  std::string form_name = "Form_" + type;
  nb::class_<Form>(m, form_name.c_str(), "CutFEMx runtime form")
      .def_prop_ro("rank", &Form::rank)
      .def_prop_ro("mesh", &Form::mesh, nb::rv_policy::reference_internal)
      .def_prop_ro("function_spaces", &Form::function_spaces,
                   nb::rv_policy::reference_internal)
      .def_prop_ro("needs_facet_permutations",
                   &Form::needs_facet_permutations);

  std::string active_domain_name = "ActiveDomain_" + type;
  nb::class_<ActiveDomain>(m, active_domain_name.c_str(),
                           "CutFEMx form-derived active domain")
      .def_prop_ro(
          "indicator",
          [](ActiveDomain& self) { return self.indicator; },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "active_cells",
          [](ActiveDomain& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.active_cells.data(), {self.active_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "inactive_dofs",
          [](ActiveDomain& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.inactive_dofs.data(), {self.inactive_dofs.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal);

  m.def(
      ("create_form_" + type).c_str(),
      [](const std::vector<std::shared_ptr<const FunctionSpace>>& spaces,
         const std::map<std::tuple<int, int, int>, integral_spec_t>& specs,
         std::shared_ptr<const Mesh> mesh,
         const std::vector<std::shared_ptr<const Function>>& coefficients,
         const std::vector<std::shared_ptr<const Constant>>& constants,
         bool needs_facet_permutations)
      {
        std::map<std::tuple<dolfinx_custom_data::fem::IntegralType, int, int>,
                 dolfinx_custom_data::fem::integral_data<T, U>>
            integrals;

        for (const auto& [key, spec] : specs)
        {
          const auto [type_index, subdomain_id, kernel_index] = key;
          const auto& [kernel_ptr, entities_array, active_coeffs, data_ptr]
              = spec;

          if (type_index < 0 or type_index > 4)
          {
            throw std::runtime_error(
                "CutFEMx Form received an invalid integral type index.");
          }

          std::vector<std::int32_t> entities(
              entities_array.data(), entities_array.data() + entities_array.size());
          std::optional<void*> custom_data = std::nullopt;
          if (data_ptr)
            custom_data = reinterpret_cast<void*>(*data_ptr);

          integrals.emplace(
              std::make_tuple(static_cast<dolfinx_custom_data::fem::IntegralType>(type_index),
                              subdomain_id, kernel_index),
              dolfinx_custom_data::fem::integral_data<T, U>(
                  kernel_from_pointer<T>(kernel_ptr), std::move(entities),
                  active_coeffs, custom_data));
        }

        std::vector<std::reference_wrapper<const dolfinx::mesh::EntityMap>>
            entity_maps;
        return std::make_shared<Form>(spaces, std::move(integrals), mesh,
                                      coefficients, constants,
                                      needs_facet_permutations, entity_maps);
      },
      nb::arg("function_spaces"), nb::arg("integrals"), nb::arg("mesh"),
      nb::arg("coefficients"), nb::arg("constants"),
      nb::arg("needs_facet_permutations"),
      "Create a CutFEMx runtime Form from compiled runintgen integral data.");

  m.def(
      ("assemble_scalar_" + type).c_str(),
      [](const Form& form) { return dolfinx_custom_data::fem::assemble_scalar(form); },
      nb::arg("form"), "Assemble a scalar CutFEMx runtime form.");

  m.def(
      ("assemble_vector_" + type).c_str(),
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b, const Form& form)
      {
        if (form.rank() != 1)
        {
          throw std::runtime_error(
              "Cannot assemble vector. Form is not linear.");
        }
        dolfinx_custom_data::fem::assemble_vector(std::span<T>(b.data(), b.size()), form);
      },
      nb::arg("b"), nb::arg("form"),
      "Assemble a linear CutFEMx runtime form into an existing array.");

  m.def(
      ("apply_lifting_" + type).c_str(),
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const std::vector<const Form*>& forms,
         const std::vector<std::vector<const DirichletBC*>>& bcs,
         std::optional<std::vector<
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig>>> x0,
         T alpha)
      {
        if (forms.size() != bcs.size())
        {
          throw std::runtime_error(
              "Mismatch in size between bilinear forms and boundary "
              "condition blocks.");
        }

        std::vector<std::optional<std::reference_wrapper<const Form>>>
            lifting_forms;
        lifting_forms.reserve(forms.size());
        for (const Form* form : forms)
        {
          if (form == nullptr)
            lifting_forms.emplace_back(std::nullopt);
          else
            lifting_forms.emplace_back(std::cref(*form));
        }

        std::vector<std::vector<std::reference_wrapper<const DirichletBC>>>
            lifting_bcs;
        lifting_bcs.reserve(bcs.size());
        for (const std::vector<const DirichletBC*>& block : bcs)
        {
          auto& out_block = lifting_bcs.emplace_back();
          out_block.reserve(block.size());
          for (const DirichletBC* bc : block)
          {
            if (bc == nullptr)
              throw std::runtime_error(
                  "CutFEMx apply_lifting received a null DirichletBC.");
            out_block.push_back(*bc);
          }
        }

        std::vector<std::span<const T>> lifting_x0;
        if (x0)
        {
          if (x0->size() != forms.size())
          {
            throw std::runtime_error(
                "Mismatch in size between x0 and bilinear forms.");
          }
          lifting_x0.reserve(x0->size());
          for (const auto& values : *x0)
            lifting_x0.emplace_back(values.data(), values.size());
        }

        dolfinx_custom_data::fem::apply_lifting(
            std::span<T>(b.data(), b.size()), std::move(lifting_forms),
            lifting_bcs, lifting_x0, alpha);
      },
      nb::arg("b"), nb::arg("forms"), nb::arg("bcs"),
      nb::arg("x0") = nb::none(), nb::arg("alpha") = T(1),
      "Apply Dirichlet lifting for CutFEMx runtime bilinear forms.");

  m.def(
      ("create_sparsity_pattern_" + type).c_str(),
      [](const Form& form)
      {
        return dolfinx_custom_data::fem::create_sparsity_pattern(form);
      },
      nb::arg("form"),
      "Create an unfinalized sparsity pattern for a CutFEMx runtime form.");

  m.def(
      ("create_matrix_" + type).c_str(),
      [](const Form& form)
      {
        dolfinx::la::SparsityPattern sp
            = dolfinx_custom_data::fem::create_sparsity_pattern(form);
        sp.finalize();
        return dolfinx::la::MatrixCSR<T>(sp);
      },
      nb::arg("form"),
      "Create a MatrixCSR compatible with a CutFEMx runtime form.");

  m.def(
      ("create_matrix_with_extension_sparsity_" + type).c_str(),
      [](const Form& form,
         const std::vector<std::shared_ptr<const FunctionSpace>>& spaces,
         const std::vector<const CellAggregation*>& aggregations)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot create matrix. Form is not bilinear.");
        }
        if (spaces.size() != aggregations.size())
        {
          throw std::runtime_error(
              "Extension sparsity spaces and aggregations have different sizes.");
        }

        dolfinx::la::SparsityPattern sp
            = dolfinx_custom_data::fem::create_sparsity_pattern(form);
        for (std::size_t i = 0; i < spaces.size(); ++i)
        {
          if (!spaces[i])
            throw std::runtime_error("Received a null extension function space.");
          if (aggregations[i] == nullptr)
            throw std::runtime_error("Received a null extension aggregation.");
          cutfemx::extensions::insert_extension_penalty_sparsity(
              sp, *spaces[i], *aggregations[i]);
        }
        sp.finalize();
        return dolfinx::la::MatrixCSR<T>(sp);
      },
      nb::arg("form"), nb::arg("spaces"), nb::arg("aggregations"),
      "Create a MatrixCSR with CutFEMx runtime form and extension sparsity.");

  m.def(
      ("assemble_matrix_" + type).c_str(),
      [](dolfinx::la::MatrixCSR<T>& A, const Form& form,
         const std::vector<const DirichletBC*>& bcs)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot assemble matrix. Form is not bilinear.");
        }

        std::vector<std::reference_wrapper<const DirichletBC>> _bcs;
        for (const DirichletBC* bc : bcs)
        {
          assert(bc);
          _bcs.push_back(*bc);
        }

        const std::array<int, 2> data_bs
            = {form.function_spaces().at(0)->dofmaps().front()->index_map_bs(),
               form.function_spaces().at(1)->dofmaps().front()->index_map_bs()};

        if (data_bs[0] != data_bs[1])
        {
          throw std::runtime_error(
              "Non-square blocksize unsupported in Python");
        }

        if (data_bs[0] == 1)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.mat_add_values(), form, _bcs);
        }
        else if (data_bs[0] == 2)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<2, 2>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 3)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<3, 3>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 4)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<4, 4>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 5)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<5, 5>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 6)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<6, 6>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 7)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<7, 7>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 8)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<8, 8>(),
                                        form, _bcs);
        }
        else if (data_bs[0] == 9)
        {
          dolfinx_custom_data::fem::assemble_matrix(A.template mat_add_values<9, 9>(),
                                        form, _bcs);
        }
        else
          throw std::runtime_error("Block size not supported in Python");
      },
      nb::arg("A"), nb::arg("form"), nb::arg("bcs"),
      "Assemble a bilinear CutFEMx runtime form into an existing MatrixCSR.");

  m.def(
      ("insert_diagonal_" + type).c_str(),
      [](dolfinx::la::MatrixCSR<T>& A, const FunctionSpace& V,
         const std::vector<const DirichletBC*>& bcs, T diagonal)
      {
        std::vector<std::reference_wrapper<const DirichletBC>> _bcs;
        for (const DirichletBC* bc : bcs)
        {
          assert(bc);
          _bcs.push_back(*bc);
        }
        dolfinx_custom_data::fem::set_diagonal(A.mat_set_values(), V, _bcs, diagonal);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("bcs"), nb::arg("diagonal"),
      "Insert a diagonal value for constrained rows.");

  m.def(
      ("active_domain_" + type).c_str(),
      [](const Form& form)
      {
        return std::make_shared<ActiveDomain>(
            cutfemx::fem::active_domain(form));
      },
      nb::arg("form"), "Build a CutFEMx active-domain support object.");

  m.def(
      ("deactivate_outside_" + type).c_str(),
      [](dolfinx::la::MatrixCSR<T>& A, ActiveDomain& active_domain,
         T diagonal)
      {
        cutfemx::fem::impl::validate_matrix_rows(A,
                                                 active_domain.inactive_dofs);
        cutfemx::fem::deactivate_outside(A.mat_set_values(), active_domain,
                                         diagonal);
      },
      nb::arg("A"), nb::arg("active_domain"),
      nb::arg("diagonal"),
      "Deactivate matrix rows outside a CutFEMx active domain.");

  m.def(
      ("deactivate_outside_matrix_vector_" + type).c_str(),
      [](dolfinx::la::MatrixCSR<T>& A, Vector& b,
         ActiveDomain& active_domain, T diagonal, T rhs_value)
      {
        cutfemx::fem::impl::validate_matrix_rows(A,
                                                 active_domain.inactive_dofs);
        auto& values = b.array();
        cutfemx::fem::deactivate_outside(
            A.mat_set_values(), std::span<T>(values.data(), values.size()),
            active_domain, diagonal, rhs_value);
      },
      nb::arg("A"), nb::arg("b"), nb::arg("active_domain"),
      nb::arg("diagonal"), nb::arg("rhs_value"),
      "Deactivate matrix rows and matching RHS entries outside a CutFEMx active domain.");

  m.def(
      ("deactivate_outside_blocks_" + type).c_str(),
      [](const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
         const std::vector<ActiveDomain*>& active_domains, T diagonal)
      {
        cutfemx::fem::deactivate_outside_blocks(A_blocks, active_domains,
                                                diagonal);
      },
      nb::arg("A_blocks"), nb::arg("active_domains"),
      nb::arg("diagonal"),
      "Deactivate a MatrixCSR block system from per-row ActiveDomain objects.");

  m.def(
      ("deactivate_outside_blocks_matrix_vector_" + type).c_str(),
      [](const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
         const std::vector<Vector*>& b_blocks,
         const std::vector<ActiveDomain*>& active_domains, T diagonal,
         T rhs_value)
      {
        cutfemx::fem::deactivate_outside_blocks(
            A_blocks, b_blocks, active_domains, diagonal, rhs_value);
      },
      nb::arg("A_blocks"), nb::arg("b_blocks"), nb::arg("active_domains"),
      nb::arg("diagonal"), nb::arg("rhs_value"),
      "Deactivate a MatrixCSR block system and matching RHS blocks from per-row ActiveDomain objects.");

  m.def(
      ("zero_rows_" + type).c_str(),
      [](const dolfinx::la::MatrixCSR<T>& A, double tol)
      { return cutfemx::fem::impl::zero_rows(A, tol); },
      nb::arg("A"), nb::arg("tol") = 0.0,
      "Return owned scalar rows whose MatrixCSR entries are all zero.");

  m.def(
      ("zero_block_rows_" + type).c_str(),
      [](const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
         double tol)
      { return cutfemx::fem::impl::zero_block_rows(A_blocks, tol); },
      nb::arg("A_blocks"), nb::arg("tol") = 0.0,
      "Return owned scalar rows whose entries are zero across each MatrixCSR block row.");
}
} // namespace

namespace cutfemx_wrappers
{
void fem_runtime(nb::module_& m)
{
  declare_runtime_fem<double>(m, "float64");
}
} // namespace cutfemx_wrappers
