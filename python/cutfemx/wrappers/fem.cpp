// Copyright (c) 2022 ONERA
// Authors: Susanne Claus
// This file is part of CutCells
//
// SPDX-License-Identifier:    MIT
#include <array.h>
#include <caster_mpi.h>
#include <numpy_dtype.h>
#include <pycoeff.h>

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <span>
#include <map>
#include <algorithm>

#include <cutfemx/fem/CutForm.h>
#include <cutfemx/fem/interpolate.h>
#include <cutfemx/quadrature/quadrature.h>
#include <cutfemx/fem/assembler.h>
#include <cutfemx/fem/pack_coefficients.h>
#include <cutfemx/fem/deactivate.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <ufcx.h>


namespace nb = nanobind;

namespace
{

template <typename T>
std::map<std::pair<cutfemx::fem::IntegralType, int>,
         std::pair<std::span<const T>, int>>
py_to_cpp_coeffs_rt(
    const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                   nb::ndarray<T, nb::ndim<2>, nb::c_contig>>& coeffs)
{
  using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key_t, std::pair<std::span<const T>, int>> c;
  std::ranges::transform(
      coeffs, std::inserter(c, c.end()),
      [](auto& e) -> typename decltype(c)::value_type
      {
        return {e.first,
                {std::span(static_cast<const T*>(e.second.data()),
                           e.second.shape(0)),
                 e.second.shape(1)}};
      });
  return c;
}

template <typename T>
void declare_fem(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  std::string pyclass_cutform_name = std::string("CutForm_") + type;
  nb::class_<cutfemx::fem::CutForm<T,U>>(m, pyclass_cutform_name.c_str(),
                                         "CutForm object")
      .def(
          "__init__",
          [](cutfemx::fem::CutForm<T, U>* fp, std::uintptr_t ufcform,
                std::shared_ptr<const dolfinx::fem::Form<T, U>> form,
                const std::map<cutfemx::fem::IntegralType,
                std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>>>&
                subdomains,
                const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                               std::vector<cutfemx::fem::ElementSource>>&
                element_sources_map = {})
          {
            ufcx_form* p = reinterpret_cast<ufcx_form*>(ufcform);
            new (fp)
                cutfemx::fem::CutForm<T, U>(cutfemx::fem::create_cut_form_factory<T>(
                    *p, form, subdomains, element_sources_map));
          }
          , nb::arg("ufcx_form"), nb::arg("form"), nb::arg("subdomains"),
            nb::arg("element_sources_map") = std::map<std::pair<cutfemx::fem::IntegralType, int>,
                                                       std::vector<cutfemx::fem::ElementSource>>{}
          , "Create a Cutform from a pointer to a ufcx_form")
      .def(
          "integral_ids",
          [](const cutfemx::fem::CutForm<T, U>& self,
             cutfemx::fem::IntegralType type)
          {
            auto ids = self.integral_ids(type);
            return dolfinx_wrappers::as_nbarray(std::move(ids));
          },
          nb::arg("type"))
      .def_prop_ro("integral_types", &cutfemx::fem::CutForm<T, U>::integral_types)
      .def_prop_ro("form", &cutfemx::fem::CutForm<T, U>::form)
      .def("update_integration_domains",[](cutfemx::fem::CutForm<T, U>& self,
             const std::map<
                 dolfinx::fem::IntegralType,
                 std::vector<std::pair<
                     std::int32_t, nb::ndarray<const std::int32_t, nb::ndim<1>,
                                               nb::c_contig>>>>& subdomains,
             const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                            nb::ndarray<const std::int32_t, nb::ndim<1>,
                                        nb::c_contig>>& entity_maps = {})
        {
          //convert array to span
          std::map<dolfinx::fem::IntegralType,
          std::vector<std::pair<std::int32_t,
                                std::span<const std::int32_t>>>>
          sd;
          for (auto& [itg, data] : subdomains)
          {
            std::vector<
                std::pair<std::int32_t, std::span<const std::int32_t>>>
                x;
            for (auto& [id, e] : data)
              x.emplace_back(id, std::span(e.data(), e.size()));
            sd.insert({itg, std::move(x)});
          }
          std::map<std::shared_ptr<const dolfinx::mesh::Mesh<U>>,
                    std::span<const int32_t>>
              _entity_maps;
          for (auto& [msh, map] : entity_maps)
            _entity_maps.emplace(msh, std::span(map.data(), map.size()));

          self.update_integration_domains(sd,_entity_maps);
        }, nb::arg("subdomains"), nb::arg("entity_maps"))
      .def("update_runtime_domains", &cutfemx::fem::CutForm<T, U>::update_runtime_domains)
      .def(
          "quadrature_rules",
          [](const cutfemx::fem::CutForm<T, U>& self,
             cutfemx::fem::IntegralType type, int i)
          {
            return self.quadrature_rules(type, i);
          },
          nb::rv_policy::reference_internal, nb::arg("type"), nb::arg("i"));

  m.def("create_cut_function", [](const dolfinx::fem::Function<T,U>& u, const cutfemx::mesh::CutMesh<U>& sub_mesh)
          {
            return cutfemx::fem::create_cut_function(u, sub_mesh);
          }
        , "create function over cut mesh");

  m.def("interpolate_cut_expression",
        [](dolfinx::fem::Expression<T, U>& expr,
           std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V_cut,
           const cutfemx::mesh::CutMesh<U>& sub_mesh)
        {
          // Cast to const shared_ptr to match function signature
          auto V_cut_const = std::const_pointer_cast<const dolfinx::fem::FunctionSpace<U>>(V_cut);
          return cutfemx::fem::interpolate_cut_expression(expr, V_cut_const, sub_mesh);
        },
        nb::arg("expr"), nb::arg("V_cut"), nb::arg("sub_mesh"),
        "Interpolate expression onto cut mesh using parent cell mapping.");

    m.def(
      "pack_coefficients",
      [](const cutfemx::fem::CutForm<T, U>& form)
      {
        using Key_t = typename std::pair<cutfemx::fem::IntegralType, int>;

        // Pack coefficients
        std::map<Key_t, std::pair<std::vector<T>, int>> coeffs
            = cutfemx::fem::allocate_coefficient_storage(form);
        cutfemx::fem::pack_coefficients(form, coeffs);

        // Move into NumPy data structures
        std::map<Key_t, nb::ndarray<T, nb::numpy>> c;
        std::ranges::transform(
            coeffs, std::inserter(c, c.end()),
            [](auto& e) -> typename decltype(c)::value_type
            {
              std::size_t num_ents
                  = e.second.first.empty()
                        ? 0
                        : e.second.first.size() / e.second.second;
              return std::pair<const std::pair<cutfemx::fem::IntegralType, int>,
                               nb::ndarray<T, nb::numpy>>(
                  e.first,
                  dolfinx_wrappers::as_nbarray(
                      std::move(e.second.first),
                      {num_ents, static_cast<std::size_t>(e.second.second)}));
            });

        return c;
      },
      nb::arg("form"), "Pack coefficients for a CutForm.");

  // Functional
  m.def(
      "assemble_scalar",
      [](const cutfemx::fem::CutForm<T, U>& M,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients,
         const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coeffs_rt)
      {
        return cutfemx::fem::assemble_scalar<T,U>(
            M, std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients),py_to_cpp_coeffs_rt(coeffs_rt));
      },
      nb::arg("M"), nb::arg("constants"), nb::arg("coefficients"), nb::arg("coefficients runtime"),
      "Assemble functional over mesh with provided constants and "
      "coefficients");
  m.def(
      "assemble_vector",
      [](nb::ndarray<T, nb::ndim<1>, nb::c_contig> b,
         const cutfemx::fem::CutForm<T, U>& L,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients,
         const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coeffs_rt)
      {
        cutfemx::fem::assemble_vector<T>(
            std::span(b.data(), b.size()), L,
            std::span(constants.data(), constants.size()),
            py_to_cpp_coeffs(coefficients),
            py_to_cpp_coeffs_rt(coeffs_rt));
      },
      nb::arg("b"), nb::arg("L"), nb::arg("constants"), nb::arg("coeffs"), nb::arg("coeffs_rt"),
      "Assemble linear form into an existing vector with pre-packed constants "
      "and coefficients");

   m.def(
      "create_sparsity_pattern",
      [](const cutfemx::fem::CutForm<T, U>& a)
      {
        return cutfemx::fem::create_sparsity_pattern(a);
      },
      nb::arg("a"),
      "Create a sparsity pattern.");
    m.def(
      "assemble_matrix",
      [](dolfinx::la::MatrixCSR<T>& A, const cutfemx::fem::CutForm<T, U>& a,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coefficients,
         const std::map<std::pair<cutfemx::fem::IntegralType, int>,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig>>&
             coeffs_rt,
         const std::vector<
             std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
      {
        const std::array<int, 2> data_bs
            = {a._form->function_spaces().at(0)->dofmap()->index_map_bs(),
               a._form->function_spaces().at(1)->dofmap()->index_map_bs()};

        if (data_bs[0] != data_bs[1])
          throw std::runtime_error(
              "Non-square blocksize unsupported in Python");

        if (data_bs[0] == 1)
        {
          cutfemx::fem::assemble_matrix(
              A.mat_add_values(), a,
              std::span<const T>(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 2)
        {
          auto mat_add = A.template mat_add_values<2, 2>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 3)
        {
          auto mat_add = A.template mat_add_values<3, 3>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 4)
        {
          auto mat_add = A.template mat_add_values<4, 4>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 5)
        {
          auto mat_add = A.template mat_add_values<5, 5>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 6)
        {
          auto mat_add = A.template mat_add_values<6, 6>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 7)
        {
          auto mat_add = A.template mat_add_values<7, 7>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 8)
        {
          auto mat_add = A.template mat_add_values<8, 8>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else if (data_bs[0] == 9)
        {
          auto mat_add = A.template mat_add_values<9, 9>();
          cutfemx::fem::assemble_matrix(
              mat_add, a, std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients),
              py_to_cpp_coeffs_rt(coeffs_rt), bcs);
        }
        else
          throw std::runtime_error("Block size not supported in Python");
      },
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"), nb::arg("coeffs_rt"),
      nb::arg("bcs"), "Experimental.");

      m.def(
      "deactivate",
      []( dolfinx::la::MatrixCSR<T>& A, std::string deactivate_domain,
          const std::shared_ptr<dolfinx::fem::Function<T,U>> level_set,
          const std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V,
          const std::vector<int>& component,
          T diagonal)
      {
          dolfinx::fem::Function<T,U> xi = cutfemx::fem::deactivate(A.mat_set_values(), 
                          deactivate_domain, level_set, V, component, diagonal);
          return xi;
      },
      nb::arg("A"), nb::arg("domain to deactivate"), nb::arg("level_set"), nb::arg("V"), nb::arg("component"), nb::arg("diagonal"),
      "Deactivate part of the domain by inserting one along the diagonal.");

}

} // namespace

namespace cutfemx_wrappers
{
  void fem(nb::module_& m)
  {
    nb::enum_<cutfemx::fem::IntegralType>(m, "IntegralType")
    .value("cutcell", cutfemx::fem::IntegralType::cutcell, "runtime integral on cell")
    .value("interface", cutfemx::fem::IntegralType::interface,
            "runtime integral on interface between two parent cells");

    // Bind ElementSourceType enum
    nb::enum_<cutfemx::fem::ElementSourceType>(m, "ElementSourceType")
        .value("Argument", cutfemx::fem::ElementSourceType::Argument)
        .value("Coefficient", cutfemx::fem::ElementSourceType::Coefficient)
        .value("Geometry", cutfemx::fem::ElementSourceType::Geometry);

    // Bind ElementSource struct
    nb::class_<cutfemx::fem::ElementSource>(m, "ElementSource")
        .def(nb::init<cutfemx::fem::ElementSourceType, int, int, int>(),
             nb::arg("type"), nb::arg("index"), nb::arg("component"), nb::arg("deriv_order"))
        .def_rw("type", &cutfemx::fem::ElementSource::type)
        .def_rw("index", &cutfemx::fem::ElementSource::index)
        .def_rw("component", &cutfemx::fem::ElementSource::component)
        .def_rw("deriv_order", &cutfemx::fem::ElementSource::deriv_order);

    declare_fem<float>(m, "float32");
    declare_fem<double>(m, "float64");
  }
} // end of namespace cutfemx_wrappers