// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

#include <array.h>
#include <caster_petsc.h>

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>

#include <dolfinx_custom_data/fem/assembler.h>

#include <cutfemx/fem/deactivate.h>

#include <cassert>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_runtime_petsc(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_t<T>;
  using Form = dolfinx_custom_data::fem::Form<T, U>;
  using FunctionSpace = dolfinx::fem::FunctionSpace<U>;
  using DirichletBC = dolfinx::fem::DirichletBC<T, U>;
  using ActiveDomain = cutfemx::fem::ActiveDomain<T, U>;

  m.def(
      ("create_matrix_" + type).c_str(),
      [](const Form& form, std::optional<std::string> mat_type)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot create PETSc matrix. Form is not bilinear.");
        }

        dolfinx::la::SparsityPattern sp
            = dolfinx_custom_data::fem::create_sparsity_pattern(form);
        sp.finalize();
        return dolfinx::la::petsc::create_matrix(form.mesh()->comm(), sp,
                                                 mat_type);
      },
      nb::rv_policy::take_ownership, nb::arg("form"),
      nb::arg("type") = nb::none(),
      "Create a PETSc Mat compatible with a CutFEMx runtime form.");

  m.def(
      ("assemble_vector_" + type).c_str(),
      [](Vec b, const Form& form)
      {
        if (form.rank() != 1)
        {
          throw std::runtime_error(
              "Cannot assemble PETSc vector. Form is not linear.");
        }

        Vec b_local;
        VecGhostGetLocalForm(b, &b_local);
        PetscInt n = 0;
        VecGetSize(b_local, &n);
        PetscScalar* array = nullptr;
        VecGetArray(b_local, &array);
        std::span<T> values(reinterpret_cast<T*>(array), n);
        dolfinx_custom_data::fem::assemble_vector(values, form);
        VecRestoreArray(b_local, &array);
        VecGhostRestoreLocalForm(b, &b_local);
      },
      nb::arg("b"), nb::arg("form"),
      "Assemble a linear CutFEMx runtime form into an existing PETSc Vec.");

  m.def(
      ("assemble_matrix_" + type).c_str(),
      [](Mat A, const Form& form, const std::vector<const DirichletBC*>& bcs)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot assemble PETSc matrix. Form is not bilinear.");
        }

        std::vector<std::reference_wrapper<const DirichletBC>> _bcs;
        for (const DirichletBC* bc : bcs)
        {
          assert(bc);
          _bcs.push_back(*bc);
        }

        const std::array<int, 2> data_bs
            = {form.function_spaces().at(0)->dofmaps(0)->index_map_bs(),
               form.function_spaces().at(1)->dofmaps(0)->index_map_bs()};

        if (data_bs[0] == data_bs[1])
        {
          dolfinx_custom_data::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), form,
              _bcs);
        }
        else
        {
          dolfinx_custom_data::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_expand_fn(
                  A, data_bs[0], data_bs[1], ADD_VALUES),
              form, _bcs);
        }
      },
      nb::arg("A"), nb::arg("form"), nb::arg("bcs"),
      "Assemble a bilinear CutFEMx runtime form into an existing PETSc Mat.");

  m.def(
      ("insert_diagonal_" + type).c_str(),
      [](Mat A, const FunctionSpace& V,
         const std::vector<const DirichletBC*>& bcs, T diagonal)
      {
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

        std::vector<std::reference_wrapper<const DirichletBC>> _bcs;
        for (const DirichletBC* bc : bcs)
        {
          assert(bc);
          _bcs.push_back(*bc);
        }
        dolfinx_custom_data::fem::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES), V, _bcs,
            diagonal);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("bcs"), nb::arg("diagonal"),
      "Insert a diagonal value for constrained PETSc matrix rows.");

  m.def(
      ("deactivate_outside_" + type).c_str(),
      [](Mat A, ActiveDomain& active_domain, T diagonal)
      {
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

        PetscInt local_rows = 0;
        PetscInt local_cols = 0;
        MatGetLocalSize(A, &local_rows, &local_cols);
        if (!active_domain.inactive_dofs.empty()
            && active_domain.inactive_dofs.back() >= local_rows)
        {
          throw std::runtime_error(
              "ActiveDomain inactive rows are incompatible with the PETSc matrix row map");
        }
        cutfemx::fem::deactivate_outside(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
            active_domain, diagonal);
      },
      nb::arg("A"), nb::arg("active_domain"), nb::arg("diagonal"),
      "Deactivate PETSc matrix rows outside a CutFEMx active domain.");

  m.def(
      ("deactivate_outside_matrix_vector_" + type).c_str(),
      [](Mat A, Vec b, ActiveDomain& active_domain, T diagonal, T rhs_value)
      {
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

        PetscInt local_rows = 0;
        PetscInt local_cols = 0;
        MatGetLocalSize(A, &local_rows, &local_cols);
        if (!active_domain.inactive_dofs.empty()
            && active_domain.inactive_dofs.back() >= local_rows)
        {
          throw std::runtime_error(
              "ActiveDomain inactive rows are incompatible with the PETSc matrix row map");
        }

        Vec b_local;
        VecGhostGetLocalForm(b, &b_local);
        PetscInt n = 0;
        VecGetLocalSize(b_local, &n);
        PetscScalar* array = nullptr;
        VecGetArray(b_local, &array);

        try
        {
          cutfemx::fem::deactivate_outside(
              dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
              std::span<T>(reinterpret_cast<T*>(array), n),
              active_domain, diagonal, rhs_value);
        }
        catch (...)
        {
          VecRestoreArray(b_local, &array);
          VecGhostRestoreLocalForm(b, &b_local);
          throw;
        }

        VecRestoreArray(b_local, &array);
        VecGhostRestoreLocalForm(b, &b_local);
      },
      nb::arg("A"), nb::arg("b"), nb::arg("active_domain"),
      nb::arg("diagonal"), nb::arg("rhs_value"),
      "Deactivate PETSc matrix rows and matching RHS entries outside a CutFEMx active domain.");
}
} // namespace

namespace cutfemx_wrappers
{
void petsc_runtime(nb::module_& m)
{
  nb::module_ petsc_mod
      = m.def_submodule("petsc", "PETSc-specific runtime FEM module");
  declare_runtime_petsc<double>(petsc_mod, "float64");
}
} // namespace cutfemx_wrappers

#endif
