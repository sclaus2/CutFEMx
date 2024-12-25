// Copyright (C) 2017-2023 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include <array.h>
#include <caster_mpi.h>
#include <caster_petsc.h>
#include <pycoeff.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <petsc4py/petsc4py.h>
#include <petscis.h>

#include <cutfemx/fem/CutForm.h>
#include <cutfemx/fem/assembler.h>
#include <cutfemx/fem/petsc.h>
#include <cutfemx/fem/deactivate.h>

#include <dolfinx/la/petsc.h>

namespace
{

void petsc_fem_module(nb::module_& m)
{
  m.def("create_matrix", cutfemx::fem::petsc::create_matrix<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"),
        nb::arg("type") = std::string(),
        "Create a PETSc Mat for bilinear form.");

  // PETSc Matrices
  m.def(
      "assemble_matrix",
      [](Mat A, const cutfemx::fem::CutForm<PetscScalar, PetscReal>& a,
         nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const PetscScalar, nb::ndim<2>,
                                    nb::c_contig>>& coefficients,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const PetscScalar, nb::ndim<2>,
                                    nb::c_contig>>& coeffs_rt,
         const std::vector<std::shared_ptr<
             const dolfinx::fem::DirichletBC<PetscScalar, PetscReal>>>& bcs)
      {
          cutfemx::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
              std::span(constants.data(), constants.size()),
              py_to_cpp_coeffs(coefficients), py_to_cpp_coeffs(coeffs_rt), bcs);
      },
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("coeffs_rt"), nb::arg("bcs"),
      "Assemble bilinear form into an existing PETSc matrix");

  m.def(
      "deactivate",
      []( Mat A, std::string deactivate_domain,
          const std::shared_ptr<dolfinx::fem::Function<PetscScalar, PetscReal>> level_set,
          const std::shared_ptr<dolfinx::fem::FunctionSpace<PetscReal>> V,
          const std::vector<int>& component,
          PetscScalar diagonal)
      {
          MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
          MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
          dolfinx::fem::Function<PetscScalar, PetscReal> xi = cutfemx::fem::deactivate(dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
                          deactivate_domain, level_set, V, component, diagonal);
          return xi;
      },
      nb::arg("A"), nb::arg("domain to deactivate"), nb::arg("level_set"), nb::arg("V"), nb::arg("component"), nb::arg("diagonal"),
      "Deactivate part of the domain by inserting one along the diagonal.");

  m.def(
      "locate_dofs",
      []( std::string deactivate_domain,
          const std::shared_ptr<dolfinx::fem::Function<PetscScalar, PetscReal>> level_set,
          const std::shared_ptr<dolfinx::fem::FunctionSpace<PetscReal>> V,
          const std::vector<int>& component)
      {
        return dolfinx_wrappers::as_nbarray(cutfemx::fem::locate_dofs(deactivate_domain, level_set, V, component));
      },
      nb::arg("domain to deactivate"), nb::arg("level_set"), nb::arg("V"), nb::arg("component"),
      "Get dofs to deactivate.");
}

} // namespace

namespace cutfemx_wrappers
{
void petsc(nb::module_& m)
{
  nb::module_ petsc_fem_mod
      = m.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_fem_module(petsc_fem_mod);
}
} // namespace cutfemx_wrappers
#endif
