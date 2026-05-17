// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cutfemx_wrappers
{
#ifdef CUTFEMX_HAS_LEVEL_SET
void level_set(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_QUADRATURE
void quadrature(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_MESH
void mesh(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_MESH_CUTTER
void mesh_cutter(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_FEM_INTERPOLATION
void fem_interpolation(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_FEM
void fem(nb::module_& m);
void petsc(nb::module_& m);
#endif
} // namespace cutfemx_wrappers

NB_MODULE(cutfemx_cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "CutFEMx Python interface";

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

#ifdef CUTFEMX_HAS_MESH_CUTTER
  cutfemx_wrappers::mesh_cutter(m);
#endif

#ifdef CUTFEMX_HAS_LEVEL_SET
  // Create level_set submodule
  nb::module_ level_set = m.def_submodule("level_set", "LevelSet module");
  cutfemx_wrappers::level_set(level_set);
#endif

#ifdef CUTFEMX_HAS_QUADRATURE
  // Create quadrature submodule
  nb::module_ quadrature = m.def_submodule("quadrature", "Quadrature module");
  cutfemx_wrappers::quadrature(quadrature);
#endif

#ifdef CUTFEMX_HAS_MESH
  // Create mesh submodule
  nb::module_ mesh = m.def_submodule("mesh", "Mesh module");
  cutfemx_wrappers::mesh(mesh);
#endif

#if defined(CUTFEMX_HAS_FEM) || defined(CUTFEMX_HAS_FEM_INTERPOLATION)
  // Create fem submodule
  nb::module_ fem = m.def_submodule("fem", "FEM module");
#ifdef CUTFEMX_HAS_FEM_INTERPOLATION
  cutfemx_wrappers::fem_interpolation(fem);
#endif
#ifdef CUTFEMX_HAS_FEM
  cutfemx_wrappers::fem(fem);

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)
  // PETSc-specific wrappers
  cutfemx_wrappers::petsc(fem);
#endif
#endif
#endif
}
