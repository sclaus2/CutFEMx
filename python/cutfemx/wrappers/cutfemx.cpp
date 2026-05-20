// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cutfemx_wrappers
{
#ifdef CUTFEMX_HAS_CUT_API
void cut(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_FEM_INTERPOLATION
void fem_interpolation(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_RUNTIME_FEM
void fem_runtime(nb::module_& m);
#endif
#ifdef CUTFEMX_HAS_RUNTIME_PETSC
void petsc_runtime(nb::module_& m);
#endif
} // namespace cutfemx_wrappers

NB_MODULE(cutfemx_cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "CutFEMx Python interface";

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

#ifdef CUTFEMX_HAS_CUT_API
  cutfemx_wrappers::cut(m);
#endif

#if defined(CUTFEMX_HAS_FEM_INTERPOLATION) || defined(CUTFEMX_HAS_RUNTIME_FEM)
  nb::module_ fem = m.def_submodule("fem", "FEM module");
#ifdef CUTFEMX_HAS_FEM_INTERPOLATION
  cutfemx_wrappers::fem_interpolation(fem);
#endif
#ifdef CUTFEMX_HAS_RUNTIME_FEM
  cutfemx_wrappers::fem_runtime(fem);
#endif
#if defined(CUTFEMX_HAS_RUNTIME_PETSC) && defined(HAS_PETSC)                   \
    && defined(HAS_PETSC4PY)
  cutfemx_wrappers::petsc_runtime(fem);
#endif
#endif
}
