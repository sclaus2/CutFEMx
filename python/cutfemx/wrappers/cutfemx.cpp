// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace cutfemx_wrappers
{
void level_set(nb::module_& m);
} // namespace cutfemx_wrappers

NB_MODULE(cutfemx_cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "CutFEMx Python interface";

#ifdef NDEBUG
  nanobind::set_leak_warnings(false);
#endif

  // Create level_set submodule
  nb::module_ level_set = m.def_submodule("level_set", "LevelSet module");
  cutfemx_wrappers::level_set(level_set);

}
