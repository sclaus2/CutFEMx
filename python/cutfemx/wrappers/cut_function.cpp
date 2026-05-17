// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <cutfemx/fem/interpolate.h>
#include <cutfemx/mesh/cut_mesh.h>

#include <dolfinx/fem/Function.h>

namespace nb = nanobind;

namespace
{
template <typename T, typename U>
void declare_cut_function(nb::module_& m)
{
  m.def(
      "create_cut_function",
      [](const dolfinx::fem::Function<T, U>& u,
         const cutfemx::mesh::CutMesh<U>& cut_mesh)
      { return cutfemx::fem::create_cut_function(u, cut_mesh); },
      nb::arg("u"), nb::arg("cut_mesh"),
      "Create a function on a CutMesh using its parent-cell mapping.");
}
} // namespace

namespace cutfemx_wrappers
{
void fem_interpolation(nb::module_& m)
{
  declare_cut_function<float, float>(m);
  declare_cut_function<double, double>(m);
}
} // namespace cutfemx_wrappers
