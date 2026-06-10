// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include <complex>
#include <stdexcept>
#include <type_traits>

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
      {
        if constexpr (
            (std::is_same_v<T, std::complex<float>>
             && std::is_same_v<U, double>)
            || (std::is_same_v<T, std::complex<double>>
                && std::is_same_v<U, float>))
        {
          throw std::runtime_error(
              "create_cut_function for complex scalar fields with "
              "non-default geometry precision is not supported by the "
              "DOLFINx Function::eval path.");
        }
        else
        {
          return cutfemx::fem::create_cut_function(u, cut_mesh);
        }
      },
      nb::arg("u"), nb::arg("cut_mesh"),
      "Create a function on a CutMesh using its parent-cell mapping.");
}
} // namespace

namespace cutfemx_wrappers
{
void fem_interpolation(nb::module_& m)
{
  declare_cut_function<float, float>(m);
  declare_cut_function<float, double>(m);
  declare_cut_function<double, float>(m);
  declare_cut_function<double, double>(m);
  declare_cut_function<std::complex<float>, float>(m);
  declare_cut_function<std::complex<float>, double>(m);
  declare_cut_function<std::complex<double>, float>(m);
  declare_cut_function<std::complex<double>, double>(m);
}
} // namespace cutfemx_wrappers
