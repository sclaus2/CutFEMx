// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <array.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>

#include <span>

#include <cutfemx/level_set/normal.h>

#include <dolfinx/fem/Function.h>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_level_set(nb::module_& m, std::string type)
{
  m.def(
      ("evaluate_normals_" + type).c_str(),
      [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> offsets,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_map,
         double sign)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::level_set::evaluate_normals<T>(
                std::move(level_set),
                std::span<const T>(points.data(), points.size()),
                points.shape(0), points.shape(1),
                std::span<const std::int32_t>(offsets.data(), offsets.size()),
                std::span<const std::int32_t>(parent_map.data(), parent_map.size()),
                sign));
      },
      nb::arg("level_set"), nb::arg("points"), nb::arg("offsets"),
      nb::arg("parent_map"), nb::arg("sign"),
      "Evaluate normalized level-set gradients at runtime quadrature points.");
}
} // namespace

namespace cutfemx_wrappers
{
void level_set(nb::module_& m)
{
  declare_level_set<float>(m, "float32");
  declare_level_set<double>(m, "float64");
}
} // namespace cutfemx_wrappers
