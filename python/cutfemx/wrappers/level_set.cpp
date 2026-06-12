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

#include <cutfemx/cut/cut.h>
#include <cutfemx/cut/runtime_quadrature.h>
#include <cutfemx/geometry/conormal.h>
#include <cutfemx/geometry/correction_distance.h>
#include <cutfemx/geometry/surface_normal.h>
#include <cutfemx/level_set/normal.h>
#include <cutfemx/level_set/value.h>

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

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

  m.def(
      ("evaluate_values_" + type).c_str(),
      [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> offsets,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_map)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::level_set::evaluate_values<T>(
                std::move(level_set),
                std::span<const T>(points.data(), points.size()),
                points.shape(0), points.shape(1),
                std::span<const std::int32_t>(offsets.data(), offsets.size()),
                std::span<const std::int32_t>(parent_map.data(), parent_map.size())));
      },
      nb::arg("level_set"), nb::arg("points"), nb::arg("offsets"),
      nb::arg("parent_map"),
      "Evaluate level-set values at runtime quadrature points.");

  m.def(
      ("evaluate_surface_normals_" + type).c_str(),
      [](const cutfemx::CutData<T>& cut_data,
         const cutfemx::RuntimeQuadrature<T>& quadrature,
         std::int32_t level_set_index,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> offsets,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_map)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::geometry::evaluate_surface_normals<T>(
                cut_data, quadrature, level_set_index,
                std::span<const T>(points.data(), points.size()),
                points.shape(0), points.shape(1),
                std::span<const std::int32_t>(offsets.data(), offsets.size()),
                std::span<const std::int32_t>(parent_map.data(), parent_map.size())));
      },
      nb::arg("cut_data"), nb::arg("quadrature"), nb::arg("level_set_index"),
      nb::arg("points"), nb::arg("offsets"), nb::arg("parent_map"),
      "Evaluate cut-surface normals at runtime quadrature points.");

  m.def(
      ("evaluate_correction_distances_" + type).c_str(),
      [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
         nb::ndarray<const T, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> offsets,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_map,
         nb::ndarray<const double, nb::ndim<2>, nb::c_contig> directions,
         double max_distance, double max_distance_factor, double tolerance,
         int max_iterations)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::geometry::evaluate_correction_distances<T>(
                std::move(level_set),
                std::span<const T>(points.data(), points.size()),
                points.shape(0), points.shape(1),
                std::span<const std::int32_t>(offsets.data(), offsets.size()),
                std::span<const std::int32_t>(parent_map.data(), parent_map.size()),
                std::span<const double>(directions.data(), directions.size()),
                directions.shape(1), max_distance, max_distance_factor,
                tolerance, max_iterations));
      },
      nb::arg("level_set"), nb::arg("points"), nb::arg("offsets"),
      nb::arg("parent_map"), nb::arg("directions"), nb::arg("max_distance"),
      nb::arg("max_distance_factor"), nb::arg("tolerance"),
      nb::arg("max_iterations"),
      "Evaluate correction distances at runtime quadrature points.");

  m.def(
      ("evaluate_conormals_" + type).c_str(),
      [](std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
         nb::ndarray<const double, nb::ndim<2>, nb::c_contig> surface_normals,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> offsets,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> parent_map,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> local_facets,
         double tolerance)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::geometry::evaluate_conormals<T>(
                std::move(mesh),
                std::span<const double>(surface_normals.data(),
                                        surface_normals.size()),
                surface_normals.shape(0), surface_normals.shape(1),
                std::span<const std::int32_t>(offsets.data(), offsets.size()),
                std::span<const std::int32_t>(parent_map.data(), parent_map.size()),
                std::span<const std::int32_t>(local_facets.data(),
                                              local_facets.size()),
                tolerance));
      },
      nb::arg("mesh"), nb::arg("surface_normals"), nb::arg("offsets"),
      nb::arg("parent_map"), nb::arg("local_facets"), nb::arg("tolerance"),
      "Evaluate surface DG conormals at runtime quadrature points.");
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
