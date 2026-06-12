// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <basix/cell.h>
#include <basix/mdspan.hpp>

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>

namespace cutfemx::geometry
{
namespace detail
{
template <typename X>
using conormal_mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

template <typename X>
using conormal_mdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

inline std::vector<double> checked_unit_vector(std::vector<double> value,
                                               double tolerance,
                                               const std::string& label)
{
  double norm = 0.0;
  for (double component : value)
    norm += component * component;
  norm = std::sqrt(norm);
  if (norm <= tolerance)
    throw std::runtime_error(label + " is degenerate.");
  for (double& component : value)
    component /= norm;
  return value;
}

template <std::floating_point T>
std::vector<double> physical_reference_points(
    const dolfinx::fem::CoordinateElement<T>& cmap,
    std::span<const T> coordinate_dofs_storage,
    int gdim,
    int tdim,
    std::span<const double> reference_points)
{
  const std::size_t num_points
      = reference_points.size() / static_cast<std::size_t>(tdim);
  std::vector<T> reference_storage(reference_points.size());
  for (std::size_t i = 0; i < reference_points.size(); ++i)
    reference_storage[i] = static_cast<T>(reference_points[i]);

  const std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(0, num_points);
  std::vector<T> phi_storage(std::reduce(
      phi_shape.begin(), phi_shape.end(), std::size_t(1), std::multiplies{}));
  cmap.tabulate(0, std::span<const T>(reference_storage.data(),
                                      reference_storage.size()),
                {num_points, static_cast<std::size_t>(tdim)}, phi_storage);
  conormal_mdspan4_t<const T> phi(phi_storage.data(), phi_shape);

  const std::size_t num_coordinate_dofs = cmap.dim();
  conormal_mdspan2_t<const T> coordinate_dofs(coordinate_dofs_storage.data(),
                                              num_coordinate_dofs,
                                              static_cast<std::size_t>(gdim));
  std::vector<double> values(num_points * static_cast<std::size_t>(gdim), 0.0);
  for (std::size_t p = 0; p < num_points; ++p)
  {
    for (int d = 0; d < gdim; ++d)
    {
      double value = 0.0;
      for (std::size_t j = 0; j < num_coordinate_dofs; ++j)
        value += static_cast<double>(
            coordinate_dofs(j, static_cast<std::size_t>(d))
            * phi(0, p, j, 0));
      values[p * static_cast<std::size_t>(gdim) + static_cast<std::size_t>(d)]
          = value;
    }
  }
  return values;
}

template <std::floating_point T>
std::vector<double> affine_background_facet_normal(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh, std::int32_t cell,
    std::int32_t local_facet, double tolerance)
{
  if (!mesh)
    throw std::runtime_error("conormal requires a background mesh.");

  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();
  if (tdim != gdim || (tdim != 2 && tdim != 3))
  {
    throw std::runtime_error(
        "conormal currently supports affine background facets in 2D or 3D "
        "meshes with gdim == tdim.");
  }

  const basix::cell::type btype
      = dolfinx::mesh::cell_type_to_basix_type(mesh->topology()->cell_type());
  const auto topology = basix::cell::topology(btype);
  const auto [geometry, geometry_shape] = basix::cell::geometry<double>(btype);
  const auto& facets = topology[static_cast<std::size_t>(tdim - 1)];
  if (local_facet < 0
      || static_cast<std::size_t>(local_facet) >= facets.size())
  {
    throw std::runtime_error("conormal received an invalid local facet index.");
  }

  const auto& facet_vertices = facets[static_cast<std::size_t>(local_facet)];
  std::vector<double> facet_reference;
  facet_reference.reserve(facet_vertices.size() * static_cast<std::size_t>(tdim));
  for (int vertex : facet_vertices)
  {
    for (int d = 0; d < tdim; ++d)
    {
      facet_reference.push_back(
          geometry[static_cast<std::size_t>(vertex)
                       * static_cast<std::size_t>(tdim)
                   + static_cast<std::size_t>(d)]);
    }
  }

  std::vector<double> cell_reference;
  const std::size_t num_cell_vertices = geometry_shape[0];
  cell_reference.reserve(num_cell_vertices * static_cast<std::size_t>(tdim));
  for (std::size_t vertex = 0; vertex < num_cell_vertices; ++vertex)
  {
    for (int d = 0; d < tdim; ++d)
    {
      cell_reference.push_back(
          geometry[vertex * static_cast<std::size_t>(tdim)
                   + static_cast<std::size_t>(d)]);
    }
  }

  const dolfinx::fem::CoordinateElement<T>& cmap
      = mesh->geometry().cmaps().front();
  auto x_dofmap = mesh->geometry().dofmaps().front();
  const std::size_t num_coordinate_dofs = cmap.dim();
  if (cell < 0 || static_cast<std::size_t>(cell) >= x_dofmap.extent(0))
    throw std::runtime_error("conormal received an invalid parent cell.");
  if (x_dofmap.extent(1) != num_coordinate_dofs)
  {
    throw std::runtime_error(
        "Mesh geometry dofmap does not match coordinate element dimension.");
  }

  std::span<const T> x = mesh->geometry().x();
  std::vector<T> coordinate_dofs_storage(
      num_coordinate_dofs * static_cast<std::size_t>(gdim));
  for (std::size_t i = 0; i < num_coordinate_dofs; ++i)
  {
    const std::int32_t geometry_dof = x_dofmap(cell, i);
    for (int d = 0; d < gdim; ++d)
    {
      coordinate_dofs_storage[i * static_cast<std::size_t>(gdim)
                              + static_cast<std::size_t>(d)]
          = x[3 * static_cast<std::size_t>(geometry_dof)
              + static_cast<std::size_t>(d)];
    }
  }

  const auto facet_points = physical_reference_points<T>(
      cmap, coordinate_dofs_storage, gdim, tdim,
      std::span<const double>(facet_reference.data(), facet_reference.size()));
  const auto cell_points = physical_reference_points<T>(
      cmap, coordinate_dofs_storage, gdim, tdim,
      std::span<const double>(cell_reference.data(), cell_reference.size()));

  std::vector<double> normal(static_cast<std::size_t>(gdim), 0.0);
  if (tdim == 2)
  {
    const double tx = facet_points[2] - facet_points[0];
    const double ty = facet_points[3] - facet_points[1];
    normal = {ty, -tx};
  }
  else
  {
    const double ax = facet_points[3] - facet_points[0];
    const double ay = facet_points[4] - facet_points[1];
    const double az = facet_points[5] - facet_points[2];
    const double bx = facet_points[6] - facet_points[0];
    const double by = facet_points[7] - facet_points[1];
    const double bz = facet_points[8] - facet_points[2];
    normal = {ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx};
  }
  normal = checked_unit_vector(std::move(normal), tolerance,
                               "conormal background facet normal");

  std::vector<double> facet_centroid(static_cast<std::size_t>(gdim), 0.0);
  const std::size_t num_facet_vertices = facet_vertices.size();
  for (std::size_t i = 0; i < num_facet_vertices; ++i)
    for (int d = 0; d < gdim; ++d)
      facet_centroid[static_cast<std::size_t>(d)]
          += facet_points[i * static_cast<std::size_t>(gdim)
                          + static_cast<std::size_t>(d)]
             / static_cast<double>(num_facet_vertices);

  std::vector<double> cell_centroid(static_cast<std::size_t>(gdim), 0.0);
  for (std::size_t i = 0; i < num_cell_vertices; ++i)
    for (int d = 0; d < gdim; ++d)
      cell_centroid[static_cast<std::size_t>(d)]
          += cell_points[i * static_cast<std::size_t>(gdim)
                         + static_cast<std::size_t>(d)]
             / static_cast<double>(num_cell_vertices);

  double inward_test = 0.0;
  for (int d = 0; d < gdim; ++d)
    inward_test += normal[static_cast<std::size_t>(d)]
                 * (cell_centroid[static_cast<std::size_t>(d)]
                    - facet_centroid[static_cast<std::size_t>(d)]);
  if (inward_test > 0.0)
    for (double& component : normal)
      component = -component;

  return normal;
}
} // namespace detail

/// Evaluate surface conormals from quadrature-point surface normals.
///
/// The surface normal can be high order and is supplied pointwise. Only the
/// background facet normal is assumed affine/constant on each side facet.
template <std::floating_point T>
std::vector<double> evaluate_conormals(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const double> surface_normals, std::size_t num_points,
    std::size_t gdim, std::span<const std::int32_t> offsets,
    std::span<const std::int32_t> parent_map,
    std::span<const std::int32_t> local_facets, double tolerance)
{
  if (offsets.size() != parent_map.size() + 1)
    throw std::runtime_error("conormal expects offsets.size() == parent_map.size() + 1.");
  if (local_facets.size() != parent_map.size())
    throw std::runtime_error("conormal local facet data is not aligned with rules.");
  if (surface_normals.size() != num_points * gdim)
    throw std::runtime_error("conormal surface-normal array has inconsistent shape.");
  if (tolerance <= 0.0)
    throw std::runtime_error("conormal tolerance must be positive.");

  std::vector<double> values(num_points * gdim, 0.0);
  for (std::size_t rule = 0; rule < parent_map.size(); ++rule)
  {
    const std::int32_t q0 = offsets[rule];
    const std::int32_t q1 = offsets[rule + 1];
    if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
      throw std::runtime_error("conormal offsets are out of range.");

    const auto n_facet = detail::affine_background_facet_normal<T>(
        mesh, parent_map[rule], local_facets[rule], tolerance);

    for (std::int32_t q = q0; q < q1; ++q)
    {
      std::vector<double> m(gdim, 0.0);
      for (std::size_t d = 0; d < gdim; ++d)
        m[d] = surface_normals[static_cast<std::size_t>(q) * gdim + d];
      m = detail::checked_unit_vector(
          std::move(m), tolerance,
          "conormal surface normal at quadrature point " + std::to_string(q));

      double dot = 0.0;
      for (std::size_t d = 0; d < gdim; ++d)
        dot += n_facet[d] * m[d];

      std::vector<double> projected(gdim, 0.0);
      for (std::size_t d = 0; d < gdim; ++d)
        projected[d] = n_facet[d] - dot * m[d];
      projected = detail::checked_unit_vector(
          std::move(projected), tolerance,
          "conormal projected facet normal at quadrature point "
              + std::to_string(q));

      for (std::size_t d = 0; d < gdim; ++d)
        values[static_cast<std::size_t>(q) * gdim + d] = projected[d];
    }
  }
  return values;
}
} // namespace cutfemx::geometry
