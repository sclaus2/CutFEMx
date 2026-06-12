// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cutcells/cell_types.h>
#include <cutcells/mapping.h>
#include <cutcells/mesh_view.h>

#include "reference_cell.h"

#include <cutfemx/cut/cut.h>
#include <cutfemx/cut/runtime_quadrature.h>
#include <cutfemx/level_set/normal.h>

namespace cutfemx::geometry
{
namespace
{
template <std::floating_point T, std::integral I>
std::vector<T> parent_cell_vertex_coords_basix(
    const cutcells::MeshView<T, I>& mesh, I cell_id)
{
  const auto ctype = mesh.cell_type(cell_id);
  const int nv = cutcells::cell::get_num_vertices(ctype);
  std::vector<T> coords(static_cast<std::size_t>(nv * mesh.gdim), T(0));
  std::vector<I> cell_node_scratch;
  const auto parent_nodes = mesh.cell_nodes(cell_id, cell_node_scratch);

  for (int basix_v = 0; basix_v < nv; ++basix_v)
  {
    const int local_v = mesh.vtk_vertex_order
                            ? cutcells::cell::basix_to_vtk_vertex(ctype, basix_v)
                            : basix_v;
    const I node_id = parent_nodes[static_cast<std::size_t>(local_v)];
    const T* x = mesh.node(node_id);
    for (int d = 0; d < mesh.gdim; ++d)
      coords[static_cast<std::size_t>(basix_v * mesh.gdim + d)] = x[d];
  }
  return coords;
}

template <std::floating_point T>
std::vector<T> entity_reference_coords(
    const cutcells::AdaptCell<T>& adapt_cell,
    std::span<const std::int32_t> entity_vertices)
{
  std::vector<T> coords(
      static_cast<std::size_t>(entity_vertices.size() * adapt_cell.tdim), T(0));
  for (std::size_t j = 0; j < entity_vertices.size(); ++j)
  {
    const std::int32_t vertex = entity_vertices[j];
    for (int d = 0; d < adapt_cell.tdim; ++d)
    {
      coords[static_cast<std::size_t>(j * adapt_cell.tdim + d)]
          = adapt_cell.vertex_coords[static_cast<std::size_t>(
              vertex * adapt_cell.tdim + d)];
    }
  }
  return coords;
}

template <std::floating_point T>
std::vector<std::int32_t> zero_entity_vertices(
    const cutcells::AdaptCell<T>& adapt_cell, std::int32_t zero_entity_id)
{
  if (zero_entity_id < 0
      || static_cast<std::size_t>(zero_entity_id)
             >= adapt_cell.zero_entity_dim.size())
  {
    throw std::runtime_error("Surface-normal provenance references an invalid "
                             "zero entity.");
  }

  const int dim
      = adapt_cell.zero_entity_dim[static_cast<std::size_t>(zero_entity_id)];
  const std::int32_t entity_id
      = adapt_cell.zero_entity_id[static_cast<std::size_t>(zero_entity_id)];
  if (dim == 0)
    return {entity_id};

  auto vertices = adapt_cell.entity_to_vertex[dim][entity_id];
  return std::vector<std::int32_t>(vertices.begin(), vertices.end());
}

std::vector<double> checked_unit(std::vector<double> value,
                                 const std::string& label)
{
  double norm = 0.0;
  for (double component : value)
    norm += component * component;
  norm = std::sqrt(norm);
  if (norm < 1.0e-14)
    throw std::runtime_error(label + " is degenerate.");
  for (double& component : value)
    component /= norm;
  return value;
}

std::vector<double> geometric_surface_normal(std::span<const double> x,
                                             int tdim, int gdim,
                                             std::size_t rule)
{
  if (tdim != gdim || (tdim != 2 && tdim != 3))
  {
    throw std::runtime_error(
        "surface_normal currently supports codimension-one cuts in 2D or 3D "
        "meshes with gdim == tdim.");
  }

  if (tdim == 2)
  {
    if (x.size() < 4)
    {
      throw std::runtime_error(
          "surface_normal encountered an interval with fewer than two vertices.");
    }
    const double tx = x[2] - x[0];
    const double ty = x[3] - x[1];
    return checked_unit({ty, -tx},
                        "surface_normal rule " + std::to_string(rule));
  }

  if (x.size() < 9)
  {
    throw std::runtime_error(
        "surface_normal encountered a surface entity with fewer than three "
        "vertices.");
  }
  const double ax = x[3] - x[0];
  const double ay = x[4] - x[1];
  const double az = x[5] - x[2];
  const double bx = x[6] - x[0];
  const double by = x[7] - x[1];
  const double bz = x[8] - x[2];
  return checked_unit({ay * bz - az * by, az * bx - ax * bz,
                       ax * by - ay * bx},
                      "surface_normal rule " + std::to_string(rule));
}
} // namespace

/// Evaluate the geometric normal of the selected cut surface.
///
/// The runtime quadrature object supplies one provenance record per rule slice.
/// The returned values are flattened row-major with shape (num_points, gdim).
template <std::floating_point T>
std::vector<double> evaluate_surface_normals(
    const CutData<T>& cut_data, const RuntimeQuadrature<T>& quadrature,
    std::int32_t level_set_index, std::span<const T> points,
    std::size_t num_points, std::size_t point_dim,
    std::span<const std::int32_t> offsets,
    std::span<const std::int32_t> parent_map)
{
  const RuntimeSurfaceProvenance& provenance = quadrature.surface_provenance;
  if (provenance.empty())
  {
    throw std::runtime_error(
        "surface_normal requires straight codimension-one runtime quadrature "
        "built from a single equality selector such as 'phi=0'.");
  }
  if (provenance.level_set_index < 0
      || static_cast<std::size_t>(provenance.level_set_index)
             >= cut_data.level_set_owners.size())
  {
    throw std::runtime_error(
        "surface_normal provenance does not identify a valid orienting level set.");
  }
  if (provenance.level_set_index != level_set_index)
  {
    throw std::runtime_error(
        "surface_normal was evaluated on runtime quadrature built for a "
        "different level-set selector.");
  }
  if (offsets.size() != parent_map.size() + 1)
  {
    throw std::runtime_error(
        "surface_normal expects offsets.size() == parent_map.size() + 1.");
  }
  if (provenance.size() != parent_map.size())
  {
    throw std::runtime_error(
        "surface_normal provenance is not aligned with runtime quadrature rules.");
  }
  if (points.size() != num_points * point_dim)
    throw std::runtime_error("surface_normal point array has inconsistent shape.");
  if (static_cast<int>(point_dim) != cut_data.tdim)
  {
    throw std::runtime_error(
        "surface_normal points must have parent-cell reference dimension.");
  }

  const int tdim = cut_data.tdim;
  const int gdim = cut_data.gdim;
  std::vector<double> level_normals = level_set::evaluate_normals<T>(
      cut_data.level_set_owners[static_cast<std::size_t>(
          provenance.level_set_index)],
      points, num_points, point_dim, offsets, parent_map, 1.0);
  std::vector<double> values(num_points * static_cast<std::size_t>(gdim), 0.0);

  for (std::size_t rule = 0; rule < parent_map.size(); ++rule)
  {
    const std::int32_t cut_cell_id = provenance.cut_cell_ids[rule];
    if (cut_cell_id < 0
        || static_cast<std::size_t>(cut_cell_id)
               >= cut_data.cut_cells.adapt_cells.size())
    {
      throw std::runtime_error(
          "surface_normal provenance references an invalid cut cell.");
    }
    if (provenance.dimensions[rule] != tdim - 1)
    {
      throw std::runtime_error(
          "surface_normal provenance is not codimension one.");
    }

    const auto& adapt_cell
        = cut_data.cut_cells.adapt_cells[static_cast<std::size_t>(cut_cell_id)];
    const auto vertices = zero_entity_vertices(
        adapt_cell, provenance.local_zero_entity_ids[rule]);
    const auto ref_coords = entity_reference_coords(
        adapt_cell, std::span<const std::int32_t>(vertices.data(), vertices.size()));
    const std::int32_t host_parent_cell = provenance.parent_cell_ids[rule];
    const auto parent_vertex_coords
        = parent_cell_vertex_coords_basix(cut_data.mesh_view, host_parent_cell);
    const auto phys_coords = cutcells::cell::push_forward_affine_map<T>(
        adapt_cell.parent_cell_type, parent_vertex_coords, gdim,
        std::span<const T>(ref_coords.data(), ref_coords.size()));

    std::vector<double> normal(phys_coords.begin(), phys_coords.end());
    normal = geometric_surface_normal(
        std::span<const double>(normal.data(), normal.size()), tdim, gdim, rule);

    const std::int32_t q0 = offsets[rule];
    const std::int32_t q1 = offsets[rule + 1];
    if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
      throw std::runtime_error("surface_normal offsets are out of range.");
    if (q0 == q1)
      continue;

    double orient = 0.0;
    for (int d = 0; d < gdim; ++d)
    {
      orient += normal[static_cast<std::size_t>(d)]
              * level_normals[static_cast<std::size_t>(q0 * gdim + d)];
    }
    if (orient < 0.0)
    {
      for (double& component : normal)
        component = -component;
    }

    for (std::int32_t q = q0; q < q1; ++q)
    {
      for (int d = 0; d < gdim; ++d)
      {
        values[static_cast<std::size_t>(q * gdim + d)]
            = normal[static_cast<std::size_t>(d)];
      }
    }
  }
  return values;
}
} // namespace cutfemx::geometry
