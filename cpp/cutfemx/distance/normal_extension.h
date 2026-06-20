// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "fast_iterative.h"
#include "point_triangle_distance.h"
#include "vertex_map.h"

#include <cutfemx/level_set/normal.h>
#include <cutfemx/mesh/cut_mesh.h>

#include <basix/mdspan.hpp>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace cutfemx::distance
{
namespace detail
{
template <typename Real>
using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    Real, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

template <typename Real>
using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const Real, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

template <typename Real>
using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const Real, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

template <typename Real>
std::vector<Real> coordinate_dofs(const dolfinx::mesh::Mesh<Real>& mesh,
                                  std::int32_t cell)
{
  const int gdim = mesh.geometry().dim();
  const auto& cmap = mesh.geometry().cmaps().front();
  const std::size_t num_coordinate_dofs = cmap.dim();
  auto x_dofmap = mesh.geometry().dofmaps().front();
  std::span<const Real> x = mesh.geometry().x();

  std::vector<Real> out(num_coordinate_dofs * static_cast<std::size_t>(gdim));
  for (std::size_t i = 0; i < num_coordinate_dofs; ++i)
  {
    const std::int32_t geometry_dof = x_dofmap(cell, i);
    for (int d = 0; d < gdim; ++d)
    {
      out[i * static_cast<std::size_t>(gdim) + static_cast<std::size_t>(d)]
          = x[3 * static_cast<std::size_t>(geometry_dof)
              + static_cast<std::size_t>(d)];
    }
  }
  return out;
}

template <typename Real>
std::vector<Real> pull_back_to_cell(const dolfinx::mesh::Mesh<Real>& mesh,
                                    std::int32_t cell,
                                    std::span<const Real> physical_point)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology()->dim();
  const auto& cmap = mesh.geometry().cmaps().front();
  const std::size_t num_coordinate_dofs = cmap.dim();

  std::vector<Real> coordinate_dofs_storage = coordinate_dofs(mesh, cell);
  cmdspan2_t<Real> cell_geometry(coordinate_dofs_storage.data(),
                                 num_coordinate_dofs,
                                 static_cast<std::size_t>(gdim));

  std::vector<Real> reference_point(static_cast<std::size_t>(tdim), Real(0));
  mdspan2_t<Real> X_ref(reference_point.data(), std::size_t(1),
                        static_cast<std::size_t>(tdim));
  cmdspan2_t<Real> x_phys(physical_point.data(), std::size_t(1),
                          static_cast<std::size_t>(gdim));

  if (cmap.is_affine())
  {
    const std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
    std::vector<Real> phi_storage(
        std::reduce(phi_shape.begin(), phi_shape.end(), std::size_t(1),
                    std::multiplies<std::size_t>{}));
    std::vector<Real> origin(static_cast<std::size_t>(tdim), Real(0));
    cmap.tabulate(1, origin, {1, static_cast<std::size_t>(tdim)},
                  std::span<Real>(phi_storage.data(), phi_storage.size()));
    cmdspan4_t<Real> phi(phi_storage.data(), phi_shape);
    auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        phi, std::pair(1, tdim + 1), 0,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

    std::vector<Real> J_storage(static_cast<std::size_t>(gdim * tdim), Real(0));
    std::vector<Real> K_storage(static_cast<std::size_t>(tdim * gdim), Real(0));
    mdspan2_t<Real> J(J_storage.data(), static_cast<std::size_t>(gdim),
                      static_cast<std::size_t>(tdim));
    mdspan2_t<Real> K(K_storage.data(), static_cast<std::size_t>(tdim),
                      static_cast<std::size_t>(gdim));
    dolfinx::fem::CoordinateElement<Real>::compute_jacobian(dphi, cell_geometry,
                                                           J);
    dolfinx::fem::CoordinateElement<Real>::compute_jacobian_inverse(J, K);

    std::array<Real, 3> x0 = {0, 0, 0};
    for (int d = 0; d < gdim; ++d)
      x0[static_cast<std::size_t>(d)] = cell_geometry(0, d);
    dolfinx::fem::CoordinateElement<Real>::pull_back_affine(X_ref, K, x0,
                                                            x_phys);
  }
  else
  {
    cmap.pull_back_nonaffine(X_ref, x_phys, cell_geometry, 1.0e-12, 15);
  }

  return reference_point;
}
} // namespace detail

template <typename Real>
struct InterfaceSurface
{
  std::vector<Real> X;
  std::vector<std::int32_t> conn;
  std::vector<std::int32_t> parent_cell;
  int verts_per_elem = 2;

  std::int32_t nV() const
  {
    return static_cast<std::int32_t>(X.size() / 3);
  }
  std::int32_t nE() const
  {
    return verts_per_elem == 0
               ? 0
               : static_cast<std::int32_t>(conn.size() / verts_per_elem);
  }
};

template <typename Real>
InterfaceSurface<Real> interface_surface_from_cut_mesh(
    const cutfemx::mesh::CutMesh<Real>& interface_cut_mesh)
{
  if (!interface_cut_mesh._cut_mesh)
    throw std::invalid_argument("normal extension requires a non-empty cut mesh.");

  auto mesh = interface_cut_mesh._cut_mesh;
  const int tdim = mesh->topology()->dim();
  if (tdim != 1 && tdim != 2)
    throw std::invalid_argument("normal extension supports interval or triangle interfaces.");

  InterfaceSurface<Real> surface;
  surface.verts_per_elem = tdim + 1;

  mesh->topology_mutable()->create_connectivity(tdim, 0);
  auto cell_to_vertex = mesh->topology()->connectivity(tdim, 0);
  auto cell_map = mesh->topology()->index_map(tdim);
  auto vertex_map = mesh->topology()->index_map(0);
  if (!cell_to_vertex || !cell_map || !vertex_map)
    throw std::runtime_error("Interface cut mesh topology is incomplete.");

  const std::int32_t num_cells
      = static_cast<std::int32_t>(cell_map->size_local() + cell_map->num_ghosts());
  const std::int32_t num_vertices = static_cast<std::int32_t>(
      vertex_map->size_local() + vertex_map->num_ghosts());
  if (interface_cut_mesh._parent_index.size()
      != static_cast<std::size_t>(num_cells))
    throw std::runtime_error("Interface cut mesh parent map has inconsistent size.");

  surface.X.assign(static_cast<std::size_t>(3 * num_vertices), Real(0));
  const std::span<const Real> x = mesh->geometry().x();
  auto x_dofmap = mesh->geometry().dofmaps().front();
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    auto vertices = cell_to_vertex->links(c);
    if (vertices.size() != static_cast<std::size_t>(surface.verts_per_elem))
      throw std::runtime_error("Interface cut mesh has an unexpected cell arity.");

    for (std::size_t i = 0; i < vertices.size(); ++i)
    {
      const std::int32_t v = vertices[i];
      const std::int32_t gdof = x_dofmap(c, i);
      for (int d = 0; d < mesh->geometry().dim(); ++d)
      {
        surface.X[static_cast<std::size_t>(3 * v + d)]
            = x[static_cast<std::size_t>(3 * gdof + d)];
      }
      surface.conn.push_back(v);
    }
    surface.parent_cell.push_back(interface_cut_mesh._parent_index[c]);
  }

  return surface;
}

template <typename Real>
Real point_interface_distance(const Real* p, const InterfaceSurface<Real>& surface,
                              std::int32_t elem, Real* closest)
{
  if (surface.verts_per_elem == 2)
  {
    const std::int32_t v0 = surface.conn[2 * elem];
    const std::int32_t v1 = surface.conn[2 * elem + 1];
    return point_segment_distance(p, &surface.X[3 * v0], &surface.X[3 * v1],
                                  closest);
  }

  const std::int32_t v0 = surface.conn[3 * elem];
  const std::int32_t v1 = surface.conn[3 * elem + 1];
  const std::int32_t v2 = surface.conn[3 * elem + 2];
  return point_triangle_distance(p, &surface.X[3 * v0], &surface.X[3 * v1],
                                 &surface.X[3 * v2], closest);
}

template <typename Real>
std::vector<std::int32_t> expand_cells_k_ring(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<const std::int32_t> seed_cells, int k_ring)
{
  auto topology = mesh.topology_mutable();
  const int tdim = topology->dim();
  const int fdim = tdim - 1;
  topology->create_connectivity(tdim, fdim);
  topology->create_connectivity(fdim, tdim);

  auto cell_map = topology->index_map(tdim);
  const std::int32_t num_cells = static_cast<std::int32_t>(
      cell_map->size_local() + cell_map->num_ghosts());
  std::vector<std::uint8_t> visited(static_cast<std::size_t>(num_cells), 0);
  std::vector<std::int32_t> all;
  std::vector<std::int32_t> frontier;

  for (std::int32_t c : seed_cells)
  {
    if (c >= 0 && c < num_cells && !visited[static_cast<std::size_t>(c)])
    {
      visited[static_cast<std::size_t>(c)] = 1;
      frontier.push_back(c);
      all.push_back(c);
    }
  }

  auto c2f = topology->connectivity(tdim, fdim);
  auto f2c = topology->connectivity(fdim, tdim);
  for (int ring = 0; ring < k_ring; ++ring)
  {
    std::vector<std::int32_t> next;
    for (std::int32_t c : frontier)
    {
      for (std::int32_t f : c2f->links(c))
      {
        for (std::int32_t n : f2c->links(f))
        {
          if (n >= 0 && n < num_cells && !visited[static_cast<std::size_t>(n)])
          {
            visited[static_cast<std::size_t>(n)] = 1;
            next.push_back(n);
            all.push_back(n);
          }
        }
      }
    }
    frontier = std::move(next);
    if (frontier.empty())
      break;
  }

  return all;
}

template <typename Real>
Real evaluate_interface_speed(
    const dolfinx::fem::Function<Real>& interface_speed,
    std::int32_t interface_cell, std::span<const Real> point)
{
  std::array<Real, 3> p{Real(0), Real(0), Real(0)};
  for (std::size_t d = 0; d < point.size() && d < 3; ++d)
    p[d] = point[d];

  std::array<std::int32_t, 1> cells{interface_cell};
  std::array<Real, 1> values{Real(0)};
  interface_speed.eval(std::span<const Real>(p.data(), 3), {1, 3},
                       std::span<const std::int32_t>(cells.data(), 1),
                       std::span<Real>(values.data(), 1), {1, 1}, 1.0e-8, 15);
  return values[0];
}

template <typename Real>
std::vector<double> evaluate_level_set_normal_at_point(
    std::shared_ptr<const dolfinx::fem::Function<Real>> phi,
    std::int32_t parent_cell, std::span<const Real> point, double sign)
{
  auto mesh = phi->function_space()->mesh();
  const int tdim = mesh->topology()->dim();
  std::vector<Real> reference_point = detail::pull_back_to_cell(
      *mesh, parent_cell, point);
  std::array<std::int32_t, 2> offsets{0, 1};
  std::array<std::int32_t, 1> parents{parent_cell};
  return cutfemx::level_set::evaluate_normals<Real>(
      std::move(phi), std::span<const Real>(reference_point.data(),
                                            static_cast<std::size_t>(tdim)),
      1, static_cast<std::size_t>(tdim),
      std::span<const std::int32_t>(offsets.data(), offsets.size()),
      std::span<const std::int32_t>(parents.data(), parents.size()), sign);
}

template <typename Real>
void extend_normal_velocity(
    std::shared_ptr<const dolfinx::fem::Function<Real>> phi,
    const dolfinx::fem::Function<Real>& interface_speed,
    const cutfemx::mesh::CutMesh<Real>& interface_cut_mesh,
    dolfinx::fem::Function<Real>& speed,
    dolfinx::fem::Function<Real>& velocity,
    dolfinx::fem::Function<Real>& signed_distance,
    int k_ring, double normal_sign, const FMMOptions& opt)
{
  if (!phi)
    throw std::invalid_argument("normal extension requires a level-set function.");
  auto mesh = phi->function_space()->mesh();
  if (speed.function_space()->mesh() != mesh
      || velocity.function_space()->mesh() != mesh
      || signed_distance.function_space()->mesh() != mesh)
    throw std::invalid_argument("normal extension output fields must use phi's mesh.");
  if (!interface_cut_mesh._cut_mesh
      || interface_speed.function_space()->mesh() != interface_cut_mesh._cut_mesh)
    throw std::invalid_argument(
        "interface_speed must be defined on the supplied interface cut mesh.");

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology()->dim();
  if (tdim != 2 && tdim != 3)
    throw std::invalid_argument("normal extension supports 2D and 3D meshes.");

  InterfaceSurface<Real> surface
      = interface_surface_from_cut_mesh(interface_cut_mesh);
  if (surface.nE() == 0)
    throw std::runtime_error("normal extension received an empty interface.");

  auto topology = mesh->topology_mutable();
  topology->create_connectivity(tdim, 0);
  topology->create_connectivity(0, tdim);
  auto cell_to_vertex = topology->connectivity(tdim, 0);
  auto vertex_map = topology->index_map(0);
  const std::int32_t num_vertices = static_cast<std::int32_t>(
      vertex_map->size_local() + vertex_map->num_ghosts());

  VertexMapCache<Real> vmap;
  vmap.build(*mesh, speed.function_space().get());
  VertexMapCache<Real> dist_vmap;
  dist_vmap.build(*mesh, signed_distance.function_space().get());
  VertexMapCache<Real> velocity_vmap;
  velocity_vmap.build(*mesh, velocity.function_space().get());
  VertexMapCache<Real> phi_vmap;
  phi_vmap.build(*mesh, phi->function_space().get());

  std::vector<Real> dist(static_cast<std::size_t>(num_vertices), opt.inf_value);
  std::vector<Real> payload_speed(static_cast<std::size_t>(num_vertices), Real(0));
  std::vector<Real> payload_normal(
      static_cast<std::size_t>(num_vertices) * static_cast<std::size_t>(gdim),
      Real(0));

  std::vector<std::int32_t> seed_cells = surface.parent_cell;
  std::sort(seed_cells.begin(), seed_cells.end());
  seed_cells.erase(std::unique(seed_cells.begin(), seed_cells.end()),
                   seed_cells.end());
  std::vector<std::int32_t> near_cells
      = expand_cells_k_ring(*mesh, seed_cells, std::max(0, k_ring));

  for (std::int32_t cell : near_cells)
  {
    auto vertices = cell_to_vertex->links(cell);
    for (std::int32_t v : vertices)
    {
      if (!vmap.has_geom(v))
        continue;
      const Real* p = vmap.coords(v);

      for (std::int32_t elem = 0; elem < surface.nE(); ++elem)
      {
        Real closest[3] = {Real(0), Real(0), Real(0)};
        Real d = point_interface_distance(p, surface, elem, closest);
        if (d < dist[static_cast<std::size_t>(v)])
        {
          dist[static_cast<std::size_t>(v)] = d;
          std::span<const Real> closest_span(closest, static_cast<std::size_t>(gdim));
          payload_speed[static_cast<std::size_t>(v)]
              = evaluate_interface_speed(interface_speed, elem, closest_span);

          std::vector<double> normal = evaluate_level_set_normal_at_point(
              phi, surface.parent_cell[static_cast<std::size_t>(elem)],
              closest_span, normal_sign);
          for (int dcomp = 0; dcomp < gdim; ++dcomp)
          {
            payload_normal[static_cast<std::size_t>(v) * gdim
                           + static_cast<std::size_t>(dcomp)]
                = static_cast<Real>(normal[static_cast<std::size_t>(dcomp)]);
          }
        }
      }
    }
  }

  std::vector<std::int32_t> seeds;
  const std::int32_t num_owned = vertex_map->size_local();
  for (std::int32_t v = 0; v < num_owned; ++v)
  {
    if (dist[static_cast<std::size_t>(v)] < opt.inf_value)
      seeds.push_back(v);
  }
  int local_num_seeds = static_cast<int>(seeds.size());
  int global_num_seeds = 0;
  MPI_Allreduce(&local_num_seeds, &global_num_seeds, 1, MPI_INT, MPI_SUM,
                mesh->comm());
  if (global_num_seeds == 0)
    throw std::runtime_error("normal extension found no near-field seeds.");

  FIMTransportPayload<Real> payload{
      std::span<Real>(payload_speed.data(), payload_speed.size()),
      std::span<Real>(payload_normal.data(), payload_normal.size()), gdim};
  compute_distance_fim(*mesh, std::span<Real>(dist.data(), dist.size()), seeds,
                       opt, &payload);

  std::span<const Real> phi_values = phi->x()->array();
  std::span<Real> speed_values = speed.x()->array();
  std::span<Real> distance_values = signed_distance.x()->array();
  std::span<Real> velocity_values = velocity.x()->array();
  const int velocity_bs = velocity.function_space()->dofmap()->index_map_bs();
  if (velocity_bs != gdim)
    throw std::invalid_argument("velocity output must be a blocked vector space.");

  for (std::int32_t v = 0; v < num_vertices; ++v)
  {
    const std::int32_t speed_dof = vmap.vert_to_dof[v];
    const std::int32_t dist_dof = dist_vmap.vert_to_dof[v];
    const std::int32_t vel_dof = velocity_vmap.vert_to_dof[v];
    const std::int32_t phi_dof = phi_vmap.vert_to_dof[v];
    if (speed_dof >= 0 && speed_dof < static_cast<std::int32_t>(speed_values.size()))
      speed_values[static_cast<std::size_t>(speed_dof)]
          = payload_speed[static_cast<std::size_t>(v)];

    if (dist_dof >= 0 && dist_dof < static_cast<std::int32_t>(distance_values.size())
        && phi_dof >= 0 && phi_dof < static_cast<std::int32_t>(phi_values.size()))
    {
      const Real sign = phi_values[static_cast<std::size_t>(phi_dof)] >= Real(0)
                            ? Real(1)
                            : Real(-1);
      distance_values[static_cast<std::size_t>(dist_dof)]
          = sign * std::abs(dist[static_cast<std::size_t>(v)]);
    }

    if (vel_dof >= 0)
    {
      for (int d = 0; d < gdim; ++d)
      {
        const std::size_t idx
            = static_cast<std::size_t>(velocity_bs * vel_dof + d);
        if (idx < velocity_values.size())
        {
          velocity_values[idx]
              = payload_speed[static_cast<std::size_t>(v)]
                * payload_normal[static_cast<std::size_t>(v) * gdim
                                 + static_cast<std::size_t>(d)];
        }
      }
    }
  }

  speed.x()->scatter_fwd();
  signed_distance.x()->scatter_fwd();
  velocity.x()->scatter_fwd();
}

} // namespace cutfemx::distance
