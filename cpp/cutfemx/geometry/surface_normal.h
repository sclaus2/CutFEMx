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
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cutcells/bernstein.h>
#include <cutcells/level_set_cell.h>
#include <cutcells/lut/cell_pieces.h>
#include <cutcells/lut/piece_rules.h>
#include <cutcells/mesh_view.h>
#include <cutcells/part/mesh_part.h>
#include <cutcells/reference_cell.h>

#include <cutfemx/cut/cut.h>
#include <cutfemx/cut/runtime_quadrature.h>
#include <cutfemx/level_set/normal.h>

namespace cutfemx::geometry
{
namespace detail
{
/// Unit normal of a straight piece of a zero set from its physical vertices:
/// a segment in 2D, a planar triangle or quadrilateral (Basix order) in 3D.
/// Zero for a degenerate piece.
inline std::array<double, 3> piece_normal(std::span<const double> x, int gdim)
{
  std::array<double, 3> n{0.0, 0.0, 0.0};
  if (gdim == 2)
    n = {x[3] - x[1], x[0] - x[2], 0.0};
  else
  {
    // Newell's normal; Basix numbers a quadrilateral 0, 1, 3, 2 around it
    const std::size_t nv = x.size() / 3;
    constexpr std::array<std::size_t, 4> quadrilateral{0, 1, 3, 2};
    for (std::size_t i = 0; i < nv; ++i)
    {
      const std::size_t a = nv == 4 ? quadrilateral[i] : i;
      const std::size_t b = nv == 4 ? quadrilateral[(i + 1) % 4] : (i + 1) % nv;
      const double* p = x.data() + 3 * a;
      const double* q = x.data() + 3 * b;
      n[0] += (p[1] - q[1]) * (p[2] + q[2]);
      n[1] += (p[2] - q[2]) * (p[0] + q[0]);
      n[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
  }
  const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (norm > 0.0)
  {
    for (double& component : n)
      component /= norm;
  }
  return n;
}

/// The rule of one cell, piece by piece: each point with the unit normal of
/// the straight piece it lies on.
template <std::floating_point T>
struct PieceRules
{
  std::vector<T> points;  ///< reference coordinates, tdim per point
  std::vector<T> weights;
  std::vector<std::array<double, 3>> normals; ///< one per point
};

/// Append the rule of a straight piece with reference vertices @p xi.
template <std::floating_point T>
void append_piece(const cutcells::lut::CellMap<T>& map, cutcells::cell::type type,
                  std::span<const T> xi, int degree, std::vector<T>& x,
                  std::vector<double>& physical, PieceRules<T>& out)
{
  cutcells::lut::append_piece_rule(map, type, xi, degree, out.points,
                                   out.weights);
  cutcells::lut::push_forward(map, xi, x);
  physical.assign(x.begin(), x.end());
  out.normals.resize(out.weights.size(),
                     piece_normal(std::span<const double>(physical), map.gdim));
}

/// The rule of @p cell in a part of the lookup tables for a single equality
/// selector on @p level_set, piece by piece as cutcells::part::quadrature_rules
/// makes it: the straight pieces of the cell in the zero set, then the zero
/// faces @p zero_faces that the cell owns.
template <std::floating_point T>
void piece_rules(const CutData<T>& cut_data,
                 const cutcells::part::MeshPart<T, std::int32_t>& part,
                 int level_set, std::span<const int> zero_faces,
                 std::int32_t cell, int degree,
                 const cutcells::lut::CellMap<T>& map,
                 cutcells::LevelSetCell<T, std::int32_t>& scratch,
                 cutcells::lut::Pieces<T>& pieces, PieceRules<T>& out)
{
  out.points.clear();
  out.weights.clear();
  out.normals.clear();
  std::vector<T> x;
  std::vector<double> physical;
  const int tdim = map.tdim();

  if (std::ranges::binary_search(part.cut_cells, cell))
  {
    // the cell's template and the level set at its vertices
    const cutcells::LevelSetFunction<T, std::int32_t>& ls
        = cut_data.level_sets[static_cast<std::size_t>(level_set)];
    const int ls_degree = std::max(1, ls.analytic ? 2 : ls.mesh_data.degree);
    const cutcells::lut::Options& options = cut_data.options.lut;
    const int k = options.template_order > 0
                      ? options.template_order
                      : std::min(ls_degree,
                                 map.type == cutcells::cell::type::pyramid ? 2 : 4);
    const std::span<const double> tv = cutcells::lut::template_vertices(map.type, k);
    const std::vector<T> xi(tv.begin(), tv.end());
    const std::size_t nv = xi.size() / static_cast<std::size_t>(tdim);
    cutcells::make_cell_level_set(ls, cell, scratch);
    std::vector<T> values(nv);
    for (std::size_t v = 0; v < nv; ++v)
    {
      values[v] = cutcells::bernstein::evaluate<T>(
          map.type, scratch.bernstein_order,
          std::span<const T>(scratch.bernstein_coeffs),
          std::span<const T>(xi).subspan(v * tdim, tdim));
    }
    cutcells::lut::cut_cell<T>(
        map.type, k, std::span<const T>(values), 1, /*zero_sets=*/1,
        /*curves=*/false,
        options.triangulate ? options.triangulation
                            : cutcells::cell::TriangulationStrategy::none,
        pieces);
    for (int p = 0; p < pieces.n_pieces(); ++p)
    {
      if (pieces.zero[static_cast<std::size_t>(p)] != 1)
        continue;
      const std::size_t begin
          = static_cast<std::size_t>(pieces.offsets[p] * tdim);
      const std::size_t size = static_cast<std::size_t>(
          (pieces.offsets[p + 1] - pieces.offsets[p]) * tdim);
      append_piece(map, pieces.types[static_cast<std::size_t>(p)],
                   std::span<const T>(pieces.vertices).subspan(begin, size),
                   degree, x, physical, out);
    }
  }

  const std::vector<T> reference = cutcells::cell::reference_vertices<T>(map.type);
  std::vector<T> facet;
  for (const int z : zero_faces)
  {
    const int f = cut_data.result.zero_face_local[static_cast<std::size_t>(z)];
    facet.clear();
    for (const int v : cutcells::part::facet_vertices(map.type, f))
    {
      facet.insert(facet.end(), reference.begin() + v * tdim,
                   reference.begin() + (v + 1) * tdim);
    }
    append_piece(map, cutcells::part::facet_type(map.type, f),
                 std::span<const T>(facet), degree, x, physical, out);
  }
}
} // namespace detail

/// Evaluate the geometric normal of the selected cut surface.
///
/// The runtime rules come from the lookup tables (backend "straight") for a
/// single equality selector such as "phi=0": their points lie on straight
/// pieces of the zero set, one rule per cell. The rule of each cell is made
/// again piece by piece, and must give the same points; each point takes the
/// unit normal of its piece, oriented along the level-set gradient there.
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
  if (points.size() != num_points * point_dim)
    throw std::runtime_error("surface_normal point array has inconsistent shape.");
  if (static_cast<int>(point_dim) != cut_data.tdim)
  {
    throw std::runtime_error(
        "surface_normal points must have parent-cell reference dimension.");
  }

  const int tdim = cut_data.tdim;
  const int gdim = cut_data.gdim;
  if (tdim != gdim || (tdim != 2 && tdim != 3))
  {
    throw std::runtime_error(
        "surface_normal currently supports codimension-one cuts in 2D or 3D "
        "meshes with gdim == tdim.");
  }

  std::vector<double> level_normals = level_set::evaluate_normals<T>(
      cut_data.level_set_owners[static_cast<std::size_t>(
          provenance.level_set_index)],
      points, num_points, point_dim, offsets, parent_map, 1.0);
  std::vector<double> values(num_points * static_cast<std::size_t>(gdim), 0.0);
  if (parent_map.empty())
    return values;

  const auto part = select_part(cut_data, provenance.selector);
  // the order of the rules is the degree they integrate exactly
  const int degree = provenance.order;

  // The zero faces of the part by owning host cell; rules name background
  // cells, which the host cells of an entity cut map to.
  std::unordered_map<std::int32_t, std::vector<int>> zero_faces;
  for (const int z : part.zero_faces)
    zero_faces[cut_data.result.zero_face_cells[static_cast<std::size_t>(z)]].push_back(z);
  std::unordered_map<std::int32_t, std::int32_t> host_cell;
  for (std::size_t h = 0; h < cut_data.parent_entities.size(); ++h)
    host_cell.emplace(cut_data.parent_entities[h], static_cast<std::int32_t>(h));

  cutcells::lut::CellMap<T> map;
  cutcells::LevelSetCell<T, std::int32_t> scratch;
  cutcells::lut::Pieces<T> pieces;
  detail::PieceRules<T> rules;
  std::vector<std::int32_t> node_scratch;
  const T tolerance = T(64) * std::numeric_limits<T>::epsilon();
  for (std::size_t rule = 0; rule < parent_map.size(); ++rule)
  {
    const std::int32_t q0 = offsets[rule];
    const std::int32_t q1 = offsets[rule + 1];
    if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
      throw std::runtime_error("surface_normal offsets are out of range.");

    std::int32_t cell = parent_map[rule];
    if (!cut_data.parent_entities.empty())
    {
      const auto it = host_cell.find(cell);
      if (it == host_cell.end())
      {
        throw std::runtime_error(
            "surface_normal rule names a cell outside the cut's host cells.");
      }
      cell = it->second;
    }
    map.type = cut_data.mesh_view.cell_type(cell);
    map.gdim = gdim;
    cutcells::cell_vertex_coords_basix(cut_data.mesh_view, cell, map.vertices,
                                       node_scratch);
    const auto zit = zero_faces.find(cell);
    detail::piece_rules(
        cut_data, part, provenance.level_set_index,
        zit != zero_faces.end() ? std::span<const int>(zit->second)
                                : std::span<const int>(),
        cell, degree, map, scratch, pieces, rules);

    bool same = rules.weights.size() == static_cast<std::size_t>(q1 - q0);
    for (std::size_t i = 0; same && i < rules.points.size(); ++i)
    {
      same = std::abs(rules.points[i]
                      - points[static_cast<std::size_t>(q0) * point_dim + i])
             <= tolerance;
    }
    if (!same)
    {
      throw std::runtime_error(
          "surface_normal: the runtime quadrature of cell "
          + std::to_string(parent_map[rule])
          + " does not match the straight pieces of the cut; recreate it "
            "after updating the cut.");
    }

    for (std::int32_t q = q0; q < q1; ++q)
    {
      std::array<double, 3> normal = rules.normals[static_cast<std::size_t>(q - q0)];
      double orient = 0.0;
      for (int d = 0; d < gdim; ++d)
      {
        orient += normal[d]
                  * level_normals[static_cast<std::size_t>(q * gdim + d)];
      }
      if (orient < 0.0)
      {
        for (double& component : normal)
          component = -component;
      }
      for (int d = 0; d < gdim; ++d)
        values[static_cast<std::size_t>(q * gdim + d)] = normal[d];
    }
  }
  return values;
}
} // namespace cutfemx::geometry
