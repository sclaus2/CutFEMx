// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cutcells/level_set.h>
#include <cutcells/lut/cell_pieces.h>
#include <cutcells/mesh_view.h>
#include <cutcells/part/cut_result.h>
#include <cutcells/part/mesh_part.h>
#include <cutcells/quadrays/engine.h>

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include "../mesh/cut_mesh.h"
#include "runtime_quadrature.h"

namespace cutfemx
{

/// Options of cut(): the classification of the host cells, and the options
/// of the backends that integrate and show the selected parts.
struct CutOptions
{
  cutcells::part::ClassifyOptions classify;
  /// The lookup tables (backend "straight"): the order of the Pk-iso-P1
  /// template that subdivides a cut cell (0: the level sets' degree) and
  /// the triangulation of the cut pieces.
  cutcells::lut::Options lut;
  /// quadrays (backend "quadrays").
  cutcells::quadrays::Options quadrays;
};

template <std::floating_point T>
struct CutData
{
  CutData() = default;
  CutData(const CutData&) = delete;
  CutData& operator=(const CutData&) = delete;

  CutData(CutData&& other) noexcept
      : level_set_owners(std::move(other.level_set_owners)),
        mesh_owner(std::move(other.mesh_owner)),
        level_set_names(std::move(other.level_set_names)),
        gdim(other.gdim), tdim(other.tdim),
        num_local_cells(other.num_local_cells),
        parent_entities(std::move(other.parent_entities)),
        owned_connectivity(std::move(other.owned_connectivity)),
        mesh_view(std::move(other.mesh_view)),
        level_sets(std::move(other.level_sets)),
        options(std::move(other.options)), result(std::move(other.result))
  {
    rebind_views();
  }

  CutData& operator=(CutData&& other) noexcept
  {
    if (this == &other)
      return *this;

    level_set_owners = std::move(other.level_set_owners);
    mesh_owner = std::move(other.mesh_owner);
    level_set_names = std::move(other.level_set_names);
    gdim = other.gdim;
    tdim = other.tdim;
    num_local_cells = other.num_local_cells;
    parent_entities = std::move(other.parent_entities);
    owned_connectivity = std::move(other.owned_connectivity);
    mesh_view = std::move(other.mesh_view);
    level_sets = std::move(other.level_sets);
    options = std::move(other.options);
    result = std::move(other.result);
    rebind_views();
    return *this;
  }

  /// Point the cut result at this object's mesh view and level sets, which
  /// it does not own.
  void rebind_views()
  {
    if (result.mesh == nullptr)
      return;
    result.mesh = &mesh_view;
    result.level_sets.resize(level_sets.size());
    for (std::size_t i = 0; i < level_sets.size(); ++i)
      result.level_sets[i] = &level_sets[i];
  }

  std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> level_set_owners;
  std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh_owner;
  std::vector<std::string> level_set_names;

  int gdim = 0;
  int tdim = 0;
  std::int32_t num_local_cells = 0;

  /// Optional map from host mesh cells to background mesh entities. Empty means
  /// the host is the original cell mesh and parent ids are already background
  /// cell ids.
  std::vector<std::int32_t> parent_entities;
  /// Owned host connectivity when the cut is performed on a selected entity
  /// mesh rather than the background cell mesh.
  std::vector<std::int32_t> owned_connectivity;

  cutcells::MeshView<T, std::int32_t> mesh_view;
  std::vector<cutcells::LevelSetFunction<T, std::int32_t>> level_sets;
  CutOptions options;
  /// Every host cell classified by every level set, and the host facets that
  /// lie in a zero set; it points at mesh_view and level_sets. A cut of the
  /// cells of a mesh holds its ghost cells after the num_local_cells owned
  /// ones (select_part leaves them out).
  cutcells::part::CutResult<T, std::int32_t> result;
};

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
    std::span<const std::int32_t> entities, int entity_dim,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets,
    std::span<const std::int32_t> entities, int entity_dim,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets,
    std::span<const std::int32_t> entities, int entity_dim,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::initializer_list<std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets,
    const CutOptions& options = CutOptions{});

template <std::floating_point T>
void update(CutData<T>& cut_data);

template <std::floating_point T>
std::vector<std::int32_t> locate_entities(const CutData<T>& cut_data,
                                          std::string_view ls_part);

/// The part @p ls_part of the host cells this process owns: the cells wholly
/// in it, the cut cells holding a piece of it and the zero faces in it.
template <std::floating_point T>
cutcells::part::MeshPart<T, std::int32_t>
select_part(const CutData<T>& cut_data, std::string_view ls_part);

/// Return sorted raw local interior facet ids whose two adjacent cells both
/// lie in `cells`.
template <std::floating_point T>
std::vector<std::int32_t> interior_facets_for_cells(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::int32_t> cells, bool include_ghosts);

/// Return sorted raw local interior facet ids that are incident to at least
/// one cell of `cells` and whose two adjacent cells both lie in
/// `active_cells`. Cells in `cells` are not implicitly active.
template <std::floating_point T>
std::vector<std::int32_t> interior_facets_for_cells(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::int32_t> cells,
    std::span<const std::int32_t> active_cells, bool include_ghosts);

template <std::floating_point T>
mesh::CutMesh<T> create_cut_mesh(const CutData<T>& cut_data,
                                 std::string_view ls_part,
                                 std::string_view mode);

/// Quadrature rules of the part @p ls_part, one rule per host cell: on the
/// cut cells, and on the facets lying in a zero set that the part asks for.
///
/// @param order    the polynomial degree integrated exactly on flat pieces,
///                 whole cells and zero faces, 1 to 10 (quadrays takes enough
///                 Gauss-Legendre points per segment of each height line for
///                 it)
/// @param backend  "straight" (or "lut"): the lookup tables' straight pieces
///                 (CutOptions::lut); "quadrays": curved rules of the level
///                 sets themselves (CutOptions::quadrays), for triangles and
///                 quadrilaterals in 2D and tetrahedra, hexahedra, prisms and
///                 pyramids in 3D
template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature(const CutData<T>& cut_data,
                                        std::string_view ls_part, int order,
                                        std::string_view backend = "straight");

template <std::floating_point T>
std::vector<std::pair<std::string, RuntimeQuadrature<T>>> runtime_quadratures(
    const CutData<T>& cut_data, std::span<const std::string> ls_parts,
    int order, std::string_view backend = "straight");

} // namespace cutfemx
