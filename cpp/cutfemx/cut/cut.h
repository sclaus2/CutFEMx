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

#include <cutcells/ho_cut_mesh.h>
#include <cutcells/level_set.h>
#include <cutcells/mesh_view.h>

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include "../mesh/cut_mesh.h"
#include "runtime_quadrature.h"

namespace cutfemx
{

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
        cut_cells(std::move(other.cut_cells)),
        parent_cells(std::move(other.parent_cells))
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
    cut_cells = std::move(other.cut_cells);
    parent_cells = std::move(other.parent_cells);
    rebind_views();
    return *this;
  }

  void rebind_views()
  {
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
  cutcells::HOCutCells<T, std::int32_t> cut_cells;
  cutcells::ParentCellClassification<T, std::int32_t> parent_cells;
};

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::fem::Function<T>> level_set);

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
               std::span<const std::int32_t> entities, int entity_dim);

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
               std::shared_ptr<const dolfinx::fem::Function<T>> level_set);

template <std::floating_point T>
CutData<T> cut(std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
                   level_sets);

template <std::floating_point T>
CutData<T> cut(std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
                   level_sets,
               std::span<const std::int32_t> entities, int entity_dim);

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
               std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
                   level_sets);

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
               std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>
                   level_sets,
               std::span<const std::int32_t> entities, int entity_dim);

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::initializer_list<std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets);

template <std::floating_point T>
void update(CutData<T>& cut_data);

template <std::floating_point T>
std::vector<std::int32_t> locate_entities(const CutData<T>& cut_data,
                                          std::string_view ls_part);

template <std::floating_point T>
std::vector<std::int32_t> interior_facets_for_cells(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::int32_t> cells, bool include_ghosts);

template <std::floating_point T>
mesh::CutMesh<T> create_cut_mesh(const CutData<T>& cut_data,
                                 std::string_view ls_part,
                                 std::string_view mode);

template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature(const CutData<T>& cut_data,
                                        std::string_view ls_part, int order,
                                        std::string_view backend = "straight");

template <std::floating_point T>
std::vector<std::pair<std::string, RuntimeQuadrature<T>>> runtime_quadratures(
    const CutData<T>& cut_data, std::span<const std::string> ls_parts,
    int order, std::string_view backend = "straight");

} // namespace cutfemx
