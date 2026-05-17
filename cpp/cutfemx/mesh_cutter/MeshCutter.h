// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cutcells/ho_cut_mesh.h>
#include <cutcells/level_set.h>
#include <cutcells/mesh_view.h>

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include "../mesh/cut_mesh.h"
#include "RuntimeQuadrature.h"

namespace cutfemx
{

template <std::floating_point T>
class MeshCutter
{
public:
  explicit MeshCutter(
      std::shared_ptr<const dolfinx::fem::Function<T>> level_set);

  std::shared_ptr<const dolfinx::fem::Function<T>> level_set() const;
  std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh() const;

  void update();

  std::vector<std::int32_t> locate_entities(std::string_view ls_part) const;
  std::vector<std::int32_t>
  locate_entities(std::span<const std::int32_t> entities, int entity_dim,
                  std::string_view ls_part) const;

  mesh::CutMesh<T> create_cut_mesh(std::string_view ls_part,
                                   std::string_view mode) const;
  mesh::CutMesh<T> create_cut_mesh(std::span<const std::int32_t> entities,
                                   int entity_dim, std::string_view ls_part,
                                   std::string_view mode) const;

  RuntimeQuadrature<T> runtime_quadrature(std::string_view ls_part,
                                          int order) const;
  RuntimeQuadrature<T> runtime_quadrature(std::span<const std::int32_t> entities,
                                          int entity_dim,
                                          std::string_view ls_part,
                                          int order) const;

private:
  void validate_level_set() const;
  void build_mesh_view();
  void build_level_set_mesh_data();
  cutcells::LevelSetFunction<T, std::int32_t>
  current_level_set_function() const;

  std::shared_ptr<const dolfinx::fem::Function<T>> _level_set;
  std::shared_ptr<const dolfinx::mesh::Mesh<T>> _mesh;

  int _gdim = 0;
  int _tdim = 0;
  std::int32_t _num_local_cells = 0;

  std::vector<T> _mesh_coordinates;
  std::vector<std::int32_t> _mesh_connectivity;
  std::vector<std::int32_t> _mesh_offsets;
  std::vector<cutcells::cell::type> _mesh_cell_types;
  cutcells::MeshView<T, std::int32_t> _mesh_view;

  std::vector<T> _level_set_dof_coordinates;
  std::vector<std::int32_t> _level_set_cell_dofs;
  std::vector<std::int32_t> _level_set_cell_offsets;
  std::shared_ptr<cutcells::LevelSetMeshData<T, std::int32_t>>
      _level_set_mesh_data;

  cutcells::HOCutCells<T, std::int32_t> _cut_cells;
  cutcells::BackgroundMeshData<T, std::int32_t> _background_data;
};

} // namespace cutfemx
