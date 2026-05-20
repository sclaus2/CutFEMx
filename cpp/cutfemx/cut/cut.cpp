// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include "cut.h"

#include "../fem/entity_dofmap.h"
#include "../mesh/convert.h"

#include <algorithm>
#include <array>
#include <basix/element-families.h>
#include <cctype>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <cutcells/ho_mesh_part_output.h>

namespace cutfemx
{
namespace
{
std::string normalize_selector(std::string_view ls_part)
{
  std::string out;
  out.reserve(ls_part.size());
  for (char c : ls_part)
  {
    if (c != ' ')
      out.push_back(c);
  }
  return out;
}

bool is_unspecified_level_set_name(std::string_view name)
{
  return name.empty() || name == "u" || name == "f";
}

bool is_valid_selector_name(std::string_view name)
{
  if (name.empty())
    return false;

  const auto is_alpha_or_underscore = [](unsigned char c)
  { return std::isalpha(c) || c == '_'; };
  const auto is_alnum_or_underscore = [](unsigned char c)
  { return std::isalnum(c) || c == '_'; };

  if (!is_alpha_or_underscore(static_cast<unsigned char>(name.front())))
    return false;
  return std::ranges::all_of(
      name.substr(1), [&](char c)
      { return is_alnum_or_underscore(static_cast<unsigned char>(c)); });
}

template <std::floating_point T>
std::vector<std::string> frozen_level_set_names(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets)
{
  std::vector<std::optional<std::string>> real_names(level_sets.size());
  std::unordered_set<std::string> real_name_set;
  real_name_set.reserve(level_sets.size());

  for (std::size_t i = 0; i < level_sets.size(); ++i)
  {
    const std::string& raw_name = level_sets[i]->name;
    if (is_unspecified_level_set_name(raw_name))
      continue;

    if (!is_valid_selector_name(raw_name))
    {
      throw std::invalid_argument(
          "Level-set function name '" + raw_name
          + "' is not a valid selector identifier. Expected "
            "[A-Za-z_][A-Za-z0-9_]*.");
    }
    if (!real_name_set.insert(raw_name).second)
    {
      throw std::invalid_argument(
          "Duplicate level-set function name '" + raw_name
          + "'. Level-set selector names must be unique.");
    }
    real_names[i] = raw_name;
  }

  std::unordered_set<std::string> used_names = real_name_set;
  std::vector<std::string> names(level_sets.size());
  for (std::size_t i = 0; i < level_sets.size(); ++i)
  {
    if (real_names[i])
    {
      names[i] = *real_names[i];
      continue;
    }

    const auto preferred = [i]()
    { return i == 0 ? std::string("phi") : "phi" + std::to_string(i); };

    std::string candidate = preferred();
    if (used_names.contains(candidate))
    {
      std::size_t fallback_counter = (i == 0) ? 1 : i + 1;
      do
      {
        candidate = "phi" + std::to_string(fallback_counter++);
      } while (used_names.contains(candidate));
    }
    used_names.insert(candidate);
    names[i] = std::move(candidate);
  }
  return names;
}

bool include_uncut_cells(int selected_dim, int tdim, std::string_view mode)
{
  const std::string normalized_mode = normalize_selector(mode);
  if (normalized_mode.empty() || normalized_mode == "auto")
    return selected_dim == tdim;

  if (normalized_mode == "cut_only")
    return false;

  if (normalized_mode == "full")
  {
    if (selected_dim != tdim)
    {
      throw std::invalid_argument(
          "cutfemx::create_cut_mesh mode='full' is only defined for volume "
          "selectors. Use mode='cut_only' for interface or lower-dimensional "
          "selections.");
    }
    return true;
  }

  throw std::invalid_argument(
      "Unsupported cut-mesh mode. Expected 'full', 'cut_only', or 'auto'.");
}

void validate_quadrature_order(int order)
{
  if (order < 0)
    throw std::invalid_argument("Quadrature order must be non-negative");
}

std::vector<std::int32_t>
identity_cell_reordering(const dolfinx::graph::AdjacencyList<std::int32_t>& graph)
{
  std::vector<std::int32_t> order(graph.num_nodes());
  std::iota(order.begin(), order.end(), 0);
  return order;
}

template <std::floating_point T>
std::int32_t num_local_entities(const CutData<T>& cut_data, int entity_dim)
{
  if (entity_dim < 0 || entity_dim > cut_data.tdim)
  {
    throw std::invalid_argument(
        "cutfemx::locate_entities entity_dim is out of range");
  }

  auto topology = cut_data.mesh_owner->topology();
  auto entity_map = topology->index_map(entity_dim);
  if (!entity_map)
  {
    cut_data.mesh_owner->topology_mutable()->create_entities(entity_dim);
    entity_map = topology->index_map(entity_dim);
  }
  if (!entity_map)
    throw std::runtime_error("Mesh topology has no entity index map");

  return static_cast<std::int32_t>(entity_map->size_local());
}

void validate_local_entities(std::span<const std::int32_t> entities,
                             std::int32_t num_local_entities)
{
  for (const std::int32_t entity : entities)
  {
    if (entity < 0 || entity >= num_local_entities)
    {
      throw std::out_of_range(
          "cutfemx::locate_entities entity index is out of range");
    }
  }
}

template <std::floating_point T>
cutcells::cell::domain
classify_entity_dofs(std::span<const std::int32_t> dofs,
                     std::span<const T> dof_values)
{
  if (dofs.empty())
    throw std::runtime_error("cutfemx::locate_entities entity has no dofs");

  bool all_negative = true;
  bool all_positive = true;

  for (const std::int32_t dof : dofs)
  {
    if (dof < 0 || static_cast<std::size_t>(dof) >= dof_values.size())
      throw std::out_of_range("Level-set dof index is out of range");

    const T value = dof_values[static_cast<std::size_t>(dof)];
    const bool is_negative = value < T(0);
    const bool is_positive = value > T(0);

    all_negative = all_negative && is_negative;
    all_positive = all_positive && is_positive;
  }

  if (all_negative)
    return cutcells::cell::domain::inside;
  if (all_positive)
    return cutcells::cell::domain::outside;
  return cutcells::cell::domain::intersected;
}

bool relation_matches_domain(cutcells::cell::domain domain,
                             cutcells::Relation relation)
{
  switch (relation)
  {
  case cutcells::Relation::LessThan:
    return domain == cutcells::cell::domain::inside;
  case cutcells::Relation::GreaterThan:
    return domain == cutcells::cell::domain::outside;
  case cutcells::Relation::EqualTo:
    return domain == cutcells::cell::domain::intersected;
  }
  return false;
}
} // namespace

template <std::floating_point T>
void validate_level_set(const dolfinx::fem::Function<T>& level_set)
{
  const auto element = level_set.function_space()->element();
  if (!element)
    throw std::invalid_argument("Level-set function has no finite element");

  if (!element->value_shape().empty())
    throw std::invalid_argument("cutfemx::cut requires a scalar level-set function");

  const auto& basix_element = element->basix_element();
  if (basix_element.family() != basix::element::family::P)
  {
    throw std::invalid_argument(
        "cutfemx::cut requires a scalar Lagrange level-set function");
  }
}

template <std::floating_point T>
void validate_equivalent_level_set_space(
    const dolfinx::fem::Function<T>& reference,
    const dolfinx::fem::Function<T>& candidate)
{
  const auto reference_space = reference.function_space();
  const auto candidate_space = candidate.function_space();
  if (!reference_space || !candidate_space)
    throw std::invalid_argument("Level-set function has no function space");

  if (reference_space->mesh() != candidate_space->mesh())
  {
    throw std::invalid_argument(
        "All level-set functions must be defined on the same background mesh");
  }

  const auto reference_element = reference_space->element();
  const auto candidate_element = candidate_space->element();
  if (!reference_element || !candidate_element)
    throw std::invalid_argument("Level-set function has no finite element");
  if (*reference_element != *candidate_element)
  {
    throw std::invalid_argument(
        "All level-set functions must use structurally equivalent scalar "
        "Lagrange elements");
  }

  const auto reference_dofmap = reference_space->dofmap();
  const auto candidate_dofmap = candidate_space->dofmap();
  if (!reference_dofmap || !candidate_dofmap)
    throw std::invalid_argument("Level-set function has no dofmap");
  if (!(*reference_dofmap == *candidate_dofmap))
  {
    throw std::invalid_argument(
        "All level-set functions must use compatible dofmap layouts");
  }
}

template <std::floating_point T>
void build_mesh_view(CutData<T>& cut_data)
{
  auto cell_map = cut_data.mesh_owner->topology()->index_map(cut_data.tdim);
  if (!cell_map)
    throw std::runtime_error("Mesh topology has no cell index map");

  cut_data.num_local_cells = static_cast<std::int32_t>(cell_map->size_local());

  std::span<const T> x = cut_data.mesh_owner->geometry().x();

  const auto cell_type = cut_data.mesh_owner->topology()->cell_type();
  const auto cutcells_cell_type = mesh::dolfinx_to_cutcells_cell_type(cell_type);
  const int num_vertices = cutcells::cell::get_num_vertices(cutcells_cell_type);

  auto x_dofmap = cut_data.mesh_owner->geometry().dofmap();
  const std::size_t x_dofmap_width = x_dofmap.extent(1);
  if (static_cast<int>(x_dofmap_width) < num_vertices)
  {
    throw std::runtime_error(
        "Mesh geometry dofmap does not contain enough vertex entries");
  }
  if (x_dofmap.extent(0) < static_cast<std::size_t>(cut_data.num_local_cells))
    throw std::runtime_error("Mesh geometry dofmap has too few cell rows");

  cut_data.mesh_view = {};
  cut_data.mesh_view.gdim = cut_data.gdim;
  cut_data.mesh_view.tdim = cut_data.tdim;
  cut_data.mesh_view.coordinates = x;
  cut_data.mesh_view.coordinate_stride = 3;
  cut_data.mesh_view.connectivity = std::span<const std::int32_t>(
      x_dofmap.data_handle(), x_dofmap.extent(0) * x_dofmap_width);
  cut_data.mesh_view.cell_count = cut_data.num_local_cells;
  cut_data.mesh_view.cell_width = static_cast<std::int32_t>(num_vertices);
  cut_data.mesh_view.cell_stride = static_cast<std::int32_t>(x_dofmap_width);
  cut_data.mesh_view.uniform_cell_type = cutcells_cell_type;
  cut_data.mesh_view.has_uniform_cell_type = true;
  cut_data.mesh_view.vtk_vertex_order = false;
}

template <std::floating_point T>
cutcells::LevelSetFunction<T, std::int32_t>
build_level_set_function(const CutData<T>& cut_data,
                         const std::shared_ptr<const dolfinx::fem::Function<T>>&
                             level_set_owner,
                         std::string name)
{
  const auto function_space = level_set_owner->function_space();
  const auto element = function_space->element();
  const int degree = element->basix_element().degree();

  std::vector<T> dof_coordinates_3
      = function_space->tabulate_dof_coordinates(false);
  const std::size_t num_dofs = dof_coordinates_3.size() / 3;
  std::vector<T> dof_coordinates;
  if (cut_data.gdim == 3)
  {
    dof_coordinates = std::move(dof_coordinates_3);
  }
  else
  {
    // tabulate_dof_coordinates returns xyz triples. CutCells stores compact
    // gdim-tuples once because they are consumed repeatedly during cutting.
    dof_coordinates.resize(num_dofs * static_cast<std::size_t>(cut_data.gdim));
    for (std::size_t dof = 0; dof < num_dofs; ++dof)
    {
      for (int d = 0; d < cut_data.gdim; ++d)
      {
        dof_coordinates[dof * static_cast<std::size_t>(cut_data.gdim) + d]
            = dof_coordinates_3[3 * dof + static_cast<std::size_t>(d)];
      }
    }
  }

  const auto dofmap = function_space->dofmap();
  const auto dofmap_view = dofmap->map();
  if (dofmap_view.extent(0) < static_cast<std::size_t>(cut_data.num_local_cells))
    throw std::runtime_error("Level-set dofmap has too few cell rows");

  const std::size_t dofmap_width = dofmap_view.extent(1);
  if (dofmap_width == 0)
    throw std::runtime_error("Level-set dofmap has empty cell rows");

  const auto dofmap_owner
      = std::shared_ptr<const void>(dofmap, static_cast<const void*>(dofmap.get()));

  auto level_set_mesh_data
      = cutcells::create_level_set_mesh_data_view<T, std::int32_t>(
          cut_data.gdim, cut_data.tdim, degree, std::move(dof_coordinates),
          std::span<const std::int32_t>(
              dofmap_view.data_handle(), dofmap_view.extent(0) * dofmap_width),
          cut_data.num_local_cells, static_cast<std::int32_t>(dofmap_width),
          static_cast<std::int32_t>(dofmap_width));
  level_set_mesh_data.uniform_cell_type = cut_data.mesh_view.uniform_cell_type;
  level_set_mesh_data.has_uniform_cell_type = true;
  level_set_mesh_data.owner = dofmap_owner;

  const std::span<const T> dof_values = level_set_owner->x()->array();
  return cutcells::create_level_set_function_view<T, std::int32_t>(
      std::move(level_set_mesh_data), dof_values, std::move(name),
      level_set_owner);
}

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::fem::Function<T>> level_set)
{
  if (!level_set)
    throw std::invalid_argument("cutfemx::cut requires a level-set function");

  auto mesh = level_set->function_space()->mesh();
  if (!mesh)
  {
    throw std::invalid_argument(
        "cutfemx::cut could not infer a mesh from the level-set function");
  }

  std::array<std::shared_ptr<const dolfinx::fem::Function<T>>, 1> level_sets{
      std::move(level_set)};
  return cut<T>(std::move(mesh),
                std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>(
                    level_sets.data(), level_sets.size()));
}

template <std::floating_point T>
CutData<T> cut(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
               std::shared_ptr<const dolfinx::fem::Function<T>> level_set)
{
  if (!level_set)
    throw std::invalid_argument("cutfemx::cut requires a level-set function");

  std::array<std::shared_ptr<const dolfinx::fem::Function<T>>, 1> level_sets{
      std::move(level_set)};
  return cut<T>(std::move(mesh),
                std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>(
                    level_sets.data(), level_sets.size()));
}

template <std::floating_point T>
CutData<T> cut(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets)
{
  if (level_sets.empty())
    throw std::invalid_argument("cutfemx::cut requires at least one level-set function");
  if (!level_sets.front())
    throw std::invalid_argument("cutfemx::cut requires non-null level-set functions");

  auto mesh = level_sets.front()->function_space()->mesh();
  if (!mesh)
  {
    throw std::invalid_argument(
        "cutfemx::cut could not infer a mesh from the level-set functions");
  }

  return cut<T>(std::move(mesh), level_sets);
}

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets)
{
  if (!mesh)
    throw std::invalid_argument("cutfemx::cut requires a mesh");
  if (level_sets.empty())
    throw std::invalid_argument("cutfemx::cut requires at least one level-set function");

  for (const auto& level_set : level_sets)
  {
    if (!level_set)
      throw std::invalid_argument("cutfemx::cut requires non-null level-set functions");
    auto level_set_mesh = level_set->function_space()->mesh();
    if (level_set_mesh && level_set_mesh.get() != mesh.get())
    {
      throw std::invalid_argument(
          "cutfemx::cut mesh must match the level-set function space mesh");
    }
    validate_level_set(*level_set);
  }

  for (std::size_t i = 1; i < level_sets.size(); ++i)
    validate_equivalent_level_set_space(*level_sets.front(), *level_sets[i]);

  CutData<T> cut_data;
  cut_data.mesh_owner = std::move(mesh);
  cut_data.level_set_owners.assign(level_sets.begin(), level_sets.end());
  cut_data.level_set_names = frozen_level_set_names<T>(level_sets);
  cut_data.gdim = cut_data.mesh_owner->geometry().dim();
  cut_data.tdim = cut_data.mesh_owner->topology()->dim();

  build_mesh_view(cut_data);
  cut_data.level_sets.reserve(cut_data.level_set_owners.size());
  for (std::size_t i = 0; i < cut_data.level_set_owners.size(); ++i)
  {
    cut_data.level_sets.push_back(build_level_set_function(
        cut_data, cut_data.level_set_owners[i], cut_data.level_set_names[i]));
  }
  update(cut_data);
  return cut_data;
}

template <std::floating_point T>
CutData<T> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::initializer_list<std::shared_ptr<const dolfinx::fem::Function<T>>>
        level_sets)
{
  return cut<T>(std::move(mesh),
                std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>(
                    level_sets.begin(), level_sets.size()));
}

template <std::floating_point T>
void update(CutData<T>& cut_data)
{
  if (cut_data.level_sets.size() != cut_data.level_set_owners.size())
  {
    throw std::runtime_error(
        "CutData level-set storage is inconsistent with its owners");
  }

  for (std::size_t i = 0; i < cut_data.level_sets.size(); ++i)
    cut_data.level_sets[i].dof_values = cut_data.level_set_owners[i]->x()->array();

  auto [cut_cells, parent_cells]
      = cutcells::cut<T, std::int32_t>(cut_data.mesh_view,
                                       cut_data.level_sets, true);

  cut_data.cut_cells = std::move(cut_cells);
  cut_data.parent_cells = std::move(parent_cells);
  if (cut_data.parent_cells.level_set_names != cut_data.level_set_names)
  {
    throw std::runtime_error(
        "CutData level-set names changed during update");
  }
  cut_data.rebind_views();
}

template <std::floating_point T>
cutcells::HOMeshPart<T, std::int32_t> select_mesh_part(
    const cutcells::MeshView<T, std::int32_t>& mesh_view,
    const cutcells::HOCutCells<T, std::int32_t>& cut_cells,
    const cutcells::ParentCellClassification<T, std::int32_t>& parent_cells,
    std::string_view ls_part);

template <std::floating_point T>
std::vector<std::int32_t>
locate_entities(const CutData<T>& cut_data, std::string_view ls_part)
{
  const std::int32_t num_cells = num_local_entities(cut_data, cut_data.tdim);
  std::vector<std::int32_t> entities(static_cast<std::size_t>(num_cells));
  std::iota(entities.begin(), entities.end(), 0);
  return locate_entities(cut_data, entities, cut_data.tdim, ls_part);
}

template <std::floating_point T>
std::vector<std::int32_t>
locate_entities(const CutData<T>& cut_data,
                std::span<const std::int32_t> entities, int entity_dim,
                std::string_view ls_part)
{
  if (cut_data.level_set_owners.empty())
    throw std::runtime_error("CutData has no level-set functions");
  if (cut_data.level_set_owners.size() != cut_data.level_set_names.size())
  {
    throw std::runtime_error(
        "CutData level-set storage is inconsistent with its names");
  }

  const std::int32_t num_entities = num_local_entities(cut_data, entity_dim);
  validate_local_entities(entities, num_entities);

  if (entities.empty())
    return {};

  cutcells::SelectionExpr expr = cutcells::parse_selection_expr(ls_part);
  cutcells::compile_selection_expr(expr, cut_data.level_set_names);

  const auto function_space = cut_data.level_set_owners.front()->function_space();
  if (!function_space || !function_space->dofmap())
    throw std::runtime_error("Level-set function has no dofmap");

  std::vector<std::vector<std::int32_t>> entity_dofs(entities.size());
  fem::create_entity_dofmap(entities, entity_dim, cut_data.mesh_owner,
                            function_space->dofmap(), entity_dofs);

  std::vector<std::span<const T>> level_set_values;
  level_set_values.reserve(cut_data.level_set_owners.size());
  for (const auto& level_set : cut_data.level_set_owners)
    level_set_values.push_back(level_set->x()->array());

  std::vector<std::int32_t> marked_entities;
  marked_entities.reserve(entities.size());

  for (std::size_t entity_pos = 0; entity_pos < entities.size(); ++entity_pos)
  {
    std::vector<cutcells::cell::domain> domains(level_set_values.size(),
                                                cutcells::cell::domain::unset);
    std::vector<std::uint8_t> computed(level_set_values.size(), 0);

    auto domain_for = [&](int level_set_index)
    {
      if (level_set_index < 0
          || static_cast<std::size_t>(level_set_index) >= level_set_values.size())
      {
        throw std::runtime_error(
            "Compiled selector contains an invalid level-set index");
      }

      const std::size_t ls = static_cast<std::size_t>(level_set_index);
      if (computed[ls] == 0)
      {
        domains[ls] = classify_entity_dofs<T>(entity_dofs[entity_pos],
                                              level_set_values[ls]);
        computed[ls] = 1;
      }
      return domains[ls];
    };

    bool entity_matches = false;
    for (const auto& term : expr.terms)
    {
      bool term_matches = true;
      for (const auto& clause : term.clauses)
      {
        if (!relation_matches_domain(domain_for(clause.level_set_index),
                                     clause.relation))
        {
          term_matches = false;
          break;
        }
      }
      if (term_matches)
      {
        entity_matches = true;
        break;
      }
    }

    if (entity_matches)
      marked_entities.push_back(entities[entity_pos]);
  }

  return marked_entities;
}

template <std::floating_point T>
cutcells::HOMeshPart<T, std::int32_t> select_mesh_part(
    const cutcells::MeshView<T, std::int32_t>& mesh_view,
    const cutcells::HOCutCells<T, std::int32_t>& cut_cells,
    const cutcells::ParentCellClassification<T, std::int32_t>& parent_cells,
    std::string_view ls_part)
{
  return cutcells::select_part<T, std::int32_t>(mesh_view, cut_cells, parent_cells,
                                                ls_part);
}

template <std::floating_point T>
void filter_mesh_part_to_cells(cutcells::HOMeshPart<T, std::int32_t>& part,
                               std::span<const std::int32_t> entities,
                               std::int32_t num_local_cells)
{
  std::vector<std::uint8_t> candidate_mask(
      static_cast<std::size_t>(num_local_cells), 0);
  for (const std::int32_t entity : entities)
  {
    if (entity < 0 || entity >= num_local_cells)
      throw std::out_of_range("cutfemx entity index is out of range");
    candidate_mask[static_cast<std::size_t>(entity)] = 1;
  }

  auto keep_parent_cell = [&candidate_mask](std::int32_t parent_cell)
  {
    return candidate_mask[static_cast<std::size_t>(parent_cell)] != 0;
  };

  std::erase_if(part.cut_cell_ids,
                [&part, &keep_parent_cell](std::int32_t cut_cell_id)
                {
                  const std::int32_t parent_cell
                      = part.cut_cells->parent_cell_ids[static_cast<std::size_t>(
                          cut_cell_id)];
                  return !keep_parent_cell(parent_cell);
                });

  std::erase_if(part.uncut_cell_ids,
                [&keep_parent_cell](std::int32_t parent_cell)
                { return !keep_parent_cell(parent_cell); });
}

template <std::floating_point T>
mesh::CutMesh<T> dolfinx_cut_mesh_from_cutcells_mesh(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> background_mesh,
    cutcells::mesh::CutMesh<T>&& cutcells_mesh,
    std::size_t num_uncut_cells)
{
  mesh::CutMesh<T> out;
  out._bg_mesh = std::move(background_mesh);

  if (cutcells_mesh._num_cells == 0)
    return out;

  if (cutcells_mesh._types.empty())
    throw std::runtime_error("CutCells visualisation mesh has no cell types");

  const cutcells::cell::type first_type = cutcells_mesh._types.front();
  if (!std::ranges::all_of(cutcells_mesh._types,
                           [first_type](cutcells::cell::type type)
                           { return type == first_type; }))
  {
    throw std::runtime_error(
        "CutFEMx currently supports single-cell-type cut meshes only");
  }

  const dolfinx::mesh::CellType cell_type
      = mesh::cutcells_to_dolfinx_cell_type(first_type);
  const int vertices_per_cell = dolfinx::mesh::num_cell_vertices(cell_type);

  if (cutcells_mesh._offset.size()
      != static_cast<std::size_t>(cutcells_mesh._num_cells + 1))
  {
    throw std::runtime_error("CutCells visualisation mesh has invalid offsets");
  }

  const std::int64_t num_vertices_local = cutcells_mesh._num_vertices;
  std::int64_t vertex_offset = 0;
  MPI_Exscan(&num_vertices_local, &vertex_offset, 1, MPI_INT64_T, MPI_SUM,
             out._bg_mesh->comm());

  std::vector<std::int64_t> cells;
  cells.reserve(static_cast<std::size_t>(cutcells_mesh._num_cells
                                         * vertices_per_cell));
  for (int c = 0; c < cutcells_mesh._num_cells; ++c)
  {
    const int begin = cutcells_mesh._offset[static_cast<std::size_t>(c)];
    const int end = cutcells_mesh._offset[static_cast<std::size_t>(c) + 1];
    if (end - begin != vertices_per_cell)
    {
      throw std::runtime_error(
          "CutCells visualisation mesh cell arity is inconsistent");
    }

    for (int j = begin; j < end; ++j)
    {
      cells.push_back(static_cast<std::int64_t>(cutcells_mesh._connectivity[
                          static_cast<std::size_t>(j)])
                      + vertex_offset);
    }
  }

  dolfinx::fem::CoordinateElement<T> element(cell_type, 1);
  const std::array<std::size_t, 2> xshape{
      static_cast<std::size_t>(cutcells_mesh._num_vertices),
      static_cast<std::size_t>(cutcells_mesh._gdim)};
  dolfinx::mesh::CellReorderFunction identity_reorder
      = identity_cell_reordering;
  out._cut_mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(
      dolfinx::mesh::create_mesh(
          out._bg_mesh->comm(), out._bg_mesh->comm(),
          std::span<const std::int64_t>(cells.data(), cells.size()), element,
          out._bg_mesh->comm(), cutcells_mesh._vertex_coords, xshape, nullptr,
          std::int32_t(2), identity_reorder));

  out._parent_index = std::move(cutcells_mesh._parent_map);
  out._is_cut_cell.assign(static_cast<std::size_t>(cutcells_mesh._num_cells),
                          std::int8_t(1));
  if (num_uncut_cells > out._is_cut_cell.size())
  {
    throw std::runtime_error(
        "CutFEMx cut-mesh uncut-cell count is inconsistent");
  }
  const std::size_t first_uncut = out._is_cut_cell.size() - num_uncut_cells;
  std::fill(out._is_cut_cell.begin() + static_cast<std::ptrdiff_t>(first_uncut),
            out._is_cut_cell.end(), std::int8_t(0));

  return out;
}

template <std::floating_point T>
mesh::CutMesh<T> create_cut_mesh(const CutData<T>& cut_data,
                                 std::string_view ls_part,
                                 std::string_view mode)
{
  auto part = select_mesh_part(cut_data.mesh_view, cut_data.cut_cells,
                               cut_data.parent_cells, ls_part);
  const bool include_uncut = include_uncut_cells(part.dim, cut_data.tdim, mode);
  cutcells::mesh::CutMesh<T> cutcells_mesh
      = cutcells::output::visualization_mesh<T, std::int32_t>(part,
                                                              include_uncut);
  const std::size_t num_uncut_cells
      = include_uncut && part.dim == cut_data.tdim ? part.uncut_cell_ids.size()
                                                   : 0;
  return dolfinx_cut_mesh_from_cutcells_mesh(
      cut_data.mesh_owner, std::move(cutcells_mesh), num_uncut_cells);
}

template <std::floating_point T>
mesh::CutMesh<T> create_cut_mesh(const CutData<T>& cut_data,
                                 std::span<const std::int32_t> entities,
                                 int entity_dim, std::string_view ls_part,
                                 std::string_view mode)
{
  if (entity_dim != cut_data.tdim)
  {
    throw std::invalid_argument(
        "cutfemx::create_cut_mesh currently supports cell entities only");
  }

  auto part = select_mesh_part(cut_data.mesh_view, cut_data.cut_cells,
                               cut_data.parent_cells, ls_part);
  const bool include_uncut = include_uncut_cells(part.dim, cut_data.tdim, mode);
  filter_mesh_part_to_cells(part, entities, cut_data.num_local_cells);
  cutcells::mesh::CutMesh<T> cutcells_mesh
      = cutcells::output::visualization_mesh<T, std::int32_t>(part,
                                                              include_uncut);
  const std::size_t num_uncut_cells
      = include_uncut && part.dim == cut_data.tdim ? part.uncut_cell_ids.size()
                                                   : 0;
  return dolfinx_cut_mesh_from_cutcells_mesh(
      cut_data.mesh_owner, std::move(cutcells_mesh), num_uncut_cells);
}

template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature(const CutData<T>& cut_data,
                                        std::string_view ls_part, int order)
{
  validate_quadrature_order(order);

  auto part = select_mesh_part(cut_data.mesh_view, cut_data.cut_cells,
                               cut_data.parent_cells, ls_part);
  cutcells::quadrature::QuadratureRules<T> rules
      = cutcells::output::quadrature_rules<T, std::int32_t>(
          part, order, /*include_uncut_cells=*/false);
  return RuntimeQuadrature<T>(std::move(rules), cut_data.mesh_owner);
}

template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature(const CutData<T>& cut_data,
                                        std::span<const std::int32_t> entities,
                                        int entity_dim,
                                        std::string_view ls_part, int order)
{
  if (entity_dim != cut_data.tdim)
  {
    throw std::invalid_argument(
        "cutfemx::runtime_quadrature currently supports cell entities only");
  }

  validate_quadrature_order(order);

  auto part = select_mesh_part(cut_data.mesh_view, cut_data.cut_cells,
                               cut_data.parent_cells, ls_part);
  filter_mesh_part_to_cells(part, entities, cut_data.num_local_cells);
  cutcells::quadrature::QuadratureRules<T> rules
      = cutcells::output::quadrature_rules<T, std::int32_t>(
          part, order, /*include_uncut_cells=*/false);
  return RuntimeQuadrature<T>(std::move(rules), cut_data.mesh_owner);
}

template CutData<double> cut(
    std::shared_ptr<const dolfinx::fem::Function<double>>);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::fem::Function<float>>);
template CutData<double> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>>,
    std::shared_ptr<const dolfinx::fem::Function<double>>);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<float>>,
    std::shared_ptr<const dolfinx::fem::Function<float>>);
template CutData<double> cut(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<double>>>);
template CutData<float> cut(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<float>>>);
template CutData<double> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>>,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<double>>>);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<float>>,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<float>>>);
template CutData<double> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>>,
    std::initializer_list<std::shared_ptr<const dolfinx::fem::Function<double>>>);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<float>>,
    std::initializer_list<std::shared_ptr<const dolfinx::fem::Function<float>>>);

template void update(CutData<double>&);
template void update(CutData<float>&);

template std::vector<std::int32_t> locate_entities(
    const CutData<double>&, std::string_view);
template std::vector<std::int32_t> locate_entities(
    const CutData<float>&, std::string_view);
template std::vector<std::int32_t> locate_entities(
    const CutData<double>&, std::span<const std::int32_t>, int,
    std::string_view);
template std::vector<std::int32_t> locate_entities(
    const CutData<float>&, std::span<const std::int32_t>, int,
    std::string_view);

template mesh::CutMesh<double> create_cut_mesh(
    const CutData<double>&, std::string_view, std::string_view);
template mesh::CutMesh<float> create_cut_mesh(
    const CutData<float>&, std::string_view, std::string_view);
template mesh::CutMesh<double> create_cut_mesh(
    const CutData<double>&, std::span<const std::int32_t>, int,
    std::string_view, std::string_view);
template mesh::CutMesh<float> create_cut_mesh(
    const CutData<float>&, std::span<const std::int32_t>, int,
    std::string_view, std::string_view);

template RuntimeQuadrature<double> runtime_quadrature(
    const CutData<double>&, std::string_view, int);
template RuntimeQuadrature<float> runtime_quadrature(
    const CutData<float>&, std::string_view, int);
template RuntimeQuadrature<double> runtime_quadrature(
    const CutData<double>&, std::span<const std::int32_t>, int,
    std::string_view, int);
template RuntimeQuadrature<float> runtime_quadrature(
    const CutData<float>&, std::span<const std::int32_t>, int,
    std::string_view, int);

} // namespace cutfemx
