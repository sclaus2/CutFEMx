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
#include <unordered_map>
#include <unordered_set>

#include <cutcells/cell_types.h>
#include <cutcells/ho_mesh_part_output.h>

namespace cutfemx
{

template <std::floating_point T>
std::vector<cutcells::LevelSetFunction<T, std::int32_t>>
build_entity_level_sets(const CutData<T>& cut_data,
                        std::span<const std::int32_t> entities,
                        int entity_dim, cutcells::cell::type cell_type);

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

std::string cutcells_cell_type_name(cutcells::cell::type cell_type)
{
  switch (cell_type)
  {
  case cutcells::cell::type::point:
    return "point";
  case cutcells::cell::type::interval:
    return "interval";
  case cutcells::cell::type::triangle:
    return "triangle";
  case cutcells::cell::type::tetrahedron:
    return "tetrahedron";
  case cutcells::cell::type::quadrilateral:
    return "quadrilateral";
  case cutcells::cell::type::hexahedron:
    return "hexahedron";
  case cutcells::cell::type::prism:
    return "prism";
  case cutcells::cell::type::pyramid:
    return "pyramid";
  }
  return "unknown";
}

bool is_algoim_backend(cutcells::output::QuadratureBackend backend)
{
  return backend == cutcells::output::QuadratureBackend::AlgoimBernstein
         || backend == cutcells::output::QuadratureBackend::AlgoimGeneral;
}

bool is_algoim_supported_host_cell(cutcells::cell::type cell_type)
{
  return cell_type == cutcells::cell::type::interval
         || cell_type == cutcells::cell::type::quadrilateral
         || cell_type == cutcells::cell::type::hexahedron;
}

template <std::floating_point T>
void validate_algoim_backend_support(
    const CutData<T>& cut_data, cutcells::output::QuadratureBackend backend)
{
  if (!is_algoim_backend(backend))
    return;

  if (!cut_data.mesh_view.has_uniform_cell_type)
  {
    throw std::invalid_argument(
        "Algoim quadrature backends require a uniform host cell type");
  }

  const cutcells::cell::type host_cell_type =
      cut_data.mesh_view.uniform_cell_type;
  if (!is_algoim_supported_host_cell(host_cell_type))
  {
    throw std::invalid_argument(
        "Algoim quadrature backends require interval, quadrilateral, or "
        "hexahedron host cells; got "
        + cutcells_cell_type_name(host_cell_type) + ".");
  }

  if (cut_data.gdim < cut_data.tdim)
  {
    throw std::invalid_argument(
        "Algoim quadrature backends require host cells with "
        "gdim >= tdim; got gdim=" + std::to_string(cut_data.gdim)
        + " and tdim=" + std::to_string(cut_data.tdim) + ".");
  }
}

std::vector<std::int32_t>
identity_cell_reordering(const dolfinx::graph::AdjacencyList<std::int32_t>& graph)
{
  std::vector<std::int32_t> order(graph.num_nodes());
  std::iota(order.begin(), order.end(), 0);
  return order;
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

cutcells::cell::type entity_cell_type(dolfinx::mesh::CellType cell_type,
                                      int entity_dim)
{
  return mesh::dolfinx_to_cutcells_cell_type(
      dolfinx::mesh::cell_entity_type(cell_type, entity_dim, 0));
}

template <std::floating_point T>
std::vector<T>
compact_dof_coordinates(const CutData<T>& cut_data,
                        const dolfinx::fem::Function<T>& level_set)
{
  const auto function_space = level_set.function_space();
  std::vector<T> dof_coordinates_3
      = function_space->tabulate_dof_coordinates(false);
  const std::size_t num_dofs = dof_coordinates_3.size() / 3;
  if (cut_data.gdim == 3)
    return dof_coordinates_3;

  std::vector<T> dof_coordinates(num_dofs
                                 * static_cast<std::size_t>(cut_data.gdim));
  for (std::size_t dof = 0; dof < num_dofs; ++dof)
  {
    for (int d = 0; d < cut_data.gdim; ++d)
    {
      dof_coordinates[dof * static_cast<std::size_t>(cut_data.gdim) + d]
          = dof_coordinates_3[3 * dof + static_cast<std::size_t>(d)];
    }
  }
  return dof_coordinates;
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
  case cutcells::Relation::LessEqual:
    return domain == cutcells::cell::domain::inside
           || domain == cutcells::cell::domain::intersected;
  case cutcells::Relation::GreaterThan:
    return domain == cutcells::cell::domain::outside;
  case cutcells::Relation::GreaterEqual:
    return domain == cutcells::cell::domain::outside
           || domain == cutcells::cell::domain::intersected;
  case cutcells::Relation::EqualTo:
    return domain == cutcells::cell::domain::intersected;
  }
  return false;
}

template <std::floating_point T>
std::int32_t host_parent_index(const CutData<T>& cut_data,
                               std::int32_t host_index)
{
  if (host_index < 0
      || static_cast<std::size_t>(host_index)
             >= static_cast<std::size_t>(cut_data.num_local_cells))
  {
    throw std::runtime_error("CutFEMx parent index is out of range");
  }

  if (cut_data.parent_entities.empty())
    return host_index;

  return cut_data.parent_entities[static_cast<std::size_t>(host_index)];
}

template <std::floating_point T>
void remap_parent_map_to_background_entities(
    std::vector<std::int32_t>& parent_map, const CutData<T>& cut_data)
{
  if (cut_data.parent_entities.empty())
    return;

  for (std::int32_t& parent : parent_map)
    parent = host_parent_index(cut_data, parent);
}

template <std::floating_point T>
std::vector<T> physical_points_for_host_mesh(
    const cutcells::quadrature::QuadratureRules<T>& rules,
    const CutData<T>& cut_data)
{
  const std::size_t num_points = rules._weights.size();
  std::vector<T> physical_points(static_cast<std::size_t>(cut_data.gdim)
                                     * num_points,
                                 T(0));
  if (num_points == 0)
    return physical_points;

  if (!cut_data.mesh_view.has_uniform_cell_type)
  {
    throw std::runtime_error(
        "CutFEMx runtime quadrature requires a uniform host entity type");
  }

  std::vector<int> connectivity;
  connectivity.reserve(static_cast<std::size_t>(cut_data.mesh_view.cell_count)
                       * static_cast<std::size_t>(cut_data.mesh_view.cell_width));
  std::vector<int> offset;
  offset.reserve(static_cast<std::size_t>(cut_data.mesh_view.cell_count) + 1);
  offset.push_back(0);
  for (std::int32_t cell = 0; cell < cut_data.mesh_view.cell_count; ++cell)
  {
    const std::int32_t begin = cell * cut_data.mesh_view.cell_stride;
    for (std::int32_t j = 0; j < cut_data.mesh_view.cell_width; ++j)
    {
      connectivity.push_back(
          cut_data.mesh_view.connectivity[static_cast<std::size_t>(begin + j)]);
    }
    offset.push_back(static_cast<int>(connectivity.size()));
  }

  const int vtk_type = static_cast<int>(
      cutcells::cell::map_cell_type_to_vtk(cut_data.mesh_view.uniform_cell_type));
  std::vector<int> vtk_types(static_cast<std::size_t>(cut_data.mesh_view.cell_count),
                             vtk_type);
  std::vector<T> row_major_points = cutcells::quadrature::physical_points(
      rules, cut_data.mesh_view.coordinates, std::span<const int>(connectivity),
      std::span<const int>(offset), std::span<const int>(vtk_types));

  for (std::size_t q = 0; q < num_points; ++q)
  {
    for (int d = 0; d < cut_data.gdim; ++d)
    {
      physical_points[static_cast<std::size_t>(d) * num_points + q]
          = row_major_points[q * 3 + static_cast<std::size_t>(d)];
    }
  }
  return physical_points;
}

} // namespace

template <std::floating_point T>
std::vector<T> cell_reference_points(
    const dolfinx::fem::FiniteElement<T>& element, int entity_dim,
    int mesh_tdim, int expected_dofs)
{
  if (entity_dim != mesh_tdim)
    return {};

  auto [points, shape] = element.interpolation_points();
  if (shape[1] != static_cast<std::size_t>(entity_dim))
    return {};
  if (shape[0] != static_cast<std::size_t>(expected_dofs))
    return {};
  return points;
}

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

  auto x_dofmap = cut_data.mesh_owner->geometry().dofmaps().front();
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
void build_entity_mesh_view(CutData<T>& cut_data,
                            std::span<const std::int32_t> entities,
                            int entity_dim)
{
  const int background_tdim = cut_data.mesh_owner->topology()->dim();
  if (entity_dim <= 0 || entity_dim > background_tdim)
  {
    throw std::invalid_argument(
        "cutfemx::cut entity_dim must select positive-dimensional mesh entities");
  }

  auto topology = cut_data.mesh_owner->topology_mutable();
  if (entity_dim < background_tdim)
  {
    topology->create_entities(entity_dim);
    topology->create_connectivity(entity_dim, background_tdim);
    topology->create_connectivity(background_tdim, entity_dim);
  }

  auto entity_map = cut_data.mesh_owner->topology()->index_map(entity_dim);
  if (!entity_map)
    throw std::runtime_error("Mesh topology has no entity index map");
  validate_local_entities(
      entities, static_cast<std::int32_t>(entity_map->size_local()));

  const auto background_cell_type = cut_data.mesh_owner->topology()->cell_type();
  const auto cell_type = entity_cell_type(background_cell_type, entity_dim);

  auto [entity_geometry, geometry_shape]
      = dolfinx::mesh::entities_to_geometry(*cut_data.mesh_owner, entity_dim,
                                            entities, false);

  cut_data.tdim = entity_dim;
  cut_data.num_local_cells = static_cast<std::int32_t>(entities.size());
  cut_data.parent_entities.assign(entities.begin(), entities.end());
  cut_data.owned_connectivity = std::move(entity_geometry);

  cut_data.mesh_view = {};
  cut_data.mesh_view.gdim = cut_data.gdim;
  cut_data.mesh_view.tdim = cut_data.tdim;
  cut_data.mesh_view.coordinates = cut_data.mesh_owner->geometry().x();
  cut_data.mesh_view.coordinate_stride = 3;
  cut_data.mesh_view.connectivity = std::span<const std::int32_t>(
      cut_data.owned_connectivity.data(), cut_data.owned_connectivity.size());
  cut_data.mesh_view.cell_count = cut_data.num_local_cells;
  cut_data.mesh_view.cell_width = static_cast<std::int32_t>(geometry_shape[1]);
  cut_data.mesh_view.cell_stride = static_cast<std::int32_t>(geometry_shape[1]);
  cut_data.mesh_view.uniform_cell_type = cell_type;
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

  std::vector<T> dof_coordinates
      = compact_dof_coordinates(cut_data, *level_set_owner);

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
  level_set_mesh_data.cell_reference_points = cell_reference_points<T>(
      *element, cut_data.tdim, cut_data.tdim, static_cast<int>(dofmap_width));
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
CutData<T> cut(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
               std::span<const std::int32_t> entities, int entity_dim)
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
                    level_sets.data(), level_sets.size()),
                entities, entity_dim);
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
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets,
    std::span<const std::int32_t> entities, int entity_dim)
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

  return cut<T>(std::move(mesh), level_sets, entities, entity_dim);
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
    std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets,
    std::span<const std::int32_t> entities, int entity_dim)
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

  build_entity_mesh_view(cut_data, entities, entity_dim);

  const auto cell_type = cut_data.mesh_view.uniform_cell_type;
  cut_data.level_sets = build_entity_level_sets(
      cut_data, cut_data.parent_entities, cut_data.tdim, cell_type);
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
  cutcells::SelectionExpr expr = cutcells::parse_selection_expr(ls_part);
  cutcells::compile_selection_expr(expr, cut_data.level_set_names);

  std::vector<std::int32_t> marked_entities;
  marked_entities.reserve(static_cast<std::size_t>(cut_data.num_local_cells));

  for (std::int32_t host_cell = 0; host_cell < cut_data.num_local_cells;
       ++host_cell)
  {
    bool entity_matches = false;
    for (const auto& term : expr.terms)
    {
      bool term_matches = true;
      for (const auto& clause : term.clauses)
      {
        if (clause.level_set_index < 0
            || static_cast<std::size_t>(clause.level_set_index)
                   >= cut_data.level_set_names.size())
        {
          throw std::runtime_error(
              "Compiled selector contains an invalid level-set index");
        }

        const cutcells::cell::domain domain =
            cut_data.parent_cells.domain(clause.level_set_index, host_cell);
        if (!relation_matches_domain(domain, clause.relation))
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
      marked_entities.push_back(host_parent_index(cut_data, host_cell));
  }

  return marked_entities;
}

template <std::floating_point T>
std::vector<std::int32_t> interior_facets_for_cells(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::int32_t> cells, bool include_ghosts)
{
  if (!mesh)
    throw std::runtime_error("Cannot locate interior facets without a mesh.");

  const int tdim = mesh->topology()->dim();
  if (tdim < 1)
    throw std::runtime_error("Interior facets require mesh tdim >= 1.");
  const int fdim = tdim - 1;

  mesh->topology_mutable()->create_entities(fdim);
  mesh->topology_mutable()->create_connectivity(tdim, fdim);
  mesh->topology_mutable()->create_connectivity(fdim, tdim);

  auto c_to_f = mesh->topology()->connectivity(tdim, fdim);
  auto f_to_c = mesh->topology()->connectivity(fdim, tdim);
  if (!c_to_f || !f_to_c)
    throw std::runtime_error("Facet-cell connectivity is unavailable.");

  auto cell_map = mesh->topology()->index_map(tdim);
  auto facet_map = mesh->topology()->index_map(fdim);
  if (!cell_map || !facet_map)
    throw std::runtime_error("Mesh index maps are unavailable.");

  const std::int32_t num_cells = static_cast<std::int32_t>(
      cell_map->size_local() + cell_map->num_ghosts());
  const std::int32_t num_owned_facets
      = static_cast<std::int32_t>(facet_map->size_local());

  std::vector<std::uint8_t> selected_cells(num_cells, 0);
  for (std::int32_t cell : cells)
  {
    if (cell < 0 || cell >= num_cells)
      throw std::out_of_range("Cell index is out of range.");
    selected_cells[cell] = 1;
  }

  std::vector<std::int32_t> facets;
  for (std::int32_t cell : cells)
  {
    for (std::int32_t facet : c_to_f->links(cell))
    {
      if (!include_ghosts && facet >= num_owned_facets)
        continue;

      std::span<const std::int32_t> adjacent_cells = f_to_c->links(facet);
      if (adjacent_cells.size() != 2)
        continue;

      if (std::ranges::all_of(adjacent_cells,
                              [&](std::int32_t adjacent)
                              {
                                return adjacent >= 0 && adjacent < num_cells
                                       && selected_cells[adjacent] != 0;
                              }))
      {
        facets.push_back(facet);
      }
    }
  }

  std::ranges::sort(facets);
  auto first_duplicate = std::unique(facets.begin(), facets.end());
  facets.erase(first_duplicate, facets.end());
  return facets;
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

std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
flatten_entity_dofs(const std::vector<std::vector<std::int32_t>>& entity_dofs)
{
  std::vector<std::int32_t> flat;
  std::vector<std::int32_t> offsets;
  offsets.reserve(entity_dofs.size() + 1);
  offsets.push_back(0);
  for (const auto& dofs : entity_dofs)
  {
    flat.insert(flat.end(), dofs.begin(), dofs.end());
    offsets.push_back(static_cast<std::int32_t>(flat.size()));
  }
  return {std::move(flat), std::move(offsets)};
}

template <std::floating_point T>
std::vector<cutcells::LevelSetFunction<T, std::int32_t>>
build_entity_level_sets(const CutData<T>& cut_data,
                        std::span<const std::int32_t> entities,
                        int entity_dim, cutcells::cell::type cell_type)
{
  std::vector<cutcells::LevelSetFunction<T, std::int32_t>> level_sets;
  level_sets.reserve(cut_data.level_set_owners.size());
  std::vector<cutcells::cell::type> cell_types(entities.size(), cell_type);

  for (std::size_t ls = 0; ls < cut_data.level_set_owners.size(); ++ls)
  {
    const auto& level_set_owner = cut_data.level_set_owners[ls];
    const auto function_space = level_set_owner->function_space();
    const auto dofmap = function_space->dofmap();
    std::vector<std::vector<std::int32_t>> entity_dofs(entities.size());
    fem::create_entity_dofmap(entities, entity_dim, cut_data.mesh_owner,
                              dofmap, entity_dofs);

    auto [flat_dofs, offsets] = flatten_entity_dofs(entity_dofs);
    std::vector<T> reference_points;
    if (!entity_dofs.empty())
    {
      reference_points = cell_reference_points<T>(
          *function_space->element(), entity_dim, cut_data.tdim,
          static_cast<int>(entity_dofs.front().size()));
    }
    auto mesh_data = cutcells::create_level_set_mesh_data<T, std::int32_t>(
        cut_data.gdim, entity_dim, function_space->element()->basix_element().degree(),
        compact_dof_coordinates(cut_data, *level_set_owner),
        std::span<const std::int32_t>(flat_dofs.data(), flat_dofs.size()),
        std::span<const std::int32_t>(offsets.data(), offsets.size()),
        std::span<const cutcells::cell::type>(cell_types.data(),
                                              cell_types.size()),
        std::span<const T>(reference_points.data(), reference_points.size()));

    level_sets.push_back(cutcells::create_level_set_function<T, std::int32_t>(
        std::move(mesh_data), level_set_owner->x()->array(),
        cut_data.level_set_names[ls]));
  }
  return level_sets;
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
  if (cutcells_mesh._offset.size()
      != static_cast<std::size_t>(cutcells_mesh._num_cells + 1))
  {
    throw std::runtime_error("CutCells visualisation mesh has invalid offsets");
  }

  const bool single_cell_type = std::ranges::all_of(
      cutcells_mesh._types, [first_type](cutcells::cell::type type)
      { return type == first_type; });

  const bool split_mixed_surface_to_triangles =
      !single_cell_type
      && std::ranges::all_of(
             cutcells_mesh._types,
             [](cutcells::cell::type type)
             {
               return type == cutcells::cell::type::triangle
                      || type == cutcells::cell::type::quadrilateral;
             });

  if (!single_cell_type && !split_mixed_surface_to_triangles)
  {
    throw std::runtime_error(
        "CutFEMx currently supports single-cell-type cut meshes only");
  }

  const dolfinx::mesh::CellType cell_type =
      split_mixed_surface_to_triangles
          ? dolfinx::mesh::CellType::triangle
          : mesh::cutcells_to_dolfinx_cell_type(first_type);
  const int vertices_per_cell = dolfinx::mesh::num_cell_vertices(cell_type);

  const std::int64_t num_vertices_local = cutcells_mesh._num_vertices;
  std::int64_t vertex_offset = 0;
  MPI_Exscan(&num_vertices_local, &vertex_offset, 1, MPI_INT64_T, MPI_SUM,
             out._bg_mesh->comm());

  std::vector<std::int64_t> cells;
  std::vector<std::int32_t> parent_index;
  if (split_mixed_surface_to_triangles)
  {
    int num_output_cells = 0;
    for (const cutcells::cell::type type : cutcells_mesh._types)
      num_output_cells += type == cutcells::cell::type::quadrilateral ? 2 : 1;
    cells.reserve(static_cast<std::size_t>(num_output_cells
                                           * vertices_per_cell));
    parent_index.reserve(static_cast<std::size_t>(num_output_cells));
  }
  else
  {
    cells.reserve(static_cast<std::size_t>(cutcells_mesh._num_cells
                                           * vertices_per_cell));
  }

  for (int c = 0; c < cutcells_mesh._num_cells; ++c)
  {
    const int begin = cutcells_mesh._offset[static_cast<std::size_t>(c)];
    const int end = cutcells_mesh._offset[static_cast<std::size_t>(c) + 1];
    const int arity = end - begin;

    if (split_mixed_surface_to_triangles)
    {
      const cutcells::cell::type type =
          cutcells_mesh._types[static_cast<std::size_t>(c)];
      const std::int32_t parent =
          cutcells_mesh._parent_map[static_cast<std::size_t>(c)];
      if (type == cutcells::cell::type::triangle)
      {
        if (arity != 3)
        {
          throw std::runtime_error(
              "CutCells triangle visualisation cell has invalid arity");
        }
        for (int j = begin; j < end; ++j)
        {
          cells.push_back(static_cast<std::int64_t>(
                              cutcells_mesh._connectivity[static_cast<std::size_t>(
                                  j)])
                          + vertex_offset);
        }
        parent_index.push_back(parent);
      }
      else
      {
        if (arity != 4)
        {
          throw std::runtime_error(
              "CutCells quadrilateral visualisation cell has invalid arity");
        }
        // CutCells stores quadrilateral vertices in Basix order
        // (0, 1, 2, 3), while the usual VTK split assumes (0, 1, 3, 2).
        // Split the Basix-ordered quad along the corresponding diagonal.
        const std::array<int, 6> triangle_vertices{0, 1, 3, 0, 3, 2};
        for (const int local : triangle_vertices)
        {
          cells.push_back(static_cast<std::int64_t>(
                              cutcells_mesh._connectivity[static_cast<std::size_t>(
                                  begin + local)])
                          + vertex_offset);
        }
        parent_index.push_back(parent);
        parent_index.push_back(parent);
      }
      continue;
    }

    if (arity != vertices_per_cell)
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

  out._parent_index = split_mixed_surface_to_triangles
                          ? std::move(parent_index)
                          : std::move(cutcells_mesh._parent_map);
  out._is_cut_cell.assign(static_cast<std::size_t>(cutcells_mesh._num_cells),
                          std::int8_t(1));
  if (split_mixed_surface_to_triangles)
    out._is_cut_cell.assign(out._parent_index.size(), std::int8_t(1));
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
  remap_parent_map_to_background_entities(cutcells_mesh._parent_map, cut_data);
  return dolfinx_cut_mesh_from_cutcells_mesh(
      cut_data.mesh_owner, std::move(cutcells_mesh), num_uncut_cells);
}

template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature_from_rules(
    const CutData<T>& cut_data,
    cutcells::quadrature::QuadratureRules<T>&& rules,
    RuntimeSurfaceProvenance&& surface_provenance = {});

namespace
{
int single_equality_level_set_index(const cutcells::SelectionExpr& expr)
{
  if (expr.terms.size() != 1)
    return -1;
  const auto& term = expr.terms.front();
  if (term.clauses.size() != 1)
    return -1;
  const auto& clause = term.clauses.front();
  if (clause.relation != cutcells::Relation::EqualTo)
    return -1;
  return clause.level_set_index;
}

template <std::floating_point T>
RuntimeSurfaceProvenance make_surface_provenance(
    const cutcells::HOMeshPart<T, std::int32_t>& part,
    std::string_view selector, std::size_t num_rules)
{
  RuntimeSurfaceProvenance provenance;
  provenance.selector = std::string(selector);
  provenance.level_set_index = single_equality_level_set_index(part.expr);
  if (provenance.level_set_index < 0)
    return provenance;
  if (part.dim != part.mesh->tdim - 1)
    return provenance;

  const std::vector<cutcells::output::SelectedZeroEntityInfo> infos
      = cutcells::output::selected_zero_entity_infos(part);
  if (infos.size() != num_rules)
  {
    throw std::runtime_error(
        "Surface-normal provenance is not aligned with runtime quadrature "
        "rules. This is only supported for straight codimension-one cut "
        "quadrature in the first pass.");
  }

  provenance.cut_cell_ids.reserve(infos.size());
  provenance.parent_cell_ids.reserve(infos.size());
  provenance.local_zero_entity_ids.reserve(infos.size());
  provenance.dimensions.reserve(infos.size());
  for (const auto& info : infos)
  {
    provenance.cut_cell_ids.push_back(info.cut_cell_id);
    provenance.parent_cell_ids.push_back(info.parent_cell_id);
    provenance.local_zero_entity_ids.push_back(info.local_zero_entity_id);
    provenance.dimensions.push_back(info.dimension);
  }
  return provenance;
}
} // namespace

template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature(const CutData<T>& cut_data,
                                        std::string_view ls_part, int order,
                                        std::string_view backend)
{
  validate_quadrature_order(order);

  const auto parsed_backend
      = cutcells::output::quadrature_backend_from_string(backend);
  validate_algoim_backend_support(cut_data, parsed_backend);

  auto part = select_mesh_part(cut_data.mesh_view, cut_data.cut_cells,
                               cut_data.parent_cells, ls_part);
  cutcells::quadrature::QuadratureRules<T> rules
      = cutcells::output::quadrature_rules<T, std::int32_t>(
          part, order, /*include_uncut_cells=*/false, parsed_backend);
  RuntimeSurfaceProvenance surface_provenance;
  if (parsed_backend == cutcells::output::QuadratureBackend::Straight)
  {
    surface_provenance
        = make_surface_provenance(part, ls_part, rules._parent_map.size());
  }
  return runtime_quadrature_from_rules(
      cut_data, std::move(rules), std::move(surface_provenance));
}

template <std::floating_point T>
RuntimeQuadrature<T> runtime_quadrature_from_rules(
    const CutData<T>& cut_data,
    cutcells::quadrature::QuadratureRules<T>&& rules,
    RuntimeSurfaceProvenance&& surface_provenance)
{
  std::optional<std::vector<T>> physical_points;
  if (!cut_data.parent_entities.empty())
    physical_points = physical_points_for_host_mesh(rules, cut_data);
  remap_parent_map_to_background_entities(rules._parent_map, cut_data);
  if (physical_points)
  {
    return RuntimeQuadrature<T>(
        std::move(rules), cut_data.mesh_owner, std::move(*physical_points),
        cut_data.gdim, std::move(surface_provenance));
  }
  return RuntimeQuadrature<T>(
      std::move(rules), cut_data.mesh_owner, std::move(surface_provenance));
}

template <std::floating_point T>
std::vector<std::pair<std::string, RuntimeQuadrature<T>>> runtime_quadratures(
    const CutData<T>& cut_data, std::span<const std::string> ls_parts,
    int order, std::string_view backend)
{
  validate_quadrature_order(order);

  const auto parsed_backend
      = cutcells::output::quadrature_backend_from_string(backend);
  validate_algoim_backend_support(cut_data, parsed_backend);

  std::vector<std::pair<std::string, cutcells::HOMeshPart<T, std::int32_t>>>
      parts;
  parts.reserve(ls_parts.size());
  for (const std::string& ls_part : ls_parts)
  {
    parts.emplace_back(
        ls_part,
        select_mesh_part(cut_data.mesh_view, cut_data.cut_cells,
                         cut_data.parent_cells, ls_part));
  }

  auto named_rules = cutcells::output::paired_quadrature_rules<T, std::int32_t>(
      parts, order, /*include_uncut_cells=*/false, parsed_backend);

  std::vector<std::pair<std::string, RuntimeQuadrature<T>>> out;
  out.reserve(named_rules.size());
  std::unordered_map<std::string, const cutcells::HOMeshPart<T, std::int32_t>*>
      part_by_name;
  for (const auto& [name, part] : parts)
    part_by_name.emplace(name, &part);
  for (auto& [name, rules] : named_rules)
  {
    RuntimeSurfaceProvenance surface_provenance;
    if (parsed_backend == cutcells::output::QuadratureBackend::Straight)
    {
      auto it = part_by_name.find(name);
      if (it != part_by_name.end())
      {
        surface_provenance = make_surface_provenance(
            *it->second, name, rules._parent_map.size());
      }
    }
    out.emplace_back(
        std::move(name),
        runtime_quadrature_from_rules(
            cut_data, std::move(rules), std::move(surface_provenance)));
  }
  return out;
}

template CutData<double> cut(
    std::shared_ptr<const dolfinx::fem::Function<double>>);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::fem::Function<float>>);
template CutData<double> cut(
    std::shared_ptr<const dolfinx::fem::Function<double>>,
    std::span<const std::int32_t>, int);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::fem::Function<float>>,
    std::span<const std::int32_t>, int);
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
    std::span<const std::shared_ptr<const dolfinx::fem::Function<double>>>,
    std::span<const std::int32_t>, int);
template CutData<float> cut(
    std::span<const std::shared_ptr<const dolfinx::fem::Function<float>>>,
    std::span<const std::int32_t>, int);
template CutData<double> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>>,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<double>>>);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<float>>,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<float>>>);
template CutData<double> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>>,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<double>>>,
    std::span<const std::int32_t>, int);
template CutData<float> cut(
    std::shared_ptr<const dolfinx::mesh::Mesh<float>>,
    std::span<const std::shared_ptr<const dolfinx::fem::Function<float>>>,
    std::span<const std::int32_t>, int);
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

template std::vector<std::int32_t> interior_facets_for_cells(
    std::shared_ptr<const dolfinx::mesh::Mesh<double>>,
    std::span<const std::int32_t>, bool);
template std::vector<std::int32_t> interior_facets_for_cells(
    std::shared_ptr<const dolfinx::mesh::Mesh<float>>,
    std::span<const std::int32_t>, bool);

template mesh::CutMesh<double> create_cut_mesh(
    const CutData<double>&, std::string_view, std::string_view);
template mesh::CutMesh<float> create_cut_mesh(
    const CutData<float>&, std::string_view, std::string_view);

template RuntimeQuadrature<double> runtime_quadrature(
    const CutData<double>&, std::string_view, int, std::string_view);
template RuntimeQuadrature<float> runtime_quadrature(
    const CutData<float>&, std::string_view, int, std::string_view);

template std::vector<std::pair<std::string, RuntimeQuadrature<double>>>
runtime_quadratures(
    const CutData<double>&, std::span<const std::string>, int,
    std::string_view);
template std::vector<std::pair<std::string, RuntimeQuadrature<float>>>
runtime_quadratures(
    const CutData<float>&, std::span<const std::string>, int,
    std::string_view);

} // namespace cutfemx
