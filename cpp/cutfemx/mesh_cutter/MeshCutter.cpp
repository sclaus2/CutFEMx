// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include "MeshCutter.h"

#include "../mesh/convert.h"

#include <algorithm>
#include <array>
#include <basix/element-families.h>
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
#include <stdexcept>

#include <cutcells/ho_mesh_part_output.h>

namespace cutfemx
{
namespace
{
constexpr std::uint8_t inside_flag = 1u << 0;
constexpr std::uint8_t outside_flag = 1u << 1;
constexpr std::uint8_t intersected_flag = 1u << 2;

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

std::uint8_t selector_mask(std::string_view ls_part)
{
  const std::string selector = normalize_selector(ls_part);
  if (selector == "phi<0")
    return inside_flag;
  if (selector == "phi>0")
    return outside_flag;
  if (selector == "phi=0")
    return intersected_flag;
  if (selector == "phi<=0")
    return inside_flag | intersected_flag;
  if (selector == "phi>=0")
    return outside_flag | intersected_flag;

  throw std::invalid_argument(
      "Unsupported level-set selector. Expected phi<0, phi>0, phi=0, "
      "phi<=0, or phi>=0.");
}

bool is_inclusive_selector(std::string_view ls_part)
{
  const std::string selector = normalize_selector(ls_part);
  return selector == "phi<=0" || selector == "phi>=0";
}

bool is_interface_selector(std::string_view ls_part)
{
  return normalize_selector(ls_part) == "phi=0";
}

bool selector_matches(cutcells::cell::domain domain, std::uint8_t mask)
{
  using cutcells::cell::domain;
  switch (domain)
  {
  case domain::inside:
    return (mask & inside_flag) != 0;
  case domain::outside:
    return (mask & outside_flag) != 0;
  case domain::intersected:
    return (mask & intersected_flag) != 0;
  default:
    return false;
  }
}

bool include_uncut_cells(std::string_view ls_part, std::string_view mode)
{
  const std::string normalized_mode = normalize_selector(mode);
  if (is_inclusive_selector(ls_part))
  {
    throw std::invalid_argument(
        "MeshCutter::create_cut_mesh rejects inclusive selectors. Use phi<0 "
        "or phi>0 with mode='full' instead.");
  }

  if (normalized_mode == "cut_only")
    return false;

  if (normalized_mode == "full")
  {
    if (is_interface_selector(ls_part))
    {
      throw std::invalid_argument(
          "MeshCutter::create_cut_mesh(..., 'phi=0', mode='full') is not "
          "defined. Use mode='cut_only' for interface meshes.");
    }
    return true;
  }

  throw std::invalid_argument(
      "Unsupported cut-mesh mode. Expected 'full' or 'cut_only'.");
}

void validate_runtime_quadrature_selector(std::string_view ls_part)
{
  if (is_inclusive_selector(ls_part))
  {
    throw std::invalid_argument(
        "MeshCutter::runtime_quadrature rejects inclusive selectors. Use a "
        "strict selector such as phi<0 or phi>0, and combine with "
        "locate_entities(...) for uncut entities.");
  }
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
} // namespace

template <std::floating_point T>
MeshCutter<T>::MeshCutter(
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set)
    : _level_set(std::move(level_set))
{
  if (!_level_set)
    throw std::invalid_argument("MeshCutter requires a level-set function");

  _mesh = _level_set->function_space()->mesh();
  if (!_mesh)
    throw std::invalid_argument(
        "MeshCutter could not infer a mesh from the level-set function");

  _gdim = _mesh->geometry().dim();
  _tdim = _mesh->topology()->dim();

  validate_level_set();
  build_mesh_view();
  build_level_set_mesh_data();
  update();
}

template <std::floating_point T>
std::shared_ptr<const dolfinx::fem::Function<T>> MeshCutter<T>::level_set() const
{
  return _level_set;
}

template <std::floating_point T>
std::shared_ptr<const dolfinx::mesh::Mesh<T>> MeshCutter<T>::mesh() const
{
  return _mesh;
}

template <std::floating_point T>
void MeshCutter<T>::validate_level_set() const
{
  const auto element = _level_set->function_space()->element();
  if (!element)
    throw std::invalid_argument("Level-set function has no finite element");

  if (!element->value_shape().empty())
    throw std::invalid_argument("MeshCutter requires a scalar level-set function");

  const auto& basix_element = element->basix_element();
  if (basix_element.family() != basix::element::family::P)
  {
    throw std::invalid_argument(
        "MeshCutter requires a scalar Lagrange level-set function");
  }
}

template <std::floating_point T>
void MeshCutter<T>::build_mesh_view()
{
  auto cell_map = _mesh->topology()->index_map(_tdim);
  if (!cell_map)
    throw std::runtime_error("Mesh topology has no cell index map");

  _num_local_cells = static_cast<std::int32_t>(cell_map->size_local());

  std::span<const T> x = _mesh->geometry().x();
  const std::size_t num_geometry_nodes = x.size() / 3;
  if (_gdim == 3)
  {
    // DOLFINx stores geometry coordinates as contiguous xyz triples. For 3D,
    // CutCells can use that storage directly; the held mesh owns the lifetime.
    _mesh_coordinates.clear();
    _mesh_view.coordinates = x;
  }
  else
  {
    // DOLFINx pads lower-dimensional coordinates to xyz. CutCells expects
    // compact gdim-tuples, so this is an intentional one-time layout cache.
    _mesh_coordinates.resize(num_geometry_nodes
                             * static_cast<std::size_t>(_gdim));
    for (std::size_t node = 0; node < num_geometry_nodes; ++node)
    {
      for (int d = 0; d < _gdim; ++d)
        _mesh_coordinates[node * static_cast<std::size_t>(_gdim) + d]
            = x[3 * node + static_cast<std::size_t>(d)];
    }
    _mesh_view.coordinates = std::span<const T>(_mesh_coordinates.data(),
                                                _mesh_coordinates.size());
  }

  const auto cell_type = _mesh->topology()->cell_type();
  const auto cutcells_cell_type = mesh::dolfinx_to_cutcells_cell_type(cell_type);
  const int num_vertices = cutcells::cell::get_num_vertices(cutcells_cell_type);

  auto x_dofmap = _mesh->geometry().dofmap();
  if (static_cast<int>(x_dofmap.extent(1)) < num_vertices)
  {
    throw std::runtime_error(
        "Mesh geometry dofmap does not contain enough vertex entries");
  }

  _mesh_offsets.resize(static_cast<std::size_t>(_num_local_cells) + 1);
  _mesh_connectivity.clear();
  _mesh_connectivity.reserve(
      static_cast<std::size_t>(_num_local_cells * num_vertices));

  _mesh_offsets[0] = 0;
  for (std::int32_t c = 0; c < _num_local_cells; ++c)
  {
    for (int j = 0; j < num_vertices; ++j)
      _mesh_connectivity.push_back(x_dofmap(c, j));

    _mesh_offsets[static_cast<std::size_t>(c) + 1]
        = static_cast<std::int32_t>(_mesh_connectivity.size());
  }

  _mesh_cell_types.assign(static_cast<std::size_t>(_num_local_cells),
                          cutcells_cell_type);

  _mesh_view.gdim = _gdim;
  _mesh_view.tdim = _tdim;
  _mesh_view.connectivity = std::span<const std::int32_t>(
      _mesh_connectivity.data(), _mesh_connectivity.size());
  _mesh_view.offsets
      = std::span<const std::int32_t>(_mesh_offsets.data(), _mesh_offsets.size());
  _mesh_view.cell_types = std::span<const cutcells::cell::type>(
      _mesh_cell_types.data(), _mesh_cell_types.size());
  _mesh_view.vtk_vertex_order = false;
}

template <std::floating_point T>
void MeshCutter<T>::build_level_set_mesh_data()
{
  const auto function_space = _level_set->function_space();
  const auto element = function_space->element();
  const int degree = element->basix_element().degree();

  std::vector<T> dof_coordinates_3
      = function_space->tabulate_dof_coordinates(false);
  const std::size_t num_dofs = dof_coordinates_3.size() / 3;
  if (_gdim == 3)
  {
    _level_set_dof_coordinates = std::move(dof_coordinates_3);
  }
  else
  {
    // tabulate_dof_coordinates returns xyz triples. Cache compact coordinates
    // once because CutCells consumes compact gdim-tuples repeatedly.
    _level_set_dof_coordinates.resize(num_dofs
                                      * static_cast<std::size_t>(_gdim));
    for (std::size_t dof = 0; dof < num_dofs; ++dof)
    {
      for (int d = 0; d < _gdim; ++d)
      {
        _level_set_dof_coordinates[dof * static_cast<std::size_t>(_gdim) + d]
            = dof_coordinates_3[3 * dof + static_cast<std::size_t>(d)];
      }
    }
  }

  const auto dofmap = function_space->dofmap();
  _level_set_cell_offsets.resize(static_cast<std::size_t>(_num_local_cells) + 1);
  _level_set_cell_dofs.clear();
  _level_set_cell_offsets[0] = 0;
  for (std::int32_t c = 0; c < _num_local_cells; ++c)
  {
    const std::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
    _level_set_cell_dofs.insert(_level_set_cell_dofs.end(), dofs.begin(),
                                dofs.end());
    _level_set_cell_offsets[static_cast<std::size_t>(c) + 1]
        = static_cast<std::int32_t>(_level_set_cell_dofs.size());
  }

  _level_set_mesh_data = std::make_shared<
      cutcells::LevelSetMeshData<T, std::int32_t>>(
      cutcells::create_level_set_mesh_data<T, std::int32_t>(
          _gdim, _tdim, degree,
          std::span<const T>(_level_set_dof_coordinates.data(),
                             _level_set_dof_coordinates.size()),
          std::span<const std::int32_t>(_level_set_cell_dofs.data(),
                                        _level_set_cell_dofs.size()),
          std::span<const std::int32_t>(_level_set_cell_offsets.data(),
                                        _level_set_cell_offsets.size()),
          std::span<const cutcells::cell::type>(_mesh_cell_types.data(),
                                                _mesh_cell_types.size())));
}

template <std::floating_point T>
cutcells::LevelSetFunction<T, std::int32_t>
MeshCutter<T>::current_level_set_function() const
{
  const std::span<const T> dof_values = _level_set->x()->array();
  return cutcells::create_level_set_function<T, std::int32_t>(
      _level_set_mesh_data, dof_values, "phi");
}

template <std::floating_point T>
void MeshCutter<T>::update()
{
  auto ls = current_level_set_function();
  auto [cut_cells, background_data] = cutcells::cut<T, std::int32_t>(
      _mesh_view, ls, true);

  _cut_cells = std::move(cut_cells);
  _background_data = std::move(background_data);
  _background_data.mesh = &_mesh_view;
}

template <std::floating_point T>
std::vector<std::int32_t>
MeshCutter<T>::locate_entities(std::string_view ls_part) const
{
  const std::uint8_t mask = selector_mask(ls_part);
  std::vector<std::int32_t> marked_entities;
  marked_entities.reserve(static_cast<std::size_t>(_num_local_cells));
  for (std::int32_t entity = 0; entity < _num_local_cells; ++entity)
  {
    const cutcells::cell::domain domain = _background_data.domain(0, entity);
    if (selector_matches(domain, mask))
      marked_entities.push_back(entity);
  }

  return marked_entities;
}

template <std::floating_point T>
std::vector<std::int32_t>
MeshCutter<T>::locate_entities(std::span<const std::int32_t> entities,
                               int entity_dim, std::string_view ls_part) const
{
  if (entity_dim != _tdim)
  {
    throw std::invalid_argument(
        "MeshCutter::locate_entities currently supports cell entities only");
  }

  const std::uint8_t mask = selector_mask(ls_part);
  std::vector<std::int32_t> marked_entities;
  marked_entities.reserve(entities.size());
  for (const std::int32_t entity : entities)
  {
    if (entity < 0 || entity >= _num_local_cells)
      throw std::out_of_range("MeshCutter entity index is out of range");

    const cutcells::cell::domain domain = _background_data.domain(0, entity);
    if (selector_matches(domain, mask))
      marked_entities.push_back(entity);
  }

  return marked_entities;
}

template <std::floating_point T>
cutcells::HOMeshPart<T, std::int32_t> select_mesh_part(
    const cutcells::HOCutCells<T, std::int32_t>& cut_cells,
    const cutcells::BackgroundMeshData<T, std::int32_t>& background_data,
    std::string_view ls_part)
{
  return cutcells::select_part<T, std::int32_t>(cut_cells, background_data,
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
      throw std::out_of_range("MeshCutter entity index is out of range");
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
mesh::CutMesh<T> MeshCutter<T>::create_cut_mesh(std::string_view ls_part,
                                                std::string_view mode) const
{
  const bool include_uncut = include_uncut_cells(ls_part, mode);
  auto part = select_mesh_part(_cut_cells, _background_data, ls_part);
  cutcells::mesh::CutMesh<T> cutcells_mesh
      = cutcells::output::visualization_mesh<T, std::int32_t>(part,
                                                              include_uncut);
  const std::size_t num_uncut_cells
      = include_uncut && part.dim == _tdim ? part.uncut_cell_ids.size() : 0;
  return dolfinx_cut_mesh_from_cutcells_mesh(_mesh, std::move(cutcells_mesh),
                                             num_uncut_cells);
}

template <std::floating_point T>
mesh::CutMesh<T> MeshCutter<T>::create_cut_mesh(
    std::span<const std::int32_t> entities, int entity_dim,
    std::string_view ls_part, std::string_view mode) const
{
  if (entity_dim != _tdim)
  {
    throw std::invalid_argument(
        "MeshCutter::create_cut_mesh currently supports cell entities only");
  }

  const bool include_uncut = include_uncut_cells(ls_part, mode);
  auto part = select_mesh_part(_cut_cells, _background_data, ls_part);
  filter_mesh_part_to_cells(part, entities, _num_local_cells);
  cutcells::mesh::CutMesh<T> cutcells_mesh
      = cutcells::output::visualization_mesh<T, std::int32_t>(part,
                                                              include_uncut);
  const std::size_t num_uncut_cells
      = include_uncut && part.dim == _tdim ? part.uncut_cell_ids.size() : 0;
  return dolfinx_cut_mesh_from_cutcells_mesh(_mesh, std::move(cutcells_mesh),
                                             num_uncut_cells);
}

template <std::floating_point T>
RuntimeQuadrature<T> MeshCutter<T>::runtime_quadrature(
    std::string_view ls_part, int order) const
{
  validate_runtime_quadrature_selector(ls_part);
  validate_quadrature_order(order);

  auto part = select_mesh_part(_cut_cells, _background_data, ls_part);
  cutcells::quadrature::QuadratureRules<T> rules
      = cutcells::output::quadrature_rules<T, std::int32_t>(
          part, order, /*include_uncut_cells=*/false);
  return RuntimeQuadrature<T>(std::move(rules), _mesh);
}

template <std::floating_point T>
RuntimeQuadrature<T> MeshCutter<T>::runtime_quadrature(
    std::span<const std::int32_t> entities, int entity_dim,
    std::string_view ls_part, int order) const
{
  if (entity_dim != _tdim)
  {
    throw std::invalid_argument(
        "MeshCutter::runtime_quadrature currently supports cell entities only");
  }

  validate_runtime_quadrature_selector(ls_part);
  validate_quadrature_order(order);

  auto part = select_mesh_part(_cut_cells, _background_data, ls_part);
  filter_mesh_part_to_cells(part, entities, _num_local_cells);
  cutcells::quadrature::QuadratureRules<T> rules
      = cutcells::output::quadrature_rules<T, std::int32_t>(
          part, order, /*include_uncut_cells=*/false);
  return RuntimeQuadrature<T>(std::move(rules), _mesh);
}

template class MeshCutter<double>;
template class MeshCutter<float>;

} // namespace cutfemx
