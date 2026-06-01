// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include "cell_aggregation.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <cutcells/ho_mesh_part_output.h>

#include <dolfinx/mesh/Topology.h>

namespace cutfemx::extensions
{
namespace
{
std::string normalize_selector(std::string_view selector)
{
  std::string out;
  out.reserve(selector.size());
  for (char c : selector)
    if (!std::isspace(static_cast<unsigned char>(c)))
      out.push_back(c);
  return out;
}

struct StrictSelector
{
  std::string name;
  char relation;
};

StrictSelector parse_strict_selector(std::string_view selector)
{
  const std::string text = normalize_selector(selector);
  const std::size_t less = text.find('<');
  const std::size_t greater = text.find('>');
  const bool has_less = less != std::string::npos;
  const bool has_greater = greater != std::string::npos;
  if (has_less == has_greater)
  {
    throw std::invalid_argument(
        "CellAggregation v1 expects a strict single level-set selector such "
        "as 'phi < 0' or 'phi > 0'.");
  }

  const std::size_t pos = has_less ? less : greater;
  if (pos == 0 || pos + 1 >= text.size() || text[pos + 1] == '='
      || text.substr(pos + 1) != "0")
  {
    throw std::invalid_argument(
        "CellAggregation v1 expects a selector of the form 'name < 0' or "
        "'name > 0'.");
  }
  return {text.substr(0, pos), text[pos]};
}

template <typename T>
std::vector<T> sorted_unique(std::vector<T> values)
{
  std::ranges::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

template <std::floating_point T>
std::vector<std::vector<std::int32_t>>
cell_neighbors(const dolfinx::mesh::Mesh<T>& mesh, std::int32_t num_cells)
{
  const int tdim = mesh.topology()->dim();
  if (tdim < 1)
    throw std::runtime_error("CellAggregation requires tdim >= 1.");
  const int fdim = tdim - 1;

  auto topology = mesh.topology_mutable();
  topology->create_entities(fdim);
  topology->create_connectivity(tdim, fdim);
  topology->create_connectivity(fdim, tdim);

  auto c_to_f = mesh.topology()->connectivity(tdim, fdim);
  auto f_to_c = mesh.topology()->connectivity(fdim, tdim);
  if (!c_to_f || !f_to_c)
    throw std::runtime_error("Could not create cell-facet connectivity.");

  std::vector<std::vector<std::int32_t>> neighbors(
      static_cast<std::size_t>(num_cells));
  for (std::int32_t cell = 0; cell < num_cells; ++cell)
  {
    for (std::int32_t facet : c_to_f->links(cell))
    {
      for (std::int32_t other : f_to_c->links(facet))
      {
        if (other != cell && other >= 0 && other < num_cells)
          neighbors[static_cast<std::size_t>(cell)].push_back(other);
      }
    }
    neighbors[static_cast<std::size_t>(cell)]
        = sorted_unique(std::move(neighbors[static_cast<std::size_t>(cell)]));
  }
  return neighbors;
}

template <std::floating_point T>
void validate_original_cell_cut(const cutfemx::CutData<T>& cut_data)
{
  if (!cut_data.parent_entities.empty())
  {
    throw std::invalid_argument(
        "CellAggregation v1 only supports cuts on the original background "
        "cell mesh, not selected entity subsets.");
  }
  if (!cut_data.mesh_owner)
    throw std::invalid_argument("CellAggregation requires CutData with a mesh.");
  if (cut_data.tdim != cut_data.mesh_owner->topology()->dim())
  {
    throw std::invalid_argument(
        "CellAggregation v1 only supports full-dimensional cell cuts.");
  }
}

} // namespace

RootPolicy root_policy_from_string(std::string_view policy)
{
  const std::string text(policy);
  if (text == "interior_only")
    return RootPolicy::interior_only;
  if (text == "interior_or_well_cut")
    return RootPolicy::interior_or_well_cut;
  throw std::invalid_argument(
      "Unknown root policy. Expected 'interior_only' or "
      "'interior_or_well_cut'.");
}

template <std::floating_point T>
CellAggregation<T> create_cell_aggregation(
    const cutfemx::CutData<T>& cut_data, std::string_view selector,
    T volume_fraction_threshold, RootPolicy root_policy, int max_iterations,
    bool allow_rootless)
{
  validate_original_cell_cut(cut_data);
  if (volume_fraction_threshold < T(0) || volume_fraction_threshold > T(1))
    throw std::invalid_argument("Volume fraction threshold must be in [0, 1].");

  const StrictSelector parsed = parse_strict_selector(selector);
  if (std::find(cut_data.level_set_names.begin(),
                cut_data.level_set_names.end(),
                parsed.name) == cut_data.level_set_names.end())
    throw std::invalid_argument("CellAggregation selector level set is unknown.");

  const std::string strict_selector = parsed.name + parsed.relation + "0";
  const std::string interface_selector = parsed.name + "=0";

  CellAggregation<T> aggregation;
  aggregation.cut_volume_fraction.assign(
      static_cast<std::size_t>(cut_data.num_local_cells), T(0));
  aggregation.root_cell.assign(static_cast<std::size_t>(cut_data.num_local_cells),
                               std::int32_t(-1));
  aggregation.aggregate_id.assign(
      static_cast<std::size_t>(cut_data.num_local_cells), std::int32_t(-1));
  aggregation.propagation_depth.assign(
      static_cast<std::size_t>(cut_data.num_local_cells), std::int32_t(-1));

  aggregation.interior_cells = locate_entities(cut_data, strict_selector);
  aggregation.cut_cells = locate_entities(cut_data, interface_selector);
  aggregation.active_cells = aggregation.interior_cells;
  aggregation.active_cells.insert(aggregation.active_cells.end(),
                                  aggregation.cut_cells.begin(),
                                  aggregation.cut_cells.end());
  aggregation.active_cells = sorted_unique(std::move(aggregation.active_cells));

  auto part = cutcells::select_part(cut_data.mesh_view, cut_data.cut_cells,
                                    cut_data.parent_cells, strict_selector);
  auto [fraction_parents, fractions]
      = cutcells::output::volume_fractions(part);
  std::unordered_map<std::int32_t, T> selected_fraction;
  for (std::size_t i = 0; i < fraction_parents.size(); ++i)
    selected_fraction[static_cast<std::int32_t>(fraction_parents[i])] = fractions[i];

  std::unordered_set<std::int32_t> root_set;
  for (std::int32_t cell : aggregation.interior_cells)
    root_set.insert(cell);

  for (std::int32_t cell : aggregation.cut_cells)
  {
    const T fraction = selected_fraction.contains(cell) ? selected_fraction[cell] : T(0);
    aggregation.cut_volume_fraction[static_cast<std::size_t>(cell)] = fraction;

    if (root_policy == RootPolicy::interior_or_well_cut
        && fraction >= volume_fraction_threshold)
    {
      root_set.insert(cell);
    }
    else
    {
      aggregation.ill_posed_cells.push_back(cell);
    }
  }

  aggregation.well_posed_cells.assign(root_set.begin(), root_set.end());
  aggregation.well_posed_cells
      = sorted_unique(std::move(aggregation.well_posed_cells));
  aggregation.ill_posed_cells
      = sorted_unique(std::move(aggregation.ill_posed_cells));

  std::int32_t next_aggregate = 0;
  for (std::int32_t root : aggregation.well_posed_cells)
  {
    aggregation.root_cell[static_cast<std::size_t>(root)] = root;
    aggregation.aggregate_id[static_cast<std::size_t>(root)] = next_aggregate++;
    aggregation.propagation_depth[static_cast<std::size_t>(root)] = 0;
  }

  const std::unordered_set<std::int32_t> active_set(aggregation.active_cells.begin(),
                                                    aggregation.active_cells.end());
  const auto neighbors = cell_neighbors(*cut_data.mesh_owner, cut_data.num_local_cells);
  const int iteration_limit
      = max_iterations < 0 ? cut_data.num_local_cells : max_iterations;

  for (int iteration = 0; iteration < iteration_limit; ++iteration)
  {
    int mapped_this_iteration = 0;
    for (std::int32_t cell : aggregation.ill_posed_cells)
    {
      if (aggregation.root_cell[static_cast<std::size_t>(cell)] >= 0)
        continue;

      for (std::int32_t other : neighbors[static_cast<std::size_t>(cell)])
      {
        if (!active_set.contains(other))
          continue;
        const std::int32_t root =
            aggregation.root_cell[static_cast<std::size_t>(other)];
        if (root < 0)
          continue;

        aggregation.root_cell[static_cast<std::size_t>(cell)] = root;
        aggregation.aggregate_id[static_cast<std::size_t>(cell)]
            = aggregation.aggregate_id[static_cast<std::size_t>(other)];
        aggregation.propagation_depth[static_cast<std::size_t>(cell)]
            = aggregation.propagation_depth[static_cast<std::size_t>(other)] + 1;
        ++mapped_this_iteration;
        break;
      }
    }
    if (mapped_this_iteration == 0)
      break;
  }

  for (std::int32_t cell : aggregation.ill_posed_cells)
  {
    if (aggregation.root_cell[static_cast<std::size_t>(cell)] < 0)
      aggregation.rootless_cells.push_back(cell);
  }

  if (!allow_rootless && !aggregation.rootless_cells.empty())
  {
    throw std::runtime_error(
        "CellAggregation found active ill-posed cells without an admissible "
        "root. Adjust the root policy or threshold, or explicitly allow "
        "rootless aggregation for diagnostics.");
  }

  return aggregation;
}

template CellAggregation<double> create_cell_aggregation(
    const cutfemx::CutData<double>&, std::string_view, double, RootPolicy, int,
    bool);
template CellAggregation<float> create_cell_aggregation(
    const cutfemx::CutData<float>&, std::string_view, float, RootPolicy, int,
    bool);

} // namespace cutfemx::extensions
