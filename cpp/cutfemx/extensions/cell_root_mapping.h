// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/common/types.h>

#include <cutcells/cell_flags.h>
#include <cutcells/cut_mesh.h>
#include <cutcells/cut_cell.h>
#include "../mesh/cut_mesh.h"

#include <span>
#include <vector>
#include <memory>
#include <concepts>
#include <unordered_set>
#include <unordered_map>
#include <spdlog/spdlog.h>

namespace cutfemx::extensions
{
  /// Cell-to-root mapping functionality for discrete extension
  /// Provides functions to identify badly cut cells and build mappings from cut cells to interior root cells
  
  /// @brief Identify badly cut cells based on volume threshold and return their parent indices
  /// @tparam U Floating point type for geometry
  /// @param cut_cells CutCells object containing all cut cells (must be triangulated to compute volume)
  /// @param mesh_size_h Characteristic mesh size
  /// @param threshold_factor Factor to multiply with mesh size (default 0.1 for Ch threshold)
  /// @return Vector of parent cell indices for badly cut cells
  /// @note The CutCells must be triangulated for accurate volume computation
  /// @todo Implement computation of volume for different cell shapes (currently assumes triangular cells)
  template <std::floating_point U>
  std::vector<std::int32_t>
  get_badly_cut_parent_indices(const cutcells::mesh::CutCells<U>& cut_cells,
                               const U mesh_size_h,
                               const U threshold_factor = 0.1)
  {
    std::vector<std::int32_t> badly_cut_parent_indices;
    
    // Compute volume threshold
    U volume_threshold = threshold_factor * mesh_size_h * mesh_size_h * 0.5;

    for (std::size_t i = 0; i < cut_cells._cut_cells.size(); ++i)
    {
      const auto& cut_cell = cut_cells._cut_cells[i];
      
      // Compute volume of this cut cell
      U cut_volume = cutcells::cell::volume(cut_cell);
      
      // Check if volume is below threshold
      if (cut_volume < volume_threshold)
      {
        // Get the parent cell index for this cut cell
        std::int32_t parent_index = cut_cells._parent_map[i];
        badly_cut_parent_indices.push_back(parent_index);
      }
    }
    
    return badly_cut_parent_indices;
  }

/// @brief Build mapping from cut cell parents to interior root cells using connectivity-based propagation
/// @tparam U Floating point type for mesh coordinates
/// @param badly_cut_parents Vector of badly cut cell parent indices
/// @param interior_cells Vector of interior cell indices (cells completely inside the level set)
/// @param mesh The mesh containing topology information
/// @param cut_cell_parents Vector of all cut cell parent indices
/// @param root Output mapping from cut cell parents to interior cell indices
/// @param max_iterations Maximum number of propagation iterations (default 10)
/// @note This function maps ALL cut cells to interior cells, not just badly cut ones
/// Algorithm:
/// 1. Use cut_cell_parents directly and create efficient lookup set
/// 2. Find cut cells that directly share facets with interior cells - these get direct mapping
/// 3. Iteratively propagate: find unmapped cut cells connected to mapped cut cells via facets
/// 4. Continue until all cut cells are mapped or max iterations reached
template <std::floating_point U>
void build_root_mapping(const std::vector<std::int32_t>& badly_cut_parents,
                               const std::vector<std::int32_t>& interior_cells,
                               const dolfinx::mesh::Mesh<U>& mesh,
                               const std::vector<std::int32_t>& cut_cell_parents,
                               std::unordered_map<std::int32_t, std::int32_t>& root,
                               int max_iterations = 10)
{ 
  // Build connectivity information for finding neighboring cells
  auto topology = mesh.topology_mutable();
  const int tdim = topology->dim();
  
  // Ensure we have cell-to-cell connectivity via facets
  topology->create_connectivity(tdim - 1, tdim);
  topology->create_connectivity(tdim, tdim - 1);
  
  // Get connectivity arrays
  auto cell_to_facet = topology->connectivity(tdim, tdim - 1);
  auto facet_to_cell = topology->connectivity(tdim - 1, tdim);
  
  if (!cell_to_facet || !facet_to_cell)
  {
    spdlog::error("Failed to create mesh connectivity");
    return;
  }
  
  // Create efficient lookups
  std::unordered_set<std::int32_t> cut_cell_set(cut_cell_parents.begin(), cut_cell_parents.end());
  std::unordered_set<std::int32_t> interior_cell_set(interior_cells.begin(), interior_cells.end());
  
  spdlog::info("Starting root mapping with {} cut cell parents, {} interior cells", 
               cut_cell_parents.size(), interior_cells.size());
  
  // Phase 1: Find cut cells directly connected to interior cells
  int direct_mappings = 0;
  for (std::int32_t cut_cell : cut_cell_parents)
  {
    // Get facets of this cut cell
    auto cut_cell_facets = cell_to_facet->links(cut_cell);
    
    // Check each facet for connections to interior cells
    for (auto facet : cut_cell_facets)
    {
      auto facet_cells = facet_to_cell->links(facet);
      
      // Look for interior cells sharing this facet
      for (auto neighbor_cell : facet_cells)
      {
        if (neighbor_cell != cut_cell && interior_cell_set.count(neighbor_cell))
        {
          // Direct connection found - map cut cell to this interior cell
          root[cut_cell] = neighbor_cell;
          direct_mappings++;
          goto next_cut_cell;  // Move to next cut cell once mapped
        }
      }
    }
    next_cut_cell:;
  }
  
  spdlog::info("Phase 1 complete: {} cut cells mapped directly to interior cells", direct_mappings);
  
  // Phase 2: Iterative propagation for remaining unmapped cut cells
  int propagation_mappings = 0;
  for (int iteration = 0; iteration < max_iterations; ++iteration)
  {
    int mappings_this_iteration = 0;
    
    for (std::int32_t cut_cell : cut_cell_parents)
    {
      // Skip already mapped cells
      if (root.count(cut_cell))
        continue;
      
      // Get facets of this unmapped cut cell
      auto cut_cell_facets = cell_to_facet->links(cut_cell);
      
      // Check each facet for connections to already mapped cut cells
      for (auto facet : cut_cell_facets)
      {
        auto facet_cells = facet_to_cell->links(facet);
        
        // Look for mapped cut cells sharing this facet
        for (auto neighbor_cell : facet_cells)
        {
          if (neighbor_cell != cut_cell && cut_cell_set.count(neighbor_cell) && root.count(neighbor_cell))
          {
            // Found a mapped cut cell neighbor - inherit its root
            root[cut_cell] = root[neighbor_cell];
            mappings_this_iteration++;
            propagation_mappings++;
            goto next_unmapped_cell;  // Move to next cut cell once mapped
          }
        }
      }
      next_unmapped_cell:;
    }
    
    spdlog::info("Iteration {}: {} new mappings", iteration + 1, mappings_this_iteration);
    
    // Stop if no new mappings were made
    if (mappings_this_iteration == 0)
      break;
  }
  
  // Phase 3: Fallback for any remaining unmapped cut cells
  int fallback_mappings = 0;
  if (!interior_cells.empty())
  {
    std::int32_t fallback_root = interior_cells[0];  // Use first interior cell as fallback
    
    for (std::int32_t cut_cell : cut_cell_parents)
    {
      if (!root.count(cut_cell))
      {
        root[cut_cell] = fallback_root;
        fallback_mappings++;
      }
    }
    
    if (fallback_mappings > 0)
    {
      spdlog::warn("Fallback: {} unmapped cut cells assigned to interior cell {}", 
                   fallback_mappings, fallback_root);
    }
  }
  
  int unmapped = cut_cell_parents.size() - root.size();
  
  spdlog::info("build_root_mapping complete: {}/{} cut cell parents mapped ({} direct, {} propagated, {} fallback, {} unmapped)", 
               root.size(), cut_cell_parents.size(), direct_mappings, propagation_mappings, fallback_mappings, unmapped);
}

/// @brief Create inverse mapping from root cells to cut cells
/// @param cell_to_root_mapping Input mapping from cut cells to interior cells
/// @return Inverse mapping from interior cells to vectors of cut cells
/// @note Each interior cell can map to multiple cut cells
std::unordered_map<std::int32_t, std::vector<std::int32_t>>
create_root_to_cell_map(const std::unordered_map<std::int32_t, std::int32_t>& cell_to_root_mapping)
{
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> root_to_cells;
    
    // Reserve space for efficiency
    root_to_cells.reserve(cell_to_root_mapping.size() / 2); // Estimate
    
    spdlog::info("Creating inverse mapping from {} cut-to-root entries", cell_to_root_mapping.size());
    
    // Iterate through original mapping and build inverse
    for (const auto& [cut_cell, root_cell] : cell_to_root_mapping)
    {
        // Add the cut cell to the vector of cells mapped to this root
        root_to_cells[root_cell].push_back(cut_cell);
    }
    
    // Log statistics
    std::size_t total_mapped_cells = 0;
    std::size_t max_cells_per_root = 0;
    for (const auto& [root_cell, cut_cells] : root_to_cells)
    {
        total_mapped_cells += cut_cells.size();
        max_cells_per_root = std::max(max_cells_per_root, cut_cells.size());
    }
    
    double avg_cells_per_root = root_to_cells.empty() ? 0.0 : 
                               static_cast<double>(total_mapped_cells) / root_to_cells.size();
    
    spdlog::info("Inverse mapping created: {} root cells, {} total mapped cells, "
                 "avg {:.2f} cells per root, max {} cells per root", 
                 root_to_cells.size(), total_mapped_cells, avg_cells_per_root, max_cells_per_root);
    
    return root_to_cells;
}

} // namespace cutfemx::extensions
