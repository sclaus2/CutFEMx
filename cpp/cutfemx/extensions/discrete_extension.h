// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/common/types.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/DofMap.h>

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
  /// Discrete extension of solutions from interior nodes to exterior nodes
  /// following the approach described in https://arxiv.org/pdf/2101.10052
  
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

/// @brief Build mapping from cut cell parents to interior cells using propagation algorithm
/// @param badly_cut_parents Vector of parent indices for badly cut cells (not used in new algorithm)
/// @param interior_cells Vector of interior cell indices
/// @param mesh DOLFINx mesh object
/// @param cut_cells CutCells object containing cut cell information
/// @param root Output mapping from cut cell parent to interior cell
/// @param max_iterations Maximum iterations for propagation (default: 10)
/// 
/// Algorithm:
/// 1. Use cut_cells._parent_map directly (no copying) and create efficient lookup set
/// 2. Find cut cells that directly share facets with interior cells - these get direct mapping
/// 3. Iteratively propagate: find unmapped cut cells connected to mapped cut cells via facets
/// 4. Continue until all cut cells are mapped or max iterations reached
template <std::floating_point U>
void build_root_mapping(const std::vector<std::int32_t>& badly_cut_parents,
                               const std::vector<std::int32_t>& interior_cells,
                               const dolfinx::mesh::Mesh<U>& mesh,
                               const cutcells::mesh::CutCells<U>& cut_cells,
                               std::unordered_map<std::int32_t, std::int32_t>& root,
                               int max_iterations = 10)
{ 
  // Build connectivity information for finding neighboring cells
  auto topology = mesh.topology_mutable();
  const int tdim = topology->dim();
  
  // Ensure we have cell-to-cell connectivity via facets
  topology->create_connectivity(tdim - 1, tdim);
  topology->create_connectivity(tdim, tdim - 1);
  
  auto c_to_f = topology->connectivity(tdim, tdim - 1);
  auto f_to_c = topology->connectivity(tdim - 1, tdim);
  
  // For efficient lookup of interior cells
  std::unordered_set<std::int32_t> interior_set(interior_cells.begin(), interior_cells.end());
  
  // Create set of cut cell parents for efficient lookup (directly from parent_map)
  std::unordered_set<std::int32_t> cut_cell_parent_set(cut_cells._parent_map.begin(), cut_cells._parent_map.end());
  
  // Track which cut cell parents have been treated
  std::unordered_set<std::int32_t> treated_cut_cells;
  
  spdlog::info("Starting root mapping for {} cut cell parents with {} interior cells", 
               cut_cell_parent_set.size(), interior_cells.size());
  
  // Phase 1: Find cut cell parents that share a facet with interior cells
  int direct_mappings = 0;
  
  for (std::int32_t cut_cell_parent : cut_cell_parent_set)
  {
    // Get facets of this cut cell parent
    auto facets = c_to_f->links(cut_cell_parent);
    
    // Look for interior cells sharing these facets
    std::int32_t found_interior = -1;
    
    for (auto facet : facets)
    {
      auto cells_on_facet = f_to_c->links(facet);
      
      for (auto neighbor_cell : cells_on_facet)
      {
        if (neighbor_cell != cut_cell_parent && 
            interior_set.find(neighbor_cell) != interior_set.end())
        {
          found_interior = neighbor_cell;
          break;
        }
      }
      
      if (found_interior != -1)
        break;
    }
    
    // If we found a directly connected interior cell, store the mapping
    if (found_interior != -1)
    {
      root[cut_cell_parent] = found_interior;
      treated_cut_cells.insert(cut_cell_parent);
      direct_mappings++;
      
      spdlog::debug("Direct mapping: cut cell parent {} → interior cell {}", 
                   cut_cell_parent, found_interior);
    }
  }
  
  spdlog::info("Phase 1 complete: {} direct mappings to interior cells", direct_mappings);
  
  // Phase 2: For remaining cut cell parents, find connections to already-mapped cut cell parents
  int propagated_mappings = 0;
  bool progress_made = true;
  int iteration = 0;
  
  while (progress_made && iteration < max_iterations)
  {
    progress_made = false;
    iteration++;
    int mappings_this_iteration = 0;
    
    spdlog::debug("Phase 2, iteration {}: {} cut cell parents still need mapping", 
                 iteration, cut_cell_parent_set.size() - treated_cut_cells.size());
    
    for (std::int32_t cut_cell_parent : cut_cell_parent_set)
    {
      if (treated_cut_cells.find(cut_cell_parent) != treated_cut_cells.end())
        continue; // Already treated
        
      // Get facets of this cut cell parent
      auto facets = c_to_f->links(cut_cell_parent);
      
      // Look for already-mapped cut cell parents sharing these facets
      std::int32_t found_root = -1;
      
      for (auto facet : facets)
      {
        auto cells_on_facet = f_to_c->links(facet);
        
        for (auto neighbor_cell : cells_on_facet)
        {
          if (neighbor_cell != cut_cell_parent && 
              cut_cell_parent_set.find(neighbor_cell) != cut_cell_parent_set.end() &&
              treated_cut_cells.find(neighbor_cell) != treated_cut_cells.end())
          {
            // This neighbor is a cut cell parent that has already been mapped
            found_root = root[neighbor_cell];
            break;
          }
        }
        
        if (found_root != -1)
          break;
      }
      
      // If we found a connected cut cell parent with a root, use the same root
      if (found_root != -1)
      {
        root[cut_cell_parent] = found_root;
        treated_cut_cells.insert(cut_cell_parent);
        propagated_mappings++;
        mappings_this_iteration++;
        progress_made = true;
        
        spdlog::debug("Propagated mapping: cut cell parent {} → root cell {} (iteration {})", 
                     cut_cell_parent, found_root, iteration);
      }
    }
    
    spdlog::info("Iteration {}: {} new propagated mappings", iteration, mappings_this_iteration);
  }
  
  spdlog::info("Phase 2 complete: {} propagated mappings over {} iterations", 
               propagated_mappings, iteration);
  
  // Phase 3: Fallback for any remaining unmapped cut cells
  int fallback_mappings = 0;
  int remaining_unmapped = cut_cell_parent_set.size() - (direct_mappings + propagated_mappings);
  
  if (remaining_unmapped > 0 && !interior_cells.empty())
  {
    spdlog::info("Phase 3: Applying fallback mapping for {} isolated cut cell parents", remaining_unmapped);
    
    // Use the closest interior cell (for simplicity, use the first one)
    std::int32_t fallback_interior = interior_cells[0];
    
    for (std::int32_t cut_cell_parent : cut_cell_parent_set)
    {
      if (treated_cut_cells.find(cut_cell_parent) == treated_cut_cells.end())
      {
        root[cut_cell_parent] = fallback_interior;
        treated_cut_cells.insert(cut_cell_parent);
        fallback_mappings++;
        
        spdlog::debug("Fallback mapping: isolated cut cell parent {} → interior cell {}", 
                     cut_cell_parent, fallback_interior);
      }
    }
    
    spdlog::info("Phase 3 complete: {} fallback mappings applied", fallback_mappings);
  }
  
  // Summary statistics
  int total_mappings = direct_mappings + propagated_mappings + fallback_mappings;
  int unmapped_cells = cut_cell_parent_set.size() - total_mappings;
  
  spdlog::info("build_root_mapping complete: {}/{} cut cell parents mapped ({} direct, {} propagated, {} fallback, {} unmapped)", 
               total_mappings, cut_cell_parent_set.size(), direct_mappings, propagated_mappings, fallback_mappings, unmapped_cells);
  
  if (unmapped_cells > 0)
  {
    spdlog::warn("{} cut cell parents could not be mapped to interior cells", unmapped_cells);
  }
}
    
/// @brief Create inverse mapping from interior cells to cut cells
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

/// @brief Build dof-to-cells mapping from cut cells and root cells
/// @tparam T Floating point type for geometry
/// @param cut_cells CutCells object containing all cut cells
/// @param root Mapping from cut cell parents to root cells
/// @param function_space Function space for which to build the dof mapping
/// @return Mapping from DOF indices to vectors of cell indices containing that DOF
/// @note This function iterates over all cut cells and their corresponding root cells
///       to collect which cells contain each DOF. This is the inverse of the cell_dofs mapping.
template <std::floating_point T>
std::unordered_map<std::int32_t, std::vector<std::int32_t>>
build_dof_to_cells_mapping(
    const cutcells::mesh::CutCells<T>& cut_cells,
    const std::unordered_map<std::int32_t, std::int32_t>& root,
    const dolfinx::fem::FunctionSpace<T>& function_space)
{
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> dof_to_cells;
    
    // Get the dofmap from the function space
    auto dofmap = function_space.dofmap();
    if (!dofmap)
    {
        spdlog::error("Function space does not have a valid dofmap");
        return dof_to_cells;
    }
    
    spdlog::info("Building dof-to-cells mapping for {} cut cells and {} root mappings", 
                 cut_cells._cut_cells.size(), root.size());
    
    std::unordered_set<std::int32_t> processed_cells;
    
    // Iterate over all cut cell parents from the cut_cells
    for (std::size_t i = 0; i < cut_cells._cut_cells.size(); ++i)
    {
        std::int32_t cut_cell_parent = cut_cells._parent_map[i];
        
        // Find the root cell for this cut cell parent
        auto root_it = root.find(cut_cell_parent);
        if (root_it == root.end())
        {
            spdlog::warn("Cut cell parent {} not found in root mapping, skipping", cut_cell_parent);
            continue;
        }
        std::int32_t root_cell = root_it->second;
        
        // Skip if we've already processed this cut cell parent
        if (processed_cells.find(cut_cell_parent) != processed_cells.end())
            continue;
        processed_cells.insert(cut_cell_parent);
        
        // Get DOFs for the cut cell parent
        std::span<const std::int32_t> cut_cell_dofs = dofmap->cell_dofs(cut_cell_parent);
        std::span<const std::int32_t> root_cell_dofs = dofmap->cell_dofs(root_cell);
        
        // Add the cut cell parent to each DOF's cell list
        for (std::int32_t dof : cut_cell_dofs)
        {
            //check if DOF is in root cell in which case we skip it as dof is interior and does not need to be constrained
            if (std::find(root_cell_dofs.begin(), root_cell_dofs.end(), dof) != root_cell_dofs.end())
                continue;

            // Add the cut cell parent to the list of cells for this DOF
            dof_to_cells[dof].push_back(cut_cell_parent);
        }
    }
    
    // Log statistics
    std::size_t total_dofs = dof_to_cells.size();
    std::size_t total_dof_cell_pairs = 0;
    std::size_t max_cells_per_dof = 0;
    
    for (const auto& [dof, cells] : dof_to_cells)
    {
        total_dof_cell_pairs += cells.size();
        max_cells_per_dof = std::max(max_cells_per_dof, cells.size());
    }
    
    double avg_cells_per_dof = total_dofs > 0 ? 
                              static_cast<double>(total_dof_cell_pairs) / total_dofs : 0.0;
    
    spdlog::info("Dof-to-cells mapping created: {} unique DOFs, {} total dof-cell pairs, "
                 "avg {:.2f} cells per DOF, max {} cells per DOF", 
                 total_dofs, total_dof_cell_pairs, avg_cells_per_dof, max_cells_per_dof);
    
    return dof_to_cells;
}
    
} // namespace cutfemx::extensions



