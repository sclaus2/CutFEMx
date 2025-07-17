// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/DofMap.h>

#include <span>
#include <vector>
#include <memory>
#include <concepts>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <spdlog/spdlog.h>

namespace cutfemx::extensions
{
  /// DOF-to-root mapping functionality for discrete extension
  /// Provides functions to build mappings from DOF indices to cells and root cells

/// @brief Build mapping from DOF indices to vectors of cut cell indices containing that DOF
/// @tparam T Floating point type for geometry
/// @param cut_cell_parents Vector of cut cell parent indices
/// @param root Mapping from cut cell parents to root cells (from build_root_mapping)
/// @param function_space Function space for which to build the dof mapping
/// @return Mapping from DOF indices to vectors of cut cell indices that contain each DOF
/// @note This function iterates over all cut cell parents and their corresponding root cells
///       to collect which cells contain each DOF. This is the inverse of the cell_dofs mapping.
template <std::floating_point T>
std::unordered_map<std::int32_t, std::vector<std::int32_t>>
build_dof_to_cells_mapping(
    const std::vector<std::int32_t>& cut_cell_parents,
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
    
    spdlog::info("Building dof-to-cells mapping for {} cut cell parents and {} root mappings", 
                 cut_cell_parents.size(), root.size());
    
    std::unordered_set<std::int32_t> processed_cells;
    
    // Iterate over all cut cell parents
    for (std::int32_t cut_cell_parent : cut_cell_parents)
    {
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

/// @brief Build dof-to-root mapping by selecting the closest root cell for each DOF
/// @tparam T Floating point type for geometry
/// @param dof_to_cells_mapping Mapping from DOF indices to vectors of cut cell indices
/// @param cut_cell_to_root_mapping Mapping from cut cell parents to root cells
/// @param dof_coords Vector of DOF coordinates (flattened, 3 components per DOF)
/// @param function_space Function space for DOF coordinate information
/// @param mesh Mesh object for computing cell midpoints
/// @return Mapping from DOF indices to closest root cell indices
/// @note For each DOF, this function finds all possible root cells (via the cut cells that contain the DOF),
///       computes the distance from the DOF coordinate to each root cell midpoint, and selects the closest one.
template <std::floating_point T>
std::unordered_map<std::int32_t, std::int32_t>
build_dof_to_root_mapping(
    const std::unordered_map<std::int32_t, std::vector<std::int32_t>>& dof_to_cells_mapping,
    const std::unordered_map<std::int32_t, std::int32_t>& root_mapping,
    const std::vector<T>& dof_coords,
    const dolfinx::fem::FunctionSpace<T>& function_space,
    const dolfinx::mesh::Mesh<T>& mesh)
{
    std::unordered_map<std::int32_t, std::int32_t> dof_to_root;
    
    if (dof_to_cells_mapping.empty())
    {
        spdlog::info("Empty dof_to_cells_mapping, returning empty dof_to_root mapping");
        return dof_to_root;
    }
    
    // Use provided DOF coordinates
    const int gdim = 3; // DOLFINx always stores coordinates as 3D
    
    spdlog::info("Building dof-to-root mapping for {} DOFs", dof_to_cells_mapping.size());
    
    // Process each DOF
    for (const auto& [dof_index, cut_cells] : dof_to_cells_mapping)
    {
        // Get DOF coordinate
        std::array<T, 3> dof_coord;
        for (int d = 0; d < gdim; ++d)
        {
            dof_coord[d] = dof_coords[dof_index * gdim + d];
        }
        
        // Find all unique root cells for this DOF's cut cells
        std::unordered_set<std::int32_t> candidate_roots;
        for (std::int32_t cut_cell : cut_cells)
        {
            auto root_it = root_mapping.find(cut_cell);
            if (root_it != root_mapping.end())
            {
                candidate_roots.insert(root_it->second);
            }
        }
        
        if (candidate_roots.empty())
        {
            spdlog::warn("No root cells found for DOF {}, skipping", dof_index);
            continue;
        }
        
        // Convert to vector for midpoint computation
        std::vector<std::int32_t> root_cells(candidate_roots.begin(), candidate_roots.end());
        
        // Compute midpoints of candidate root cells
        std::vector<T> midpoints = dolfinx::mesh::compute_midpoints(mesh, mesh.topology()->dim(), root_cells);
        
        // Find root cell with midpoint closest to DOF
        std::int32_t closest_root = root_cells[0];
        T min_distance_sq = std::numeric_limits<T>::max();
        
        for (std::size_t i = 0; i < root_cells.size(); ++i)
        {
            // Compute squared distance from DOF to root cell midpoint
            T distance_sq = 0.0;
            for (int d = 0; d < gdim; ++d)
            {
                T diff = dof_coord[d] - midpoints[i * gdim + d];
                distance_sq += diff * diff;
            }
            
            if (distance_sq < min_distance_sq)
            {
                min_distance_sq = distance_sq;
                closest_root = root_cells[i];
            }
        }
        
        // Store the mapping
        dof_to_root[dof_index] = closest_root;
        
        spdlog::debug("DOF {} mapped to root cell {} (distance: {:.6f})", 
                     dof_index, closest_root, std::sqrt(min_distance_sq));
    }
    
    spdlog::info("Dof-to-root mapping created: {} DOF-root pairs", dof_to_root.size());
    
    return dof_to_root;
}
    
} // namespace cutfemx::extensions
