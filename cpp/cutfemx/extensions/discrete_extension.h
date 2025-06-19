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

#include <cutcells/cell_flags.h>
#include <cutcells/cut_mesh.h>
#include <cutcells/cut_cell.h>
#include "../mesh/cut_mesh.h"

#include <span>
#include <vector>
#include <memory>
#include <concepts>
#include <unordered_set>
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

/// @brief Build mapping from badly cut parents to interior cells
template <std::floating_point U>
void build_root_to_cut_mapping(const std::vector<std::int32_t>& badly_cut_parents,
                               const std::vector<std::int32_t>& interior_cells,
                               const dolfinx::mesh::Mesh<U>& mesh,
                               const cutcells::mesh::CutCells<U>& cut_cells,
                               std::vector<std::int32_t>& root_to_cut_mapping)
{
  // Reserve memory for efficiency - we store 2 elements per badly cut parent
  root_to_cut_mapping.reserve(2 * badly_cut_parents.size());
  
  // Build connectivity information for finding neighboring cells
  auto topology = mesh.topology_mutable();  // Get mutable topology
  const int tdim = topology->dim();
  
  // Ensure we have cell-to-cell connectivity via facets
  topology->create_connectivity(tdim - 1, tdim);
  topology->create_connectivity(tdim, tdim - 1);
  
  auto c_to_f = topology->connectivity(tdim, tdim - 1);
  auto f_to_c = topology->connectivity(tdim - 1, tdim);
  
  // For efficient lookup of interior cells
  std::unordered_set<std::int32_t> interior_set(interior_cells.begin(), interior_cells.end());
  
  // Create set of all cut cell parent indices for efficient lookup
  std::unordered_set<std::int32_t> cut_cell_parents;
  for (std::size_t i = 0; i < cut_cells._parent_map.size(); ++i)
  {
    cut_cell_parents.insert(cut_cells._parent_map[i]);
  }
  
  // For each badly cut parent, find exactly one nearby interior cell
  int successful_mappings = 0;
  int fallback_used = 0;
  
  for (std::size_t idx = 0; idx < badly_cut_parents.size(); ++idx)
  {
    std::int32_t badly_cut_parent = badly_cut_parents[idx];
    std::int32_t found_interior = -1;
    std::unordered_set<std::int32_t> visited;
    std::vector<std::int32_t> to_visit = {badly_cut_parent};
    visited.insert(badly_cut_parent);
    
    int search_depth = 0;
    
    // Phase 1: Breadth-first search to find interior neighbors via facets
    while (!to_visit.empty() && found_interior == -1)
    {
      std::vector<std::int32_t> current_level = std::move(to_visit);
      to_visit.clear();
      search_depth++;
      
      for (std::int32_t current_cell : current_level)
      {
        // Check if current cell is interior (skip the badly cut parent itself)
        if (current_cell != badly_cut_parent && interior_set.find(current_cell) != interior_set.end())
        {
          found_interior = current_cell;
          break;
        }
        
        // Add neighbors to search via facets
        auto facets = c_to_f->links(current_cell);
        
        for (auto facet : facets)
        {
          auto cells = f_to_c->links(facet);
          for (auto neighbor : cells)
          {
            // Only add neighbor to search if it is a cut cell or interior cell
            bool is_cut_cell = cut_cell_parents.find(neighbor) != cut_cell_parents.end();
            bool is_interior_cell = interior_set.find(neighbor) != interior_set.end();
            
            if (neighbor != current_cell && visited.find(neighbor) == visited.end() && 
                (is_cut_cell || is_interior_cell))
            {
              visited.insert(neighbor);
              to_visit.push_back(neighbor);
            }
          }
        }
      }
      
      if (found_interior != -1) break;
      
      if (search_depth > 10) // Prevent infinite search
      {
        break;
      }
    }
    
    
    // Phase 3: Fallback - use the first interior cell if no connection found
    if (found_interior == -1 && !interior_cells.empty())
    {
      found_interior = interior_cells[0];
      fallback_used++;
    }
    
    // Store the mapping (only if an interior cell was found)
    if (found_interior != -1)
    {
      root_to_cut_mapping.push_back(badly_cut_parent);
      root_to_cut_mapping.push_back(found_interior);
      successful_mappings++;
    }
  }
  
  // Summary output using spdlog
  spdlog::info("build_root_to_cut_mapping: Mapped {}/{} badly cut cells ({} interior cells available, {} fallbacks used)", 
               successful_mappings, badly_cut_parents.size(), interior_cells.size(), fallback_used);
}
    
// template <std::floating_point U>
// void build_extension_matrix(
//     const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& function_space,
//     const std::vector<std::int32_t>& root_to_cut_mapping,
//     dolfinx::la::MatrixCSR<U>& extension_matrix)
// {
//   // Get DOF map and index map
//   auto dofmap = function_space->dofmap();
//   auto index_map = dofmap->index_map;
//   const int bs = dofmap->index_map_bs();
  
//   // Create sparsity pattern for extension matrix
//   std::vector<std::vector<std::int32_t>> sparsity_pattern(index_map->size_local());
  
//   // For each badly cut cell, create entries linking to its single interior cell
//   // Flat vector format: [badly_cut_0, interior_0, badly_cut_1, interior_1, ...]
//   for (std::size_t i = 0; i < root_to_cut_mapping.size(); i += 2)
//   {
//     std::int32_t badly_cut_parent = root_to_cut_mapping[i];
//     std::int32_t interior_cell = root_to_cut_mapping[i + 1];
    
//     // Get DOFs for badly cut cell
//     auto badly_cut_dofs = dofmap->cell_dofs(badly_cut_parent);
    
//     // Get DOFs for the single interior cell
//     auto interior_dofs = dofmap->cell_dofs(interior_cell);
    
//     // Add connections in sparsity pattern
//     for (auto badly_cut_dof : badly_cut_dofs)
//     {
//       for (auto interior_dof : interior_dofs)
//       {
//         sparsity_pattern[badly_cut_dof].push_back(interior_dof);
//       }
//     }
//   }
  
//   // Remove duplicates and sort
//   for (auto& row : sparsity_pattern)
//   {
//     std::sort(row.begin(), row.end());
//     row.erase(std::unique(row.begin(), row.end()), row.end());
//   }
  
//   // Build the matrix with harmonic extension weights
//   // For now, use simple averaging weights (can be improved later)
//   std::vector<U> values;
//   std::vector<std::int32_t> row_ptr = {0};
//   std::vector<std::int32_t> col_indices;
  
//   for (std::size_t i = 0; i < sparsity_pattern.size(); ++i)
//   {
//     const auto& row = sparsity_pattern[i];
    
//     // Check if this DOF belongs to a badly cut cell
//     bool is_badly_cut_dof = false;
//     for (const auto& [badly_cut_parent, interior_cell] : root_to_cut_mapping)
//     {
//       auto dofs = dofmap->cell_dofs(badly_cut_parent);
//       if (std::find(dofs.begin(), dofs.end(), i) != dofs.end())
//       {
//         is_badly_cut_dof = true;
//         break;
//       }
//     }
    
//     if (is_badly_cut_dof && !row.empty())
//     {
//       // Use equal weights for averaging (harmonic extension would require solving)
//       U weight = U(1.0) / static_cast<U>(row.size());
//       for (auto col : row)
//       {
//         col_indices.push_back(col);
//         values.push_back(weight);
//       }
//     }
//     else
//     {
//       // Identity mapping for non-badly-cut DOFs
//       col_indices.push_back(static_cast<std::int32_t>(i));
//       values.push_back(U(1.0));
//     }
    
//     row_ptr.push_back(static_cast<std::int32_t>(col_indices.size()));
//   }
  
//   // Create the sparse matrix
//   _extension_matrix = dolfinx::la::MatrixCSR<U>(
//       dolfinx::MPI::comm_world, 
//       {index_map->size_local(), index_map->size_local()},
//       {row_ptr, col_indices, values}
//   );
// }
//     // Root-to-cut cell mapping: flat vector [badly_cut_0, interior_0, badly_cut_1, interior_1, ...]
//     std::vector<std::int32_t> _root_to_cut_mapping;
    
//     // Sparse extension matrix
//     dolfinx::la::MatrixCSR<U> _extension_matrix;



    
//     /// @brief Build the sparse extension matrix
//     void _build_extension_matrix();


//   /// @brief Averaging projector class for DOF-level averaging
//   /// @tparam U Floating point type for geometry
//   template <std::floating_point U>
//   class AveragingProjector
//   {
//   public:
//     /// @brief Constructor
//     /// @param function_space DOLFINx function space
//     /// @param extension_operator Discrete extension operator
//     AveragingProjector(
//         std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> function_space,
//         const DiscreteExtensionOperator<U>& extension_operator)
//         : _function_space(function_space), _extension_operator(extension_operator)
//     {
//       _build_averaging_matrix();
//     }

//     /// @brief Get the full embedding matrix
//     const dolfinx::la::MatrixCSR<U>& embedding_matrix() const { return _embedding_matrix; }

//     /// @brief Apply averaging projection
//     /// @param extended_values Extended values from DiscreteExtensionOperator
//     /// @return Averaged DOF values
//     dolfinx::la::Vector<U> apply(const dolfinx::la::Vector<U>& extended_values) const;

//     /// @brief Construct full embedding from interior function to extended function
//     /// @param u_interior Function defined on interior nodes
//     /// @return Function defined on both interior and exterior nodes
//     dolfinx::fem::Function<U, U> embed(const dolfinx::fem::Function<U, U>& u_interior) const;

//   private:
//     std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> _function_space;
//     const DiscreteExtensionOperator<U>& _extension_operator;
    
//     // Full embedding matrix combining extension and averaging
//     dolfinx::la::MatrixCSR<U> _embedding_matrix;

//     /// @brief Build the averaging matrix for DOF-level contributions
//     void _build_averaging_matrix();
//   };

//   /// @brief Patch extension solver using local variational problems
//   /// @tparam U Floating point type for geometry
//   template <std::floating_point U>
//   class PatchExtensionSolver
//   {
//   public:
//     /// @brief Constructor
//     /// @param function_space DOLFINx function space
//     /// @param cut_cells CutCells object containing cut cell information
//     /// @param badly_cut_parents Vector of parent indices for badly cut cells
//     /// @param well_cut_parents Vector of parent indices for well cut cells
//     /// @param patch_radius Radius for patch construction (default: 1)
//     PatchExtensionSolver(
//         std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> function_space,
//         const cutcells::mesh::CutCells<U>& cut_cells,
//         const std::vector<std::int32_t>& badly_cut_parents,
//         const std::vector<std::int32_t>& well_cut_parents,
//         int patch_radius = 1)
//         : _function_space(function_space), _cut_cells(cut_cells),
//           _badly_cut_parents(badly_cut_parents), _well_cut_parents(well_cut_parents),
//           _patch_radius(patch_radius)
//     {
//       _build_patches();
//       _precompute_local_problems();
//     }

//     /// @brief Solve local patch problems for extension
//     /// @param u_interior Function defined on interior nodes
//     /// @return Extended function
//     dolfinx::fem::Function<U, U> solve(const dolfinx::fem::Function<U, U>& u_interior) const;

//     /// @brief Get precomputed extension matrix (if available)
//     /// @return Extension matrix for patch-based method
//     const dolfinx::la::MatrixCSR<U>& extension_matrix() const { return _extension_matrix; }

//     /// @brief Build extension matrix by solving all local problems
//     void build_extension_matrix();

//   private:
//     std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> _function_space;
//     const cutcells::mesh::CutCells<U>& _cut_cells;
//     std::vector<std::int32_t> _badly_cut_parents;
//     std::vector<std::int32_t> _well_cut_parents;
//     int _patch_radius;
    
//     // Patch information: maps badly cut cell to its patch
//     std::map<std::int32_t, std::vector<std::int32_t>> _patches;
    
//     // Precomputed extension matrix
//     dolfinx::la::MatrixCSR<U> _extension_matrix;

//     /// @brief Build patches around badly cut cells
//     void _build_patches();
    
//     /// @brief Precompute local variational problems for each patch
//     void _precompute_local_problems();
    
//     /// @brief Solve harmonic extension on a single patch
//     /// @param patch_cells Cells in the patch
//     /// @param badly_cut_cell The badly cut cell to extend to
//     /// @param interior_values Values on interior boundary of patch
//     /// @return Extended values on the patch
//     std::vector<U> _solve_patch_problem(
//         const std::vector<std::int32_t>& patch_cells,
//         std::int32_t badly_cut_cell,
//         const std::map<std::int32_t, U>& interior_values) const;
//   };

  // TODO: Implement these utility functions later
  /*
  /// @brief Extend a function from interior nodes to exterior nodes of the domain
  /// @tparam T Scalar type for the function values
  /// @tparam U Floating point type for geometry
  /// @param u_interior Function defined on interior nodes
  /// @param cut_mesh Cut mesh containing interior/exterior node information
  /// @param extension_method Method for extension (e.g., "harmonic", "constant")
  /// @return Extended function defined on both interior and exterior nodes
  template <dolfinx::scalar T, std::floating_point U>
  dolfinx::fem::Function<T, U> extend_to_exterior(
    const dolfinx::fem::Function<T, U>& u_interior,
    const cutfemx::mesh::CutMesh<U>& cut_mesh,
    const std::string& extension_method = "harmonic");

  /// @brief Identify interior and exterior nodes based on level set function
  /// @tparam T Scalar type for the level set function
  /// @tparam U Floating point type for geometry
  /// @param mesh Background mesh
  /// @param level_set Level set function (negative inside, positive outside)
  /// @return Pair of vectors containing interior and exterior node indices
  template <dolfinx::scalar T, std::floating_point U>
  std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
  classify_interior_exterior_nodes(
    const dolfinx::mesh::Mesh<U>& mesh,
    const dolfinx::fem::Function<T, U>& level_set);

  /// @brief Compute extension weights for exterior nodes
  /// @tparam U Floating point type
  /// @param mesh Background mesh
  /// @param interior_nodes Vector of interior node indices
  /// @param exterior_nodes Vector of exterior node indices
  /// @param extension_method Method for computing weights
  /// @return Matrix of extension weights
  template <std::floating_point U>
  std::vector<std::vector<U>> compute_extension_weights(
    const dolfinx::mesh::Mesh<U>& mesh,
    const std::vector<std::int32_t>& interior_nodes,
    const std::vector<std::int32_t>& exterior_nodes,
    const std::string& extension_method = "harmonic");

  /// @brief Apply extension operator to extend values from interior to exterior
  /// @tparam T Scalar type for function values
  /// @tparam U Floating point type for geometry
  /// @param interior_values Values at interior nodes
  /// @param extension_weights Pre-computed extension weights
  /// @return Extended values at exterior nodes
  template <dolfinx::scalar T, std::floating_point U>
  std::vector<T> apply_extension_operator(
    const std::span<const T>& interior_values,
    const std::vector<std::vector<U>>& extension_weights);
  */

} // namespace cutfemx::extensions

// Include template implementations
#include "discrete_extension_impl.h"
