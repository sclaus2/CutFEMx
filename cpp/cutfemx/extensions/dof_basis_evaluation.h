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
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/la/MatrixCSR.h>
#include "matrix_operations.h"

#include <span>
#include <vector>
#include <memory>
#include <concepts>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <array>
#include <spdlog/spdlog.h>

// Use DOLFINx mdspan namespace
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

namespace cutfemx::extensions
{
  /// DOF coordinate mapping and basis function evaluation functionality
  /// Provides functions to map DOF coordinates to reference space and evaluate basis functions

/// @brief Map DOF coordinates from root cell to reference space and evaluate all shape functions
/// @tparam T Floating point type for geometry
/// @param dof_to_root_mapping Mapping from DOF indices to root cell indices
/// @param function_space Function space containing the basis functions to evaluate
/// @param mesh Mesh object for coordinate transformations
/// @return Mapping from DOF indices to vectors of basis function values evaluated at the DOF location
/// @note This function:
///   1. For each DOF, gets its physical coordinate
///   2. Maps the DOF coordinate to reference space of its root cell  
///   3. Evaluates all basis functions of the root cell at the mapped coordinate
///   4. Returns DOF constraints as pairs of (root_dof_index, basis_value)
template <std::floating_point T>
std::unordered_map<std::int32_t, std::vector<std::pair<std::int32_t, T>>>
evaluate_basis_at_dof_coordinates(
    const std::unordered_map<std::int32_t, std::int32_t>& dof_to_root_mapping,
    const dolfinx::fem::FunctionSpace<T>& function_space,
    const dolfinx::mesh::Mesh<T>& mesh)
{
    std::unordered_map<std::int32_t, std::vector<std::pair<std::int32_t, T>>> constraints;
    
    if (dof_to_root_mapping.empty())
    {
        spdlog::info("Empty dof_to_root_mapping, returning empty basis evaluation");
        return constraints;
    }
    
    spdlog::info("Evaluating basis functions at {} DOF coordinates", dof_to_root_mapping.size());
    
    // Get DOF coordinates from function space
    const std::vector<T> dof_coords = function_space.tabulate_dof_coordinates(false);
    const int gdim = mesh.geometry().dim();
    const int tdim = mesh.topology()->dim();
    
    // Get finite element and coordinate element
    auto element = function_space.element();
    const dolfinx::fem::CoordinateElement<T>& cmap = mesh.geometry().cmap();
    
    // Get space dimension (number of basis functions per cell)
    const std::size_t space_dimension = element->space_dimension();
    const std::size_t reference_value_size = element->reference_value_size();
    const int bs_element = element->block_size();
    const std::size_t scalar_space_dimension = space_dimension / bs_element;
    
    // Get mesh geometry data
    auto x_dofmap = mesh.geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const T> x_g = mesh.geometry().x();
    
    // Get dofmap for accessing cell DOFs
    auto dofmap = function_space.dofmap();
    
    // Prepare working arrays for coordinate transformations
    std::vector<T> coord_dofs_b(num_dofs_g * gdim);
    using mdspan2_t = md::mdspan<T, md::dextents<std::size_t, 2>>;
    mdspan2_t coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);
    
    // Working arrays for reference coordinate computation
    std::vector<T> X_ref_b(1 * tdim);  // Single point in reference space
    mdspan2_t X_ref(X_ref_b.data(), 1, tdim);
    
    std::vector<T> x_phys_b(1 * gdim);  // Single point in physical space
    mdspan2_t x_phys(x_phys_b.data(), 1, gdim);
    
    // Working arrays for Jacobian computations (for affine case)
    std::vector<T> J_b(gdim * tdim);
    mdspan2_t J(J_b.data(), gdim, tdim);
    std::vector<T> K_b(tdim * gdim);
    mdspan2_t K(K_b.data(), tdim, gdim);
    std::vector<T> det_scratch(2 * gdim * tdim);
    
    // Working arrays for basis function evaluation
    std::vector<T> basis_values_b(scalar_space_dimension * reference_value_size);
    using mdspan3_t = md::mdspan<T, md::dextents<std::size_t, 3>>;
    mdspan3_t basis_values(basis_values_b.data(), 1, scalar_space_dimension, reference_value_size);
    
    // Arrays for coordinate element basis (for coordinate transformation)
    // Compute coordinate element basis derivatives at reference origin (0,0,0) once
    std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
    std::vector<T> phi_b(std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    using mdspan4_t = md::mdspan<const T, md::dextents<std::size_t, 4>>;
    mdspan4_t phi(phi_b.data(), phi_shape);
    
    // Evaluate coordinate element basis at reference origin (0,0,0) - used for affine cells
    std::vector<T> ref_origin(tdim, 0.0);
    cmap.tabulate(1, ref_origin, {1, tdim}, phi_b);
    auto dphi = md::submdspan(phi, std::pair(1, tdim + 1), 0, md::full_extent, 0);
    
    // Process each DOF
    for (const auto& [dof_index, root_cell] : dof_to_root_mapping)
    {
        // Get DOF coordinate in physical space
        for (int d = 0; d < gdim; ++d)
        {
            x_phys(0, d) = dof_coords[dof_index * 3 + d];  // DOLFINx stores as 3D
        }
        
        // Get root cell geometry (coordinate DOFs)
        auto x_dofs = md::submdspan(x_dofmap, root_cell, md::full_extent);
        for (std::size_t i = 0; i < num_dofs_g; ++i)
        {
            const int pos = 3 * x_dofs[i];
            for (std::size_t j = 0; j < gdim; ++j)
                coord_dofs(i, j) = x_g[pos + j];
        }
        
        // Map DOF coordinate to reference space of root cell
        if (cmap.is_affine())
        {
            // Reset Jacobian matrices to ensure clean computation
            std::fill(J_b.begin(), J_b.end(), 0);
            std::fill(K_b.begin(), K_b.end(), 0);
            
            // For affine cells, use precomputed coordinate element derivatives
            dolfinx::fem::CoordinateElement<T>::compute_jacobian(dphi, coord_dofs, J);            
            dolfinx::fem::CoordinateElement<T>::compute_jacobian_inverse(J, K);
            
            // Check for NaN or infinite values in K
            bool K_has_nan = false;
            for (std::size_t i = 0; i < tdim; ++i) {
                for (std::size_t j = 0; j < gdim; ++j) {
                    if (!std::isfinite(K(i, j))) {
                        K_has_nan = true;
                        break;
                    }
                }
                if (K_has_nan) break;
            }
            if (K_has_nan) {
                spdlog::warn("NaN or infinite values detected in inverse Jacobian K for DOF {}", dof_index);
            }
            
            // Compute reference coordinate of origin in physical space
            std::array<T, 3> x0 = {0, 0, 0};
            for (std::size_t i = 0; i < gdim; ++i)
                x0[i] = coord_dofs(0, i);  // First node is origin for affine elements
            
            // Pull back to reference coordinates: X = K(x - x0)
            dolfinx::fem::CoordinateElement<T>::pull_back_affine(X_ref, K, x0, x_phys);
            
            // Check for NaN or infinite values
            bool has_nan = false;
            for (std::size_t i = 0; i < tdim; ++i) {
                if (!std::isfinite(X_ref(0, i))) {
                    has_nan = true;
                    break;
                }
            }
            if (has_nan) {
                spdlog::warn("NaN or infinite values detected in X_ref for DOF {}", dof_index);
            }
        }
        else
        {
            // For non-affine cells, use Newton iteration
            cmap.pull_back_nonaffine(X_ref, x_phys, coord_dofs);
        }
        
        // Evaluate basis functions at reference coordinate
        element->tabulate(basis_values_b, X_ref_b, {X_ref.extent(0), X_ref.extent(1)}, 0);
        
        // Get DOFs of the root cell to create constraints
        auto dofmap = function_space.dofmap();
        auto root_dofs = dofmap->cell_dofs(root_cell);
        
        // Create constraint entries: each cut DOF is constrained by basis function values on root DOFs
        std::vector<std::pair<std::int32_t, T>> dof_constraints;
        dof_constraints.reserve(root_dofs.size());
        
        for (std::size_t i = 0; i < root_dofs.size(); ++i)
        {
            // For scalar elements, basis values are stored directly
            T basis_value = basis_values(0, i, 0);  // (point, basis_function, component)
            
            if (std::abs(basis_value) > 1e-12)  // Only store non-zero constraints
            {
                dof_constraints.emplace_back(root_dofs[i], basis_value);
                spdlog::debug("  Constraint: DOF {} -> root DOF {} with weight {:.6f}", 
                             dof_index, root_dofs[i], basis_value);
            }
        }
        
        constraints[dof_index] = std::move(dof_constraints);
    }
    
    spdlog::info("Constraint evaluation complete: {} cut DOFs constrained by {} unique constraint pairs", 
                 constraints.size(),
                 [&]() {
                     std::size_t total_constraints = 0;
                     for (const auto& [dof, constraint_list] : constraints) {
                         total_constraints += constraint_list.size();
                     }
                     return total_constraints;
                 }());
    
    return constraints;
}


/// @brief Apply constraints to a matrix: A ↦ C^T A C
/// @tparam T Scalar type
/// @tparam MatrixType Matrix type (e.g., dolfinx::la::MatrixCSR or PETSc Mat)
/// @param A Input/output matrix to be transformed
/// @param constraints Mapping from cut DOFs to interior DOF constraints
/// @param function_space Function space for DOF indexing
/// @note This function implements the constraint transformation A ↦ C^T A C
///       where C is the constraint matrix expressing cut DOFs as linear combinations of interior DOFs.
///       The transformation eliminates cut DOFs from the system while preserving the underlying physics.
template <std::floating_point T, typename MatrixType>
void apply_constraints(
    MatrixType& A,
    const std::unordered_map<std::int32_t, std::vector<std::pair<std::int32_t, T>>>& constraints,
    const dolfinx::fem::FunctionSpace<T>& function_space)
{
    if (constraints.empty())
    {
        spdlog::info("No constraints to apply");
        return;
    }
    
    spdlog::info("Applying constraints to matrix: {} cut DOFs will be eliminated", constraints.size());
    
    // Get DOF map for accessing matrix structure
    auto dofmap = function_space.dofmap();
    const std::int32_t ndofs = dofmap->index_map->size_local() + dofmap->index_map->num_ghosts();
    
    // For each constrained DOF, we need to:
    // 1. Distribute its row to the constraint DOFs (C^T A operation)
    // 2. Distribute its column to the constraint DOFs (A C operation)
    // 3. Zero out the constrained DOF row and column
    // 4. Set diagonal entry to 1 for constrained DOF
    
    // First pass: Apply C^T A (modify rows)
    for (const auto& [cut_dof, constraint_list] : constraints)
    {
        if (constraint_list.empty()) continue;
        
        // Get the row for the cut DOF
        // auto row_data = get_row(A, cut_dof);  // Available if needed for debugging
        
        // For each constraint DOF, add scaled contribution
        for (const auto& [root_dof, weight] : constraint_list)
        {
            // Add weight * (cut_dof row) to root_dof row
            add_to_row(A, root_dof, cut_dof, weight);  // Function from matrix_operations.h
        }
    }
    
    // Second pass: Apply A C (modify columns)  
    for (const auto& [cut_dof, constraint_list] : constraints)
    {
        if (constraint_list.empty()) continue;
        
        // For each constraint DOF, distribute the cut DOF column
        for (const auto& [root_dof, weight] : constraint_list)
        {
            // Add weight * (cut_dof column) to root_dof column
            add_to_column(A, root_dof, cut_dof, weight);  // Function from matrix_operations.h
        }
    }
    
    // Third pass: Zero constrained rows and columns, set diagonal to 1
    for (const auto& [cut_dof, constraint_list] : constraints)
    {
        // Zero the row and column for the constrained DOF
        zero_row(A, cut_dof);
        zero_column(A, cut_dof);
        
        // Set diagonal entry to 1
        set_value(A, cut_dof, cut_dof, static_cast<T>(1.0));
    }
    
    spdlog::info("Matrix constraint application complete");
}

/// @brief Extend a solution vector from interior DOFs to full domain: u_full = C u_interior
/// @tparam T Scalar type
/// @param u_interior Input solution vector defined on interior DOFs only
/// @param constraints Mapping from cut DOFs to interior DOF constraints
/// @param function_space Function space for DOF indexing
/// @return Extended solution vector including both interior and cut DOFs
/// @note This function reconstructs the full field by applying the constraint relationships:
///       u_cut = Σ weight_i * u_interior[root_dof_i] for each cut DOF
template <std::floating_point T>
std::vector<T> extend_function(
    const std::span<const T>& u_interior,
    const std::unordered_map<std::int32_t, std::vector<std::pair<std::int32_t, T>>>& constraints,
    const dolfinx::fem::FunctionSpace<T>& function_space)
{
    if (constraints.empty())
    {
        spdlog::info("No constraints found, returning copy of interior solution");
        return std::vector<T>(u_interior.begin(), u_interior.end());
    }
    
    spdlog::info("Extending solution from interior DOFs to full domain: {} cut DOFs to be computed", 
                 constraints.size());
    
    // Get total number of DOFs (including ghosts)
    auto dofmap = function_space.dofmap();
    const std::int32_t ndofs_total = dofmap->index_map->size_local() + dofmap->index_map->num_ghosts();
    
    // Initialize extended solution vector
    std::vector<T> u_full(ndofs_total, static_cast<T>(0.0));
    
    // Copy interior solution (assuming interior DOFs come first)
    // Note: This assumes u_interior contains values for non-constrained DOFs
    std::size_t interior_idx = 0;
    for (std::int32_t dof = 0; dof < ndofs_total; ++dof)
    {
        if (constraints.find(dof) == constraints.end())
        {
            // This is an interior (non-constrained) DOF
            if (interior_idx < u_interior.size())
            {
                u_full[dof] = u_interior[interior_idx];
                ++interior_idx;
            }
        }
    }
    
    // Apply constraints to compute cut DOF values
    std::size_t extended_dofs = 0;
    for (const auto& [cut_dof, constraint_list] : constraints)
    {
        if (constraint_list.empty()) continue;
        
        T cut_value = static_cast<T>(0.0);
        bool valid_constraint = true;
        
        // Compute: u_cut = Σ weight_i * u_interior[root_dof_i]
        for (const auto& [root_dof, weight] : constraint_list)
        {
            if (root_dof >= 0 && root_dof < ndofs_total)
            {
                cut_value += weight * u_full[root_dof];
            }
            else
            {
                spdlog::warn("Invalid root DOF {} in constraint for cut DOF {}", root_dof, cut_dof);
                valid_constraint = false;
                break;
            }
        }
        
        if (valid_constraint && cut_dof >= 0 && cut_dof < ndofs_total)
        {
            u_full[cut_dof] = cut_value;
            ++extended_dofs;
            
            spdlog::debug("Extended DOF {}: value = {:.6f} from {} constraints", 
                         cut_dof, cut_value, constraint_list.size());
        }
        else if (!valid_constraint)
        {
            spdlog::warn("Skipping invalid constraint for cut DOF {}", cut_dof);
        }
    }
    
    spdlog::info("Function extension complete: {} cut DOFs computed from constraints", extended_dofs);
    
    return u_full;
}

/// @brief Specialized version of apply_constraints for DOLFINx MatrixCSR
/// @tparam T Scalar type
/// @param A DOLFINx CSR matrix to be transformed
/// @param constraints Mapping from cut DOFs to interior DOF constraints
/// @param function_space Function space for DOF indexing
/// @note This is a simplified implementation that zeros constrained rows/columns
///       and sets diagonal entries. Full C^T A C transformation requires more
///       sophisticated matrix operations.
template <std::floating_point T>
void apply_constraints(
    dolfinx::la::MatrixCSR<T>& A,
    const std::unordered_map<std::int32_t, std::vector<std::pair<std::int32_t, T>>>& constraints,
    const dolfinx::fem::FunctionSpace<T>& function_space)
{
    if (constraints.empty())
    {
        spdlog::info("No constraints to apply to MatrixCSR");
        return;
    }
    
    spdlog::info("Applying constraints to DOLFINx MatrixCSR: {} cut DOFs will be eliminated", 
                 constraints.size());
    
    // Get matrix data - we need mutable access for modification
    auto& data = const_cast<typename dolfinx::la::MatrixCSR<T>::container_type&>(A.values());
    const auto& col_indices = A.cols();
    const auto& row_ptrs = A.row_ptr();
    
    auto dofmap = function_space.dofmap();
    const std::int32_t ndofs = dofmap->index_map->size_local();
    
    // Simple constraint application: zero constrained rows and set diagonal
    // Note: Full C^T A C transformation would require more sophisticated matrix operations
    auto mat_set = A.template mat_set_values<1, 1>();
    
    for (const auto& [cut_dof, constraint_list] : constraints)
    {
        if (cut_dof >= ndofs) continue;
        
        // Zero the entire row
        const std::int64_t row_start = row_ptrs[cut_dof];
        const std::int64_t row_end = row_ptrs[cut_dof + 1];
        
        for (std::int64_t idx = row_start; idx < row_end; ++idx)
        {
            const std::int32_t col = col_indices[idx];
            std::array<std::int32_t, 1> rows = {cut_dof};
            std::array<std::int32_t, 1> cols = {col};
            std::array<T, 1> vals = {static_cast<T>(0.0)};
            mat_set(rows, cols, vals);
        }
        
        // Set diagonal to 1
        std::array<std::int32_t, 1> rows = {cut_dof};
        std::array<std::int32_t, 1> cols = {cut_dof};
        std::array<T, 1> vals = {static_cast<T>(1.0)};
        mat_set(rows, cols, vals);
    }
    
    spdlog::info("DOLFINx MatrixCSR constraint application complete (simplified version)");
}


} // namespace cutfemx::extensions
