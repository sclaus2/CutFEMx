// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <dolfinx/la/MatrixCSR.h>
#include <span>
#include <vector>
#include <concepts>
#include <algorithm>
#include <numeric>
#include <spdlog/spdlog.h>

#ifdef HAS_PETSC
#include <petscmat.h>
#include <dolfinx/la/petsc.h>
#endif

namespace cutfemx::extensions
{

/// @brief Matrix constraint operations for row and column manipulations
/// This header provides implementations for add_to_row and add_to_column operations
/// for both DOLFINx MatrixCSR and PETSc matrices.

//=============================================================================
// DOLFINx MatrixCSR operations
//=============================================================================

/// @brief Add a scaled row to another row in a DOLFINx MatrixCSR
/// @tparam T Scalar type
/// @param A The matrix to modify
/// @param target_row The row to add to
/// @param source_row The row to add from
/// @param scale Scaling factor for the source row
template <std::floating_point T>
void add_to_row(dolfinx::la::MatrixCSR<T>& A, std::int32_t target_row, 
                std::int32_t source_row, T scale)
{
    // Get matrix data
    auto& data = const_cast<typename dolfinx::la::MatrixCSR<T>::container_type&>(A.values());
    const auto& col_indices = A.cols();
    const auto& row_ptrs = A.row_ptr();
    
    const std::int32_t num_rows = A.num_all_rows();
    
    if (target_row >= num_rows || source_row >= num_rows || target_row < 0 || source_row < 0)
    {
        spdlog::warn("Invalid row indices: target={}, source={}, num_rows={}", 
                     target_row, source_row, num_rows);
        return;
    }
    
    if (std::abs(scale) < 1e-15)
    {
        return; // Nothing to add
    }
    
    // Get source row data
    const std::int64_t source_start = row_ptrs[source_row];
    const std::int64_t source_end = row_ptrs[source_row + 1];
    
    // Get target row data
    const std::int64_t target_start = row_ptrs[target_row];
    const std::int64_t target_end = row_ptrs[target_row + 1];
    
    // For each entry in source row, find corresponding entry in target row and add
    for (std::int64_t src_idx = source_start; src_idx < source_end; ++src_idx)
    {
        const std::int32_t col = col_indices[src_idx];
        const T source_value = data[src_idx];
        
        if (std::abs(source_value) < 1e-15) continue; // Skip zero entries
        
        // Find the column in target row
        auto target_col_it = std::lower_bound(
            col_indices.begin() + target_start,
            col_indices.begin() + target_end,
            col);
        
        if (target_col_it != col_indices.begin() + target_end && *target_col_it == col)
        {
            // Found the column in target row
            const std::int64_t target_idx = std::distance(col_indices.begin(), target_col_it);
            data[target_idx] += scale * source_value;
        }
        else
        {
            // Column not found in target row - this means the sparsity pattern doesn't allow this operation
            spdlog::debug("Column {} not found in target row {} sparsity pattern", col, target_row);
        }
    }
}

/// @brief Add a scaled column to another column in a DOLFINx MatrixCSR
/// @tparam T Scalar type
/// @param A The matrix to modify
/// @param target_col The column to add to
/// @param source_col The column to add from
/// @param scale Scaling factor for the source column
/// @note This operation is expensive for CSR format as it requires searching through all rows
template <std::floating_point T>
void add_to_column(dolfinx::la::MatrixCSR<T>& A, std::int32_t target_col, 
                   std::int32_t source_col, T scale)
{
    // Get matrix data
    auto& data = const_cast<typename dolfinx::la::MatrixCSR<T>::container_type&>(A.values());
    const auto& col_indices = A.cols();
    const auto& row_ptrs = A.row_ptr();
    
    const std::int32_t num_rows = A.num_all_rows();
    
    if (std::abs(scale) < 1e-15)
    {
        return; // Nothing to add
    }
    
    // Iterate through all rows to find entries in source and target columns
    for (std::int32_t row = 0; row < num_rows; ++row)
    {
        const std::int64_t row_start = row_ptrs[row];
        const std::int64_t row_end = row_ptrs[row + 1];
        
        // Find source column entry in this row
        auto source_it = std::lower_bound(
            col_indices.begin() + row_start,
            col_indices.begin() + row_end,
            source_col);
        
        if (source_it != col_indices.begin() + row_end && *source_it == source_col)
        {
            // Found source column entry
            const std::int64_t source_idx = std::distance(col_indices.begin(), source_it);
            const T source_value = data[source_idx];
            
            if (std::abs(source_value) < 1e-15) continue; // Skip zero entries
            
            // Find target column entry in the same row
            auto target_it = std::lower_bound(
                col_indices.begin() + row_start,
                col_indices.begin() + row_end,
                target_col);
            
            if (target_it != col_indices.begin() + row_end && *target_it == target_col)
            {
                // Found target column entry
                const std::int64_t target_idx = std::distance(col_indices.begin(), target_it);
                data[target_idx] += scale * source_value;
            }
            else
            {
                // Target column not found in this row - sparsity pattern doesn't allow this operation
                spdlog::debug("Target column {} not found in row {} sparsity pattern", target_col, row);
            }
        }
    }
}

/// @brief Get a row from DOLFINx MatrixCSR as a sparse representation
/// @tparam T Scalar type
/// @param A The matrix
/// @param row The row index
/// @return Vector of pairs (column_index, value) for non-zero entries
template <std::floating_point T>
std::vector<std::pair<std::int32_t, T>> get_row(const dolfinx::la::MatrixCSR<T>& A, std::int32_t row)
{
    const auto& data = A.values();
    const auto& col_indices = A.cols();
    const auto& row_ptrs = A.row_ptr();
    
    const std::int32_t num_rows = A.num_all_rows();
    
    if (row >= num_rows || row < 0)
    {
        spdlog::warn("Invalid row index: row={}, num_rows={}", row, num_rows);
        return {};
    }
    
    std::vector<std::pair<std::int32_t, T>> row_data;
    
    const std::int64_t row_start = row_ptrs[row];
    const std::int64_t row_end = row_ptrs[row + 1];
    
    row_data.reserve(row_end - row_start);
    
    for (std::int64_t idx = row_start; idx < row_end; ++idx)
    {
        row_data.emplace_back(col_indices[idx], data[idx]);
    }
    
    return row_data;
}

/// @brief Zero a row in DOLFINx MatrixCSR
/// @tparam T Scalar type
/// @param A The matrix to modify
/// @param row The row to zero
template <std::floating_point T>
void zero_row(dolfinx::la::MatrixCSR<T>& A, std::int32_t row)
{
    auto& data = const_cast<typename dolfinx::la::MatrixCSR<T>::container_type&>(A.values());
    const auto& row_ptrs = A.row_ptr();
    
    const std::int32_t num_rows = A.num_all_rows();
    
    if (row >= num_rows || row < 0)
    {
        spdlog::warn("Invalid row index: row={}, num_rows={}", row, num_rows);
        return;
    }
    
    const std::int64_t row_start = row_ptrs[row];
    const std::int64_t row_end = row_ptrs[row + 1];
    
    std::fill(data.begin() + row_start, data.begin() + row_end, static_cast<T>(0.0));
}

/// @brief Zero a column in DOLFINx MatrixCSR
/// @tparam T Scalar type
/// @param A The matrix to modify
/// @param col The column to zero
template <std::floating_point T>
void zero_column(dolfinx::la::MatrixCSR<T>& A, std::int32_t col)
{
    auto& data = const_cast<typename dolfinx::la::MatrixCSR<T>::container_type&>(A.values());
    const auto& col_indices = A.cols();
    const auto& row_ptrs = A.row_ptr();
    
    const std::int32_t num_rows = A.num_all_rows();
    
    // Iterate through all rows to find entries in the specified column
    for (std::int32_t row = 0; row < num_rows; ++row)
    {
        const std::int64_t row_start = row_ptrs[row];
        const std::int64_t row_end = row_ptrs[row + 1];
        
        // Find column entry in this row
        auto col_it = std::lower_bound(
            col_indices.begin() + row_start,
            col_indices.begin() + row_end,
            col);
        
        if (col_it != col_indices.begin() + row_end && *col_it == col)
        {
            // Found column entry
            const std::int64_t idx = std::distance(col_indices.begin(), col_it);
            data[idx] = static_cast<T>(0.0);
        }
    }
}

/// @brief Set a single value in DOLFINx MatrixCSR
/// @tparam T Scalar type
/// @param A The matrix to modify
/// @param row The row index
/// @param col The column index
/// @param value The value to set
template <std::floating_point T>
void set_value(dolfinx::la::MatrixCSR<T>& A, std::int32_t row, std::int32_t col, T value)
{
    auto mat_set = A.template mat_set_values<1, 1>();
    std::array<std::int32_t, 1> rows = {row};
    std::array<std::int32_t, 1> cols = {col};
    std::array<T, 1> vals = {value};
    mat_set(rows, cols, vals);
}

//=============================================================================
// PETSc Matrix operations
//=============================================================================

#ifdef HAS_PETSC

/// @brief Add a scaled row to another row in a PETSc matrix
/// @param A The PETSc matrix
/// @param target_row The row to add to
/// @param source_row The row to add from
/// @param scale Scaling factor for the source row
inline void add_to_row(Mat A, PetscInt target_row, PetscInt source_row, PetscScalar scale)
{
    if (std::abs(scale) < 1e-15)
    {
        return; // Nothing to add
    }
    
    PetscErrorCode ierr;
    
    // Get the source row
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;
    
    ierr = MatGetRow(A, source_row, &ncols, &cols, &vals);
    if (ierr != 0)
    {
        spdlog::error("Failed to get source row {} from PETSc matrix", source_row);
        return;
    }
    
    // Scale the values and add to target row
    std::vector<PetscScalar> scaled_vals(ncols);
    for (PetscInt i = 0; i < ncols; ++i)
    {
        scaled_vals[i] = scale * vals[i];
    }
    
    // Add scaled values to target row
    ierr = MatSetValues(A, 1, &target_row, ncols, cols, scaled_vals.data(), ADD_VALUES);
    if (ierr != 0)
    {
        spdlog::error("Failed to add values to target row {} in PETSc matrix", target_row);
    }
    
    // Restore the source row
    ierr = MatRestoreRow(A, source_row, &ncols, &cols, &vals);
    if (ierr != 0)
    {
        spdlog::error("Failed to restore source row {} in PETSc matrix", source_row);
    }
}

/// @brief Add a scaled column to another column in a PETSc matrix
/// @param A The PETSc matrix
/// @param target_col The column to add to
/// @param source_col The column to add from
/// @param scale Scaling factor for the source column
/// @note This operation requires matrix transpose which can be expensive
inline void add_to_column(Mat A, PetscInt target_col, PetscInt source_col, PetscScalar scale)
{
    if (std::abs(scale) < 1e-15)
    {
        return; // Nothing to add
    }
    
    PetscErrorCode ierr;
    
    // For column operations, we need to work with the transpose or use row-wise access
    // This is a simplified implementation - for better performance, consider using MatTranspose
    
    PetscInt m, n;
    ierr = MatGetSize(A, &m, &n);
    if (ierr != 0)
    {
        spdlog::error("Failed to get matrix size");
        return;
    }
    
    // Iterate through all rows and modify entries in the specified columns
    for (PetscInt row = 0; row < m; ++row)
    {
        // Get source column value in this row
        PetscScalar source_val;
        ierr = MatGetValues(A, 1, &row, 1, &source_col, &source_val);
        if (ierr == 0 && std::abs(source_val) > 1e-15)
        {
            // Add scaled value to target column
            PetscScalar scaled_val = scale * source_val;
            ierr = MatSetValues(A, 1, &row, 1, &target_col, &scaled_val, ADD_VALUES);
            if (ierr != 0)
            {
                spdlog::debug("Failed to set value at ({}, {}) in PETSc matrix", row, target_col);
            }
        }
    }
}

/// @brief Get a row from PETSc matrix
/// @param A The PETSc matrix
/// @param row The row index
/// @return Vector of pairs (column_index, value) for non-zero entries
inline std::vector<std::pair<PetscInt, PetscScalar>> get_row(Mat A, PetscInt row)
{
    std::vector<std::pair<PetscInt, PetscScalar>> row_data;
    
    PetscErrorCode ierr;
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;
    
    ierr = MatGetRow(A, row, &ncols, &cols, &vals);
    if (ierr != 0)
    {
        spdlog::error("Failed to get row {} from PETSc matrix", row);
        return row_data;
    }
    
    row_data.reserve(ncols);
    for (PetscInt i = 0; i < ncols; ++i)
    {
        row_data.emplace_back(cols[i], vals[i]);
    }
    
    ierr = MatRestoreRow(A, row, &ncols, &cols, &vals);
    if (ierr != 0)
    {
        spdlog::error("Failed to restore row {} in PETSc matrix", row);
    }
    
    return row_data;
}

/// @brief Zero a row in PETSc matrix
/// @param A The PETSc matrix
/// @param row The row to zero
inline void zero_row(Mat A, PetscInt row)
{
    PetscErrorCode ierr = MatZeroRows(A, 1, &row, 0.0, nullptr, nullptr);
    if (ierr != 0)
    {
        spdlog::error("Failed to zero row {} in PETSc matrix", row);
    }
}

/// @brief Zero a column in PETSc matrix
/// @param A The PETSc matrix
/// @param col The column to zero
inline void zero_column(Mat A, PetscInt col)
{
    // PETSc doesn't have a direct MatZeroColumns function
    // We'll manually zero the column by getting matrix info and setting values to zero
    PetscInt m, n;
    PetscErrorCode ierr = MatGetSize(A, &m, &n);
    if (ierr != 0)
    {
        spdlog::error("Failed to get matrix size for zeroing column {}", col);
        return;
    }
    
    // Set all entries in the specified column to zero
    PetscScalar zero = 0.0;
    for (PetscInt row = 0; row < m; ++row)
    {
        ierr = MatSetValue(A, row, col, zero, INSERT_VALUES);
        if (ierr != 0)
        {
            spdlog::error("Failed to set zero value at ({}, {}) for column zeroing", row, col);
            return;
        }
    }
    
    // Assemble the matrix after modifications
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    if (ierr == 0)
    {
        ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    }
    
    if (ierr != 0)
    {
        spdlog::error("Failed to assemble matrix after zeroing column {}", col);
    }
}

/// @brief Set a single value in PETSc matrix
/// @param A The PETSc matrix
/// @param row The row index
/// @param col The column index
/// @param value The value to set
inline void set_value(Mat A, PetscInt row, PetscInt col, PetscScalar value)
{
    PetscErrorCode ierr = MatSetValues(A, 1, &row, 1, &col, &value, INSERT_VALUES);
    if (ierr != 0)
    {
        spdlog::error("Failed to set value at ({}, {}) in PETSc matrix", row, col);
    }
}

#endif // HAS_PETSC

//=============================================================================
// Generic interface using function overloading
//=============================================================================

/// @brief Generic interface for matrix operations using function overloading
/// These functions automatically dispatch to the appropriate implementation
/// based on the matrix type.

/// Generic add_to_row for DOLFINx MatrixCSR
template <std::floating_point T>
void matrix_add_to_row(dolfinx::la::MatrixCSR<T>& A, std::int32_t target_row, 
                      std::int32_t source_row, T scale)
{
    add_to_row(A, target_row, source_row, scale);
}

/// Generic add_to_column for DOLFINx MatrixCSR
template <std::floating_point T>
void matrix_add_to_column(dolfinx::la::MatrixCSR<T>& A, std::int32_t target_col, 
                         std::int32_t source_col, T scale)
{
    add_to_column(A, target_col, source_col, scale);
}

#ifdef HAS_PETSC
/// Generic add_to_row for PETSc Mat
inline void matrix_add_to_row(Mat A, PetscInt target_row, PetscInt source_row, PetscScalar scale)
{
    add_to_row(A, target_row, source_row, scale);
}

/// Generic add_to_column for PETSc Mat
inline void matrix_add_to_column(Mat A, PetscInt target_col, PetscInt source_col, PetscScalar scale)
{
    add_to_column(A, target_col, source_col, scale);
}
#endif

} // namespace cutfemx::extensions
