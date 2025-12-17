// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

/// @brief Demonstration of matrix_operations.h functionality
/// This file shows how to use the add_to_row and add_to_column functions
/// for both DOLFINx MatrixCSR and PETSc matrices in constraint applications.

#include <cutfemx/extensions/matrix_operations.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/common/IndexMap.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <mpi.h>

#ifdef HAS_PETSC
#include <petscmat.h>
#include <dolfinx/la/petsc.h>
#endif

using namespace cutfemx::extensions;

void demo_matrixcsr_operations()
{
    spdlog::info("=== DOLFINx MatrixCSR Operations Demo ===");
    
    // Create a simple 4x4 matrix for demonstration
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Create index map for a 4x4 matrix (4 owned, 0 ghosts)
    std::vector<std::int64_t> ghosts;
    auto index_map = std::make_shared<dolfinx::common::IndexMap>(comm, 4, ghosts, 1);
    
    // Create sparsity pattern - simple diagonal + off-diagonal pattern
    dolfinx::la::SparsityPattern sp(comm, {index_map, index_map}, {1, 1});
    for (int i = 0; i < 4; ++i)
    {
        std::vector<std::int64_t> dofs_i = {i};
        for (int j = 0; j < 4; ++j)
        {
            std::vector<std::int64_t> dofs_j = {j};
            sp.insert(dofs_i, dofs_j);
        }
    }
    sp.finalize();
    
    // Create matrix
    dolfinx::la::MatrixCSR<double> A(sp);
    
    // Initialize with some test values
    auto mat_set = A.template mat_set_values<1, 1>();
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::array<std::int32_t, 1> rows = {i};
            std::array<std::int32_t, 1> cols = {j};
            std::array<double, 1> vals = {static_cast<double>(i + j + 1)};
            mat_set(rows, cols, vals);
        }
    }
    
    spdlog::info("Original matrix values:");
    const auto& data = A.values();
    const auto& col_indices = A.cols();
    const auto& row_ptrs = A.row_ptr();
    
    for (int i = 0; i < 4; ++i)
    {
        std::string row_str = "Row " + std::to_string(i) + ": ";
        for (std::int64_t idx = row_ptrs[i]; idx < row_ptrs[i + 1]; ++idx)
        {
            row_str += "[" + std::to_string(col_indices[idx]) + ":" + 
                       std::to_string(data[idx]) + "] ";
        }
        spdlog::info(row_str);
    }
    
    // Demonstrate add_to_row: Add 0.5 * row 0 to row 1
    spdlog::info("\nApplying: add_to_row(A, 1, 0, 0.5)");
    add_to_row(A, 1, 0, 0.5);
    
    spdlog::info("Matrix after add_to_row:");
    for (int i = 0; i < 4; ++i)
    {
        std::string row_str = "Row " + std::to_string(i) + ": ";
        for (std::int64_t idx = row_ptrs[i]; idx < row_ptrs[i + 1]; ++idx)
        {
            row_str += "[" + std::to_string(col_indices[idx]) + ":" + 
                       std::to_string(data[idx]) + "] ";
        }
        spdlog::info(row_str);
    }
    
    // Demonstrate add_to_column: Add 0.25 * column 0 to column 2
    spdlog::info("\nApplying: add_to_column(A, 2, 0, 0.25)");
    add_to_column(A, 2, 0, 0.25);
    
    spdlog::info("Matrix after add_to_column:");
    for (int i = 0; i < 4; ++i)
    {
        std::string row_str = "Row " + std::to_string(i) + ": ";
        for (std::int64_t idx = row_ptrs[i]; idx < row_ptrs[i + 1]; ++idx)
        {
            row_str += "[" + std::to_string(col_indices[idx]) + ":" + 
                       std::to_string(data[idx]) + "] ";
        }
        spdlog::info(row_str);
    }
    
    // Demonstrate constraint application: zero row 3 and set diagonal
    spdlog::info("\nApplying constraint: zero_row(A, 3) and set_value(A, 3, 3, 1.0)");
    zero_row(A, 3);
    set_value(A, 3, 3, 1.0);
    
    spdlog::info("Matrix after constraint application:");
    for (int i = 0; i < 4; ++i)
    {
        std::string row_str = "Row " + std::to_string(i) + ": ";
        for (std::int64_t idx = row_ptrs[i]; idx < row_ptrs[i + 1]; ++idx)
        {
            row_str += "[" + std::to_string(col_indices[idx]) + ":" + 
                       std::to_string(data[idx]) + "] ";
        }
        spdlog::info(row_str);
    }
}

#ifdef HAS_PETSC
void demo_petsc_operations()
{
    spdlog::info("\n=== PETSc Matrix Operations Demo ===");
    
    // Create a simple 4x4 PETSc matrix
    Mat A;
    PetscErrorCode ierr;
    
    ierr = MatCreate(PETSC_COMM_WORLD, &A);
    ierr = MatSetSizes(A, 4, 4, 4, 4);
    ierr = MatSetType(A, MATMPIAIJ);
    ierr = MatMPIAIJSetPreallocation(A, 4, nullptr, 4, nullptr);
    
    // Initialize with test values
    for (PetscInt i = 0; i < 4; ++i)
    {
        for (PetscInt j = 0; j < 4; ++j)
        {
            PetscScalar val = static_cast<PetscScalar>(i + j + 1);
            ierr = MatSetValues(A, 1, &i, 1, &j, &val, INSERT_VALUES);
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    spdlog::info("Original PETSc matrix:");
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    
    // Demonstrate add_to_row: Add 0.5 * row 0 to row 1
    spdlog::info("Applying: add_to_row(A, 1, 0, 0.5)");
    add_to_row(A, 1, 0, 0.5);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    spdlog::info("Matrix after add_to_row:");
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    
    // Demonstrate add_to_column: Add 0.25 * column 0 to column 2
    spdlog::info("Applying: add_to_column(A, 2, 0, 0.25)");
    add_to_column(A, 2, 0, 0.25);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    spdlog::info("Matrix after add_to_column:");
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    
    // Demonstrate constraint application
    spdlog::info("Applying constraint: zero_row(A, 3) and set_value(A, 3, 3, 1.0)");
    zero_row(A, 3);
    set_value(A, 3, 3, 1.0);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    spdlog::info("Matrix after constraint application:");
    ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    
    // Clean up
    ierr = MatDestroy(&A);
}
#endif

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
#ifdef HAS_PETSC
    // Initialize PETSc
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif
    
    spdlog::set_level(spdlog::level::info);
    spdlog::info("Matrix Operations Demo - CutFEMx");
    
    try
    {
        // Demonstrate DOLFINx MatrixCSR operations
        demo_matrixcsr_operations();
        
#ifdef HAS_PETSC
        // Demonstrate PETSc operations
        demo_petsc_operations();
#endif
    }
    catch (const std::exception& e)
    {
        spdlog::error("Error in demo: {}", e.what());
    }
    
#ifdef HAS_PETSC
    PetscFinalize();
#endif
    MPI_Finalize();
    
    return 0;
}
