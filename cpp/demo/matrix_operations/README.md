# Matrix Operations Demo

This demo showcases the `add_to_row` and `add_to_column` functions implemented for CutFEMx constraint applications.

## Overview

The demo demonstrates:

1. **DOLFINx MatrixCSR Operations**:
   - Creating a sparse matrix with a simple sparsity pattern
   - Using `add_to_row` to add a scaled row to another row
   - Using `add_to_column` to add a scaled column to another column
   - Applying constraints by zeroing rows/columns and setting diagonal entries

2. **PETSc Matrix Operations** (if available):
   - Similar operations using PETSc Mat interface
   - Comparison with DOLFINx implementation

## Building and Running

### Prerequisites

- DOLFINx with C++20 support
- spdlog library
- MPI
- PETSc (optional, for PETSc matrix demonstrations)

### Build

From this directory:

```bash
mkdir build
cd build
cmake ..
make
```

### Run

```bash
# Run with single process
./matrix_operations_demo

# Run with MPI (for testing parallel matrix operations)
mpirun -n 2 ./matrix_operations_demo
```

## What the Demo Shows

### Sample Output

```
Matrix Operations Demo - CutFEMx
=== DOLFINx MatrixCSR Operations Demo ===
Original matrix values:
Row 0: [0:1] [1:2] [2:3] [3:4] 
Row 1: [0:2] [1:3] [2:4] [3:5] 
Row 2: [0:3] [1:4] [2:5] [3:6] 
Row 3: [0:4] [1:5] [2:6] [3:7] 

Applying: add_to_row(A, 1, 0, 0.5)
Matrix after add_to_row:
Row 0: [0:1] [1:2] [2:3] [3:4] 
Row 1: [0:2.5] [1:4] [2:5.5] [3:7] 
Row 2: [0:3] [1:4] [2:5] [3:6] 
Row 3: [0:4] [1:5] [2:6] [3:7] 

Applying: add_to_column(A, 2, 0, 0.25)
Matrix after add_to_column:
Row 0: [0:1] [1:2] [2:3.25] [3:4] 
Row 1: [0:2.5] [1:4] [2:6.125] [3:7] 
Row 2: [0:3] [1:4] [2:5.75] [3:6] 
Row 3: [0:4] [1:5] [2:7] [3:7] 

Applying constraint: zero_row(A, 3) and set_value(A, 3, 3, 1.0)
Matrix after constraint application:
Row 0: [0:1] [1:2] [2:3.25] [3:4] 
Row 1: [0:2.5] [1:4] [2:6.125] [3:7] 
Row 2: [0:3] [1:4] [2:5.75] [3:6] 
Row 3: [0:0] [1:0] [2:0] [3:1] 
```

### Key Operations Demonstrated

1. **Matrix Creation**: Setting up a 4x4 dense matrix using DOLFINx SparsityPattern and MatrixCSR
2. **Row Addition**: `add_to_row(A, 1, 0, 0.5)` adds 0.5 times row 0 to row 1
3. **Column Addition**: `add_to_column(A, 2, 0, 0.25)` adds 0.25 times column 0 to column 2
4. **Constraint Application**: Zeroing row 3 and setting the diagonal to 1 (typical constraint pattern)

## Integration with CutFEMx

These operations are designed for use in the CutFEMx constraint application workflow:

```cpp
// In constraint application (C^T A C transformation):
for (const auto& [cut_dof, constraint_list] : constraints) {
    for (const auto& [root_dof, weight] : constraint_list) {
        add_to_row(A, root_dof, cut_dof, weight);     // Apply C^T A
        add_to_column(A, root_dof, cut_dof, weight);  // Apply A C
    }
}

// Zero constrained DOFs and set diagonal
for (const auto& [cut_dof, constraint_list] : constraints) {
    zero_row(A, cut_dof);
    zero_column(A, cut_dof);  
    set_value(A, cut_dof, cut_dof, 1.0);
}
```

## Performance Notes

- **Row operations** are efficient for CSR format (O(nnz_source_row * log(nnz_target_row)))
- **Column operations** are expensive for CSR format (O(total_nnz)) as they require scanning all rows
- **PETSc operations** handle distributed matrices automatically
- The demo uses small matrices for clarity; real applications will have much larger sparse matrices

## Files

- `matrix_operations_demo.cpp` - Main demonstration program
- `CMakeLists.txt` - Build configuration
- `README.md` - This documentation

## See Also

- `../../cutfemx/extensions/matrix_operations.h` - Implementation
- `../../cutfemx/extensions/dof_basis_evaluation.h` - Constraint application usage
- `../../../docs/matrix_operations.md` - Detailed technical documentation
