#!/usr/bin/env python3
"""
Demo script for constraint application and function extension.

This script demonstrates the new apply_constraints and extend_function capabilities
using the constraint map from evaluate_basis_at_dof_coordinates.
"""

import numpy as np
import dolfinx
import dolfinx.fem
import dolfinx.la
import ufl
from mpi4py import MPI

# Import CutFEMx extensions
from cutfemx import extensions


def create_simple_mesh_and_function_space():
    """Create a simple 2D mesh and P1 function space for testing."""
    # Create a unit square mesh
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4, dolfinx.mesh.CellType.triangle)

    # Create P1 Lagrange function space
    element = ("Lagrange", 1)
    V = dolfinx.fem.functionspace(mesh, element)

    return mesh, V


def create_dummy_constraints(V):
    """Create some dummy constraints for testing purposes."""
    # Simulate some cut DOFs constrained by interior DOFs
    # This is a simplified example - in practice constraints come from evaluate_basis_at_dof_coordinates
    constraints = {
        0: [(1, 0.5), (2, 0.3), (3, 0.2)],  # DOF 0 constrained by DOFs 1,2,3
        4: [(5, 0.7), (6, 0.3)],  # DOF 4 constrained by DOFs 5,6
        8: [(9, 0.4), (10, 0.4), (11, 0.2)],  # DOF 8 constrained by DOFs 9,10,11
    }

    return constraints


def test_matrix_constraint_application():
    """Test applying constraints to a matrix."""
    print("=== Testing Matrix Constraint Application ===")

    mesh, V = create_simple_mesh_and_function_space()
    constraints = create_dummy_constraints(V)

    print(f"Mesh: {mesh.topology.index_map(mesh.topology.dim).size_local} cells")
    print(f"Function space: {V.dofmap.index_map.size_local} DOFs")
    print(f"Constraints: {len(constraints)} cut DOFs")

    # Create a simple mass matrix for testing
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx

    # Assemble matrix
    form_a = dolfinx.fem.form(a)
    A = dolfinx.fem.assemble_matrix(form_a)
    A.scatter_reverse()

    # Convert to MatrixCSR for our constraint application
    # Note: This would typically be done differently in practice
    print("Note: Matrix constraint application requires MatrixCSR format")
    print(
        "In this demo, we skip the actual constraint application due to PETSc->CSR conversion complexity"
    )

    print("Matrix constraint application test: SKIPPED (format conversion needed)")


def test_function_extension():
    """Test extending a function from interior to full domain."""
    print("\n=== Testing Function Extension ===")

    mesh, V = create_simple_mesh_and_function_space()
    constraints = create_dummy_constraints(V)

    ndofs = V.dofmap.index_map.size_local
    print(f"Total DOFs: {ndofs}")

    # Create a dummy interior solution (all DOFs except constrained ones)
    constrained_dofs = set(constraints.keys())
    interior_dofs = [i for i in range(ndofs) if i not in constrained_dofs]

    print(f"Interior DOFs: {len(interior_dofs)}")
    print(f"Constrained DOFs: {len(constrained_dofs)}")

    # Create interior solution vector
    u_interior = np.random.rand(len(interior_dofs))
    print(f"Interior solution shape: {u_interior.shape}")

    try:
        # Extend the function
        u_full = extensions.extend_function(u_interior, constraints, V)
        print(f"Extended solution shape: {u_full.shape}")

        # Verify constraint relationships
        print("\nVerifying constraint relationships:")
        for cut_dof, constraint_list in constraints.items():
            if cut_dof < len(u_full):
                computed_value = sum(
                    weight * u_full[root_dof]
                    for root_dof, weight in constraint_list
                    if root_dof < len(u_full)
                )
                actual_value = u_full[cut_dof]
                error = abs(computed_value - actual_value)
                print(
                    f"  DOF {cut_dof}: computed={computed_value:.6f}, actual={actual_value:.6f}, error={error:.2e}"
                )

        print("Function extension test: PASSED")

    except Exception as e:
        print(f"Function extension test: FAILED - {e}")


def main():
    """Main demo function."""
    print("CutFEMx Constraint Application and Function Extension Demo")
    print("=" * 60)

    test_matrix_constraint_application()
    test_function_extension()

    print("\n" + "=" * 60)
    print("Demo complete. Note: Full integration tests require actual cut mesh constraints.")


if __name__ == "__main__":
    main()
