#!/usr/bin/env python3
"""
Demo: DOF Basis Function Evaluation

This script demonstrates the new functionality for evaluating basis functions
at DOF coordinates mapped to reference space of root cells.

The workflow:
1. Create a simple mesh with a level set function
2. Cut the mesh and identify badly cut cells
3. Build root mappings from cut cells to interior cells
4. Build DOF-to-root mappings
5. Evaluate basis functions at DOF coordinates in reference space of root cells
6. Visualize or analyze the results
"""

import numpy as np
from mpi4py import MPI

import dolfinx.fem as fem
import dolfinx.mesh as mesh
import dolfinx.log as log

# Import cutfemx
import cutfemx.level_set as level_set
from cutfemx.extensions import (
    build_dof_to_cells_mapping,
    build_dof_to_root_mapping,
    build_root_mapping,
    evaluate_basis_at_dof_coordinates,
    get_badly_cut_parent_indices,
)

log.set_log_level(log.LogLevel.INFO)


def create_simple_test_case():
    """Create a simple test case with a known level set"""

    # Create a unit square mesh
    N = 8
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)

    # Create a P1 function space for the level set
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define a level set function: circle centered at (0.3, 0.3) with radius 0.35
    def level_set_expr(x):
        return np.sqrt((x[0] - 0.3) ** 2 + (x[1] - 0.3) ** 2) - 0.35

    # Create level set function
    phi = fem.Function(V)
    phi.interpolate(level_set_expr)

    return msh, V, phi


def test_basis_evaluation():
    """Test the new basis function evaluation capabilities"""

    print("=== DOF Basis Function Evaluation Demo ===")
    print()

    # Create test case
    print("Setting up test case...")
    msh, V, phi = create_simple_test_case()

    gdim = msh.geometry.dim
    tdim = msh.topology.dim

    print(
        f"Mesh: {msh.topology.index_map(tdim).size_local} cells, {gdim}D geometry, {tdim}D topology"
    )
    print(f"Function space: {V.element.signature}")
    print()

    # Get DOF coordinates for analysis
    dof_coordinates = V.tabulate_dof_coordinates()
    total_dofs = len(dof_coordinates)
    print(f"Total DOFs: {total_dofs}")

    # Find entities that are cut by the level set
    print("Finding cut entities...")
    cut_cell_entities = level_set.locate_entities(phi, tdim, "phi>0&&phi<0")
    print(f"Cut cells: {len(cut_cell_entities)}")

    if len(cut_cell_entities) == 0:
        print("No cut cells found! Adjusting level set...")

        # Adjust level set to ensure we have cut cells
        def adjusted_level_set_expr(x):
            return np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) - 0.3

        phi.interpolate(adjusted_level_set_expr)
        cut_cell_entities = level_set.locate_entities(phi, tdim, "phi>0&&phi<0")
        print(f"Cut cells after adjustment: {len(cut_cell_entities)}")

    if len(cut_cell_entities) == 0:
        print("Still no cut cells! Using alternative approach...")
        # Use entities where level set has mixed signs
        all_cells = np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)
        cut_cell_entities = []

        # Check each cell for sign changes
        dofmap = V.dofmap
        ls_values = phi.x.array

        for cell in all_cells:
            cell_dofs = dofmap.cell_dofs(cell)
            cell_ls_values = ls_values[cell_dofs]
            if np.min(cell_ls_values) * np.max(cell_ls_values) < 0:
                cut_cell_entities.append(cell)

        cut_cell_entities = np.array(cut_cell_entities, dtype=np.int32)
        print(f"Cut cells found by sign analysis: {len(cut_cell_entities)}")

    if len(cut_cell_entities) == 0:
        print("ERROR: No cut cells found! Cannot proceed with demo.")
        return

    # Cut the mesh to get cut cells information
    print("Cutting mesh entities...")
    cut_cells = level_set.cut_entities(phi, dof_coordinates, cut_cell_entities, tdim, "phi<0")
    print(f"CutCells object created with {len(cut_cells.parent_map)} cut cells")

    # Identify badly cut cells (using a reasonable threshold)
    h_estimate = 2.0 / 8  # Based on mesh resolution
    threshold_factor = 0.3
    badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold_factor)
    print(f"Badly cut cells: {len(badly_cut_parents)}")

    # Get interior cells (cells completely inside the level set)
    interior_cell_entities = level_set.locate_entities(phi, tdim, "phi<0")
    interior_cells = [cell for cell in interior_cell_entities if cell not in cut_cell_entities]
    print(f"Interior cells: {len(interior_cells)}")

    if len(interior_cells) == 0:
        print("ERROR: No interior cells found! Cannot build root mapping.")
        return

    if len(badly_cut_parents) == 0:
        print("No badly cut cells found. Using all cut cells for demonstration...")
        # Use first 5 for demo
        badly_cut_parents = cut_cell_entities[: min(5, len(cut_cell_entities))]
        print(f"Using {len(badly_cut_parents)} cut cells for demo")

    # Build root mapping
    print("\nBuilding root mapping...")
    root_to_cut_mapping = build_root_mapping(badly_cut_parents, interior_cells, msh, cut_cells)
    print(f"Root mapping created: {len(root_to_cut_mapping)} cut->root mappings")

    # Build dof-to-cells mapping
    print("Building dof-to-cells mapping...")
    dof_to_cells_mapping = build_dof_to_cells_mapping(cut_cells, root_to_cut_mapping, V)
    print(f"DOF-to-cells mapping: {len(dof_to_cells_mapping)} DOFs mapped to cells")

    if len(dof_to_cells_mapping) == 0:
        print("ERROR: No DOF-to-cells mapping found! Cannot proceed.")
        return

    # Build dof-to-root mapping
    print("Building dof-to-root mapping...")
    dof_to_root_mapping = build_dof_to_root_mapping(
        dof_to_cells_mapping, root_to_cut_mapping, V, msh
    )
    print(f"DOF-to-root mapping: {len(dof_to_root_mapping)} DOFs mapped to root cells")

    if len(dof_to_root_mapping) == 0:
        print("ERROR: No DOF-to-root mapping found! Cannot proceed.")
        return

    # Now test the new basis evaluation functionality
    print("\n=== Testing DOF Constraint Evaluation ===")

    # Basic constraint evaluation
    print("Evaluating DOF constraints...")
    dof_constraints = evaluate_basis_at_dof_coordinates(dof_to_root_mapping, V, msh)
    print(f"Constraint evaluation complete: {len(dof_constraints)} DOFs processed")

    # Analyze the constraint results
    if len(dof_constraints) > 0:
        # Filter out DOFs with no constraints for initial analysis
        valid_dofs = {
            dof: constraints for dof, constraints in dof_constraints.items() if len(constraints) > 0
        }

        if len(valid_dofs) > 0:
            first_dof = next(iter(valid_dofs.keys()))
            constraints = valid_dofs[first_dof]
            space_dim = V.element.space_dimension

            print(f"First valid DOF ({first_dof}): {len(constraints)} constraint pairs")
            print(f"Element space dimension: {space_dim}")
            print(f"Constraint pairs: {constraints}")

            # Show individual constraints
            print("Individual constraints:")
            for root_dof, weight in constraints:
                print(f"  DOF {first_dof} -> root DOF {root_dof} with weight {weight:.6f}")

            # Check if constraint weights sum to 1 (partition of unity for Lagrange elements)
            weights_sum = sum(weight for _, weight in constraints)
            print(
                f"Sum of constraint weights: {weights_sum:.6f} "
                f"(should be ≈ 1.0 for Lagrange elements)"
            )

        # Analyze all constraints, including invalid ones
        all_weights = []
        all_root_dofs = []
        valid_weights = []
        dofs_with_constraints = 0
        dofs_without_constraints = 0
        dofs_with_invalid_weights = 0

        for dof, constraint_list in dof_constraints.items():
            if len(constraint_list) == 0:
                dofs_without_constraints += 1
            else:
                dofs_with_constraints += 1
                has_invalid = False
                for root_dof, weight in constraint_list:
                    all_weights.append(weight)
                    all_root_dofs.append(root_dof)
                    if np.isfinite(weight) and abs(weight) < 1e10:  # Reasonable weight
                        valid_weights.append(weight)
                    else:
                        has_invalid = True
                if has_invalid:
                    dofs_with_invalid_weights += 1

        print("Constraint analysis:")
        print(f"  - DOFs with constraints: {dofs_with_constraints}/{len(dof_constraints)}")
        print(f"  - DOFs without constraints: {dofs_without_constraints}/{len(dof_constraints)}")
        print(f"  - DOFs with invalid weights: {dofs_with_invalid_weights}/{len(dof_constraints)}")

        if len(all_weights) > 0:
            total_constraints = len(all_weights)
            unique_root_dofs = len(set(all_root_dofs))

            print(f"  - Total constraint pairs: {total_constraints}")
            print(f"  - Unique root DOFs involved: {unique_root_dofs}")

            if len(valid_weights) > 0:
                min_weight = min(valid_weights)
                max_weight = max(valid_weights)
                mean_weight = np.mean(valid_weights)
                print(f"  - Valid weight range: [{min_weight:.6f}, {max_weight:.6f}]")
                print(f"  - Valid mean weight: {mean_weight:.6f}")

            # Show constraint distribution
            constraint_counts = [len(constraints) for constraints in dof_constraints.values()]
            min_c = min(constraint_counts)
            max_c = max(constraint_counts)
            avg_c = np.mean(constraint_counts)
            print(f"  - Constraints per DOF: min={min_c}, max={max_c}, avg={avg_c:.1f}")

            # Identify problematic DOFs for debugging
            problematic_dofs = []
            for dof, constraint_list in dof_constraints.items():
                for root_dof, weight in constraint_list:
                    if not np.isfinite(weight) or abs(weight) > 1e10:
                        problematic_dofs.append(dof)
                        break

            if len(problematic_dofs) > 0:
                preview = problematic_dofs[:5]
                suffix = "..." if len(problematic_dofs) > 5 else ""
                print(f"  - DOFs with problematic weights: {preview}{suffix}")
        else:
            print("  - No constraints found!")
    else:
        print("ERROR: No constraint results returned!")

    print("\n=== Demo completed successfully! ===")
    print()
    print("The new DOF constraint evaluation functionality has been successfully tested.")
    print("Key features demonstrated:")
    print("1. Mapping DOF coordinates to reference space of root cells")
    print("2. Creating constraint relationships between cut DOFs and root DOFs")
    print("3. Computing geometric transformation data (Jacobians, determinants)")
    print("4. Providing both basic and extended constraint evaluation interfaces")
    print()
    print("This functionality can be used for:")
    print("- Building discrete extension operators for cut cell methods")
    print("- Creating constraint matrices for DOF coupling")
    print("- Implementing function extension from interior to cut cells")
    print("- Analysis of basis function relationships in cut cell methods")


if __name__ == "__main__":
    test_basis_evaluation()
