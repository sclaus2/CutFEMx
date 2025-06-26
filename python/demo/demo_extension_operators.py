"""
Demo: Free Function Approach for Discrete Extension

This demo showcases the new free function approach for building root-to-cut mappings:
1. build_root_to_cut_mapping - builds mapping from badly cut cells to interior cells

Based on the Burman-Hansbo approach for extending values from interior to badly cut cells.
The new design uses free functions instead of classes for better modularity and performance.
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, plot

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.extensions import (
    get_badly_cut_parent_indices,
    build_root_mapping,
    create_root_to_cell_map,
    build_dof_to_cells_mapping,
)

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)


def main():
    print("=== Discrete Extension Operators Demo ===")

    # Create a triangular mesh
    N = 21
    print(f"Creating {N}x{N} triangular mesh...")

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(N, N),
        cell_type=mesh.CellType.triangle,
    )

    tdim = msh.topology.dim

    # Create function space for level set and solution
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define circular level set function (radius = 0.6)
    def circle_level_set(x):
        return np.sqrt((x[0]) ** 2 + (x[1]) ** 2) - 0.6

    print("Creating level set function...")
    level_set = fem.Function(V)
    level_set.interpolate(circle_level_set)

    # Locate entities that are cut by the level set
    print("Locating cut entities...")
    cut_cell_entities = locate_entities(level_set, tdim, "phi=0")

    print(f"Found {len(cut_cell_entities)} cut cells")

    if len(cut_cell_entities) == 0:
        print("No cut cells found! Adjust the level set function.")
        return

    # Cut the entities
    print("Cutting entities...")
    dof_coordinates = V.tabulate_dof_coordinates()
    cut_cells = cut_entities(level_set, dof_coordinates, cut_cell_entities, tdim, "phi<0")

    print(f"Generated {len(cut_cells.cut_cells)} cut cells")

    # Estimate mesh size
    h_estimate = 2.0 / N

    # Identify badly cut cells
    threshold_factor = 0.4
    badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold_factor)

    print(f"Badly cut cells: {len(badly_cut_parents)}")

    if len(badly_cut_parents) == 0:
        print("No badly cut cells found! Try a larger threshold factor.")
        return

    # Identify well-cut cells (cut cells that are not badly cut)
    all_cut_parents = [cut_cells.parent_map[i] for i in range(len(cut_cells.cut_cells))]
    well_cut_parents = [p for p in all_cut_parents if p not in badly_cut_parents]

    print(f"Well-cut cells: {len(well_cut_parents)}")

    # Identify interior cells (cells completely inside the level set, phi < 0)
    print("Identifying interior cells...")
    interior_cell_entities = locate_entities(level_set, tdim, "phi<0")
    # Filter out cut cells to get purely interior cells
    interior_cells = [cell for cell in interior_cell_entities if cell not in cut_cell_entities]

    print(f"Interior cells: {len(interior_cells)}")

    if len(interior_cells) == 0:
        print("No interior cells found! Adjust the level set function.")
        return

    # Create a test function on interior domain for visualization
    def test_function(x):
        return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])

    u_test = fem.Function(V)
    u_test.interpolate(test_function)

    # Test 1: Build root-to-cut mapping using free function
    print("\n=== Testing build_root_to_cut_mapping ===")
    try:
        # Build the mapping from all cut cell parents to interior cells
        root_to_cut_mapping = build_root_mapping(badly_cut_parents, interior_cells, msh, cut_cells)
        print("Built root-to-cut mapping")
        print(
            f"Mapping dictionary size: {len(root_to_cut_mapping)} "
            + f"(maps ALL cut cells, not just {len(badly_cut_parents)} badly cut cells)"
        )

    except Exception as e:
        print(f"build_root_to_cut_mapping test failed: {e}")
        root_to_cut_mapping = {}  # Set empty for visualization

    # Test 2: Create inverse mapping from root cells to cut cells
    print("\n=== Testing create_root_to_cell_map ===")
    inverse_mapping = {}  # Initialize empty for scope
    if len(root_to_cut_mapping) > 0:
        try:
            inverse_mapping = create_root_to_cell_map(root_to_cut_mapping)
            print("Created inverse mapping (root cells → cut cells)")
            print(f"Inverse mapping size: {len(inverse_mapping)} root cells")

            # Print some sample inverse mappings
            print("Sample inverse mappings (root_cell -> [cut_cells]):")
            count = 0
            for root_cell, cut_cells_list in inverse_mapping.items():
                if count >= 40:  # Show first 5 mappings
                    break
                print(f"  {root_cell} -> {cut_cells_list}")
                count += 1

            # Verify the mapping consistency
            total_cut_cells_in_inverse = sum(
                len(cut_cells_list) for cut_cells_list in inverse_mapping.values()
            )
            print(f"Total cut cells in inverse mapping: {total_cut_cells_in_inverse}")
            print(f"Original mapping size: {len(root_to_cut_mapping)}")

            if total_cut_cells_in_inverse == len(root_to_cut_mapping):
                print("✓ Inverse mapping is consistent with original mapping")
            else:
                print(
                    "⚠ Inverse mapping size mismatch - some root cells may map "
                    "to multiple cut cells"
                )

        except Exception as e:
            print(f"create_root_to_cell_map test failed: {e}")
    else:
        print("Skipping inverse mapping test (no root mapping available)")

    # Diagnostic: Check if all cut cells are mapped
    print("\n=== Diagnostic Information ===")

    # Get all unique cut cell parents from cut_cells.parent_map
    all_cut_cell_parents_from_map = list(set(cut_cells.parent_map))
    mapped_cut_cells = set(root_to_cut_mapping.keys())
    unmapped_cut_cells = set(all_cut_cell_parents_from_map) - mapped_cut_cells

    print(f"Total unique cut cell parents in parent_map: {len(all_cut_cell_parents_from_map)}")
    print(f"Cut cells mapped by algorithm: {len(mapped_cut_cells)}")
    print(f"Unmapped cut cells: {len(unmapped_cut_cells)}")

    if unmapped_cut_cells:
        print(f"Unmapped cut cell IDs: {sorted(list(unmapped_cut_cells))}")

        # Check if unmapped cells are interior, cut, or isolated
        unmapped_in_cut_entities = [
            cell for cell in unmapped_cut_cells if cell in cut_cell_entities
        ]
        unmapped_in_interior = [cell for cell in unmapped_cut_cells if cell in interior_cells]

        print(
            f"  - Unmapped cells that are also in cut_cell_entities: "
            f"{len(unmapped_in_cut_entities)}"
        )
        print(f"  - Unmapped cells that are also in interior_cells: {len(unmapped_in_interior)}")

        # These might be isolated cut cells with no path to interior cells
        print(
            "This suggests some cut cells are isolated and cannot reach "
            "interior cells via connectivity."
        )

    # Test 3: Build dof-to-cells mapping
    print("\n=== Testing build_dof_to_cells_mapping ===")
    dof_to_cells_mapping = {}  # Initialize empty for scope
    if len(root_to_cut_mapping) > 0:
        try:
            dof_to_cells_mapping = build_dof_to_cells_mapping(cut_cells, root_to_cut_mapping, V)
            print("Created dof-to-cells mapping (cut boundary DOFs only)")
            print(f"Dof-to-cells mapping size: {len(dof_to_cells_mapping)} DOFs")

            # Print some sample dof mappings
            print("Sample dof mappings (cut boundary DOF -> [cut cells]):")
            count = 0
            for dof_id, cells_list in dof_to_cells_mapping.items():
                if count >= 10:  # Show first 10 mappings
                    break
                print(f"  DOF {dof_id} -> {cells_list} ({len(cells_list)} cells)")
                count += 1

            # Verify the mapping statistics
            total_dof_cell_pairs = sum(
                len(cells_list) for cells_list in dof_to_cells_mapping.values()
            )
            max_cells_per_dof = (
                max(len(cells_list) for cells_list in dof_to_cells_mapping.values())
                if dof_to_cells_mapping
                else 0
            )
            avg_cells_per_dof = (
                total_dof_cell_pairs / len(dof_to_cells_mapping) if dof_to_cells_mapping else 0
            )

            print(f"Total dof-cell pairs: {total_dof_cell_pairs}")
            print(f"Average cells per DOF: {avg_cells_per_dof:.2f}")
            print(f"Maximum cells per DOF: {max_cells_per_dof}")

        except Exception as e:
            print(f"build_dof_to_cells_mapping test failed: {e}")
    else:
        print("Skipping dof-to-cells mapping test (no root mapping available)")

    # DOF Visualization
    print("\n=== DOF Visualization ===")
    if len(dof_to_cells_mapping) > 0:
        try:
            # Get DOF coordinates for the function space
            dof_coords = V.tabulate_dof_coordinates()
            print(f"Total DOFs in function space: {len(dof_coords)}")

            # Extract coordinates of the mapped DOFs (cut boundary DOFs)
            mapped_dof_indices = list(dof_to_cells_mapping.keys())
            mapped_dof_coords = dof_coords[mapped_dof_indices]

            print(f"Cut boundary DOFs being mapped: {len(mapped_dof_coords)}")
            print(f"Mapped DOF indices (first 20): {mapped_dof_indices[:20]}")

            # Create a simple point cloud for the mapped DOFs
            dof_point_cloud = pyvista.PolyData(mapped_dof_coords)

            # Add information about how many cells each DOF maps to
            cells_per_dof = [len(dof_to_cells_mapping[dof_idx]) for dof_idx in mapped_dof_indices]
            dof_point_cloud.point_data["Cells_Per_DOF"] = cells_per_dof

            print(f"DOF point cloud created with {len(mapped_dof_coords)} points")
            print(f"Cells per DOF range: {min(cells_per_dof)} to {max(cells_per_dof)}")

        except Exception as e:
            print(f"DOF visualization preparation failed: {e}")
            dof_point_cloud = None
    else:
        print("No DOF mapping available for visualization")
        dof_point_cloud = None

    # Visualization
    print("\n=== Visualization ===")

    # Create pyvista mesh from dolfinx mesh
    topology, cell_types, x = plot.vtk_mesh(msh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Create cell data for visualization
    num_cells = msh.topology.index_map(tdim).size_local
    cell_colors = np.zeros(num_cells)

    # Mark cut cells
    cell_colors[cut_cell_entities] = 1  # Cut cells = 1

    # Mark badly cut cells
    if len(badly_cut_parents) > 0:
        cell_colors[badly_cut_parents] = 2  # Badly cut cells = 2

    grid.cell_data["Cell_Type"] = cell_colors

    # Create pair visualization data using inverse mapping
    pair_colors = np.zeros(num_cells)
    if len(inverse_mapping) > 0:
        # Assign colors based on inverse mapping (limit to 20 colors for distinct visualization)
        max_colors = 40

        for color_id, (root_cell, cut_cells_list) in enumerate(inverse_mapping.items(), start=1):
            if color_id > max_colors:
                break

            # Assign the same color to the root cell and all its cut cells
            pair_colors[root_cell] = color_id + 1
            for cut_cell in cut_cells_list:
                pair_colors[cut_cell] = color_id + 1

        print(f"Assigned colors to {min(len(inverse_mapping), max_colors)} root-cut cell groups")

    grid.cell_data["Pair_Colors"] = pair_colors

    # Add level set and test function data
    level_set_values = level_set.x.array
    test_values = u_test.x.array
    grid.point_data["Level_Set"] = level_set_values
    grid.point_data["Test_Function"] = test_values

    # Plot
    plotter = pyvista.Plotter(shape=(2, 3), window_size=(2000, 1400))

    # Plot 1: Level set
    plotter.subplot(0, 0)
    plotter.add_mesh(grid, scalars="Level_Set", show_edges=True, cmap="RdBu", clim=[-1, 1])
    # Create contour from level set data
    contour_mesh = grid.contour([0], scalars="Level_Set")
    plotter.add_mesh(contour_mesh, color="black", line_width=3)
    plotter.add_title("Level Set Function")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot 2: Badly cut cells with level set contour
    plotter.subplot(0, 1)
    # Create a specific color map for badly cut cells
    badly_cut_colors = np.zeros(num_cells)
    badly_cut_colors[badly_cut_parents] = 1  # Badly cut cells = 1
    grid.cell_data["Badly_Cut"] = badly_cut_colors

    plotter.add_mesh(
        grid, scalars="Badly_Cut", show_edges=True, categories=True, cmap=["lightgray", "red"]
    )
    contour_mesh = grid.contour([0], scalars="Level_Set")
    plotter.add_mesh(contour_mesh, color="black", line_width=3)
    plotter.add_title(f"ONLY Badly Cut Cells ({len(badly_cut_parents)} red cells)")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot 3: Root-to-cut cell groups with level set contour (using inverse mapping)
    plotter.subplot(1, 0)
    if len(inverse_mapping) > 0:
        # Create a copy of the grid for the groups plot to avoid interference
        groups_grid = grid.copy()
        plotter.add_mesh(groups_grid, scalars="Pair_Colors", show_edges=True, cmap="tab20")
        contour_mesh = groups_grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title(f"Root-Cut Cell Groups: {len(inverse_mapping)} root cells")

        # Add text annotations for the first few groups
        group_count = min(5, len(inverse_mapping))
        print(f"\nShowing first {group_count} root-cut cell groups in visualization:")
        for i, (root_cell, cut_cells_list) in enumerate(inverse_mapping.items()):
            if i >= group_count:
                break
            print(
                f"  Group {i + 1}: Root cell {root_cell} → "
                + f"Cut cells {cut_cells_list} ({len(cut_cells_list)} cells)"
            )
    else:
        # Fallback to showing badly cut cells
        plotter.add_mesh(
            grid, scalars="Badly_Cut", show_edges=True, categories=True, cmap=["lightgray", "red"]
        )
        contour_mesh = grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title("No Root-Cut Groups Found")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot 4: Root cells only (interior cells that serve as extension sources)
    plotter.subplot(1, 1)
    if len(inverse_mapping) > 0:
        # Create root cell visualization
        root_colors = np.zeros(num_cells)
        for color_id, root_cell in enumerate(inverse_mapping.keys(), start=1):
            # Only color the root cells, not the cut cells
            root_colors[root_cell] = color_id

        # Create a dedicated grid for root cells
        root_grid = grid.copy()
        root_grid.cell_data.clear()
        root_grid.cell_data["Root_Colors"] = root_colors

        plotter.add_mesh(root_grid, scalars="Root_Colors", show_edges=True, cmap="tab20")
        contour_mesh = root_grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title(f"Root Cells Only: {len(inverse_mapping)} interior extension sources")

        # Print statistics about root cell distribution
        cut_cells_per_root = [len(cut_cells_list) for cut_cells_list in inverse_mapping.values()]
        avg_cuts_per_root = np.mean(cut_cells_per_root) if cut_cells_per_root else 0
        max_cuts_per_root = max(cut_cells_per_root) if cut_cells_per_root else 0
        print(
            f"Root cell statistics: avg {avg_cuts_per_root:.1f} cut cells per root, "
            f"max {max_cuts_per_root} cut cells per root"
        )
    else:
        # Fallback to showing interior cells
        interior_colors = np.zeros(num_cells)
        interior_colors[interior_cells] = 1
        plotter.add_mesh(
            grid,
            scalars=interior_colors,
            show_edges=True,
            categories=True,
            cmap=["lightgray", "blue"],
        )
        contour_mesh = grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title("Interior Cells (No Root Mapping)")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot 5: DOF mapping visualization - mesh with DOF points overlaid
    plotter.subplot(0, 2)
    if dof_point_cloud is not None:
        # Show the mesh as background with cut cells highlighted
        dof_mesh_grid = grid.copy()
        plotter.add_mesh(
            dof_mesh_grid,
            scalars="Cell_Type",
            show_edges=True,
            categories=True,
            cmap=["lightgray", "yellow", "red"],
            opacity=0.7,
        )

        # Add level set contour
        contour_mesh = dof_mesh_grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=2)

        # Overlay the DOF points
        plotter.add_mesh(
            dof_point_cloud,
            scalars="Cells_Per_DOF",
            point_size=15,
            render_points_as_spheres=True,
            cmap="viridis",
            clim=[1, max(cells_per_dof)],
        )

        plotter.add_title(f"Cut Boundary DOFs: {len(mapped_dof_coords)} DOFs")
    else:
        # Fallback - show regular mesh
        plotter.add_mesh(
            grid,
            scalars="Cell_Type",
            show_edges=True,
            categories=True,
            cmap=["lightgray", "yellow", "red"],
        )
        contour_mesh = grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title("No DOF Data Available")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot 6: DOF points only with color based on connectivity
    plotter.subplot(1, 2)
    if dof_point_cloud is not None:
        # Show just the DOF points
        plotter.add_mesh(
            dof_point_cloud,
            scalars="Cells_Per_DOF",
            point_size=20,
            render_points_as_spheres=True,
            cmap="plasma",
            clim=[1, max(cells_per_dof)],
        )

        # Add a background grid for reference
        plotter.add_mesh(grid, style="wireframe", color="lightgray", opacity=0.3, line_width=1)

        # Add level set contour for reference
        contour_mesh = grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=2)

        dof_range = f"{min(cells_per_dof)}-{max(cells_per_dof)}"
        plotter.add_title(f"DOF Connectivity: {dof_range} cells per DOF")

        # Print some statistics about DOF distribution
        print("DOF connectivity statistics:")
        print(f"  - DOFs connected to 1 cell: {sum(1 for c in cells_per_dof if c == 1)}")
        print(f"  - DOFs connected to 2 cells: {sum(1 for c in cells_per_dof if c == 2)}")
        print(f"  - DOFs connected to 3+ cells: {sum(1 for c in cells_per_dof if c >= 3)}")
    else:
        # Fallback - show regular mesh
        plotter.add_mesh(grid, style="wireframe", color="gray")
        plotter.add_title("No DOF Point Cloud Available")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    plotter.show()

    print("\n=== Demo completed ===")
    print("Summary:")
    print(f"  - Total cells: {num_cells}")
    print(f"  - Cut cells: {len(cut_cell_entities)}")
    print(f"  - Badly cut cells: {len(badly_cut_parents)}")
    print(f"  - Well-cut cells: {len(well_cut_parents)}")
    print(f"  - Interior cells: {len(interior_cells)}")
    print(f"  - Cut cells mapped to roots: {len(root_to_cut_mapping)} (all cut cells)")
    print(f"  - Root cells (extension sources): {len(inverse_mapping)}")
    if len(inverse_mapping) > 0:
        total_cut_cells_mapped = sum(
            len(cut_cells_list) for cut_cells_list in inverse_mapping.values()
        )
        print(f"  - Verification: {total_cut_cells_mapped} cut cells assigned to roots")
    if len(dof_to_cells_mapping) > 0:
        total_dofs = len(V.tabulate_dof_coordinates())
        mapped_dofs = len(dof_to_cells_mapping)
        print(f"  - Total DOFs in function space: {total_dofs}")
        print(f"  - Cut boundary DOFs requiring extension: {mapped_dofs}")
        print(f"  - DOF mapping coverage: {mapped_dofs / total_dofs * 100:.1f}%")
    print(f"  - Threshold factor: {threshold_factor}")


if __name__ == "__main__":
    main()
