"""
Demo: Free Function Approach for Discrete Extension

This demo showcases the new free function approach for building root-to-cut mappings:
1. build_root_to_cut_mapping - builds mapping from    # Plot 2: Badly cut cells ONLY with level set contour
    plotter.subplot(0, 1)
    # Create a specific color map for badly cut cells
    badly_cut_colors = np.zeros(num_cells)
    badly_cut_colors[badly_cut_parents] = 1  # Badly cut cells = 1

    # Create a dedicated grid for badly cut cells to avoid interference
    badly_cut_grid = grid.copy()
    badly_cut_grid.cell_data.clear()  # Clear all cell data
    badly_cut_grid.cell_data["Badly_Cut"] = badly_cut_colors  # Add only badly cut data

    plotter.add_mesh(
        badly_cut_grid, scalars="Badly_Cut", show_edges=True,
        categories=True, cmap=["lightgray", "red"]
    )
    contour_mesh = badly_cut_grid.contour([0], scalars="Level_Set")
    plotter.add_mesh(contour_mesh, color="black", line_width=3)
    plotter.add_title(f"Badly Cut Cells ONLY ({len(badly_cut_parents)} red cells)")
    plotter.view_xy()
    plotter.camera.parallel_projection = Truenterior cells

Based on the Burman-Hansbo approach for extending values from interior to badly cut cells.
The new design uses free functions instead of classes for better modularity and performance.
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, plot

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.extensions import (
    get_badly_cut_parent_indices,
    build_root_to_cut_mapping,
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
        # Build the mapping from badly cut parents to interior cells
        root_to_cut_mapping = build_root_to_cut_mapping(
            badly_cut_parents, interior_cells, msh, cut_cells
        )
        print("Built root-to-cut mapping")
        print(
            f"Mapping vector length: {len(root_to_cut_mapping)} "
            + f"(should be 2 * {len(badly_cut_parents)})"
        )

        # Print some mapping details (first few entries)
        if len(root_to_cut_mapping) > 0:
            print("Sample mappings (badly_cut -> interior):")
            for i in range(0, min(10, len(root_to_cut_mapping)), 2):
                if i + 1 < len(root_to_cut_mapping):
                    badly_cut = root_to_cut_mapping[i]
                    interior = root_to_cut_mapping[i + 1]
                    print(f"  {badly_cut} -> {interior}")

    except Exception as e:
        print(f"build_root_to_cut_mapping test failed: {e}")
        root_to_cut_mapping = []  # Set empty for visualization

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

    # Create pair visualization data
    pair_colors = np.zeros(num_cells)
    if len(root_to_cut_mapping) > 0:
        # Assign pair IDs to cells (limit to 20 pairs for distinct colors)
        max_pairs = 20

        for i in range(0, min(len(root_to_cut_mapping), max_pairs * 2), 2):
            if i + 1 < len(root_to_cut_mapping):
                badly_cut = root_to_cut_mapping[i]
                interior = root_to_cut_mapping[i + 1]
                pair_id = (i // 2) + 1  # Pair ID starting from 1

                # Assign same color to both cells in the pair
                pair_colors[badly_cut] = pair_id
                pair_colors[interior] = pair_id

    grid.cell_data["Pair_Colors"] = pair_colors

    # Add level set and test function data
    level_set_values = level_set.x.array
    test_values = u_test.x.array
    grid.point_data["Level_Set"] = level_set_values
    grid.point_data["Test_Function"] = test_values

    # Plot
    plotter = pyvista.Plotter(shape=(2, 2), window_size=(1400, 1400))

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

    # Plot 3: Badly cut to interior cell pairs with level set contour
    plotter.subplot(1, 0)
    if len(root_to_cut_mapping) > 0:
        # Create a copy of the grid for the pairs plot to avoid interference
        pairs_grid = grid.copy()
        plotter.add_mesh(pairs_grid, scalars="Pair_Colors", show_edges=True, cmap="tab20")
        contour_mesh = pairs_grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title(
            f"Pairs: {len(root_to_cut_mapping) // 2} badly cut + "
            + f"{len(root_to_cut_mapping) // 2} interior"
        )

        # Add text annotations for the first few pairs
        pair_count = min(5, len(root_to_cut_mapping) // 2)
        print(f"\nShowing first {pair_count} pairs in visualization:")
        for i in range(0, pair_count * 2, 2):
            if i + 1 < len(root_to_cut_mapping):
                badly_cut = root_to_cut_mapping[i]
                interior = root_to_cut_mapping[i + 1]
                print(
                    f"  Pair {i // 2 + 1}: Cell {badly_cut} (badly cut) → "
                    + f"Cell {interior} (interior)"
                )
    else:
        # Fallback to showing badly cut cells
        plotter.add_mesh(
            grid, scalars="Badly_Cut", show_edges=True, categories=True, cmap=["lightgray", "red"]
        )
        contour_mesh = grid.contour([0], scalars="Level_Set")
        plotter.add_mesh(contour_mesh, color="black", line_width=3)
        plotter.add_title("No Pairs Found")
    # plotter.view_xy()
    # plotter.camera.parallel_projection = True

    plotter.show()

    print("\n=== Demo completed ===")
    print("Summary:")
    print(f"  - Total cells: {num_cells}")
    print(f"  - Cut cells: {len(cut_cell_entities)}")
    print(f"  - Badly cut cells: {len(badly_cut_parents)}")
    print(f"  - Well-cut cells: {len(well_cut_parents)}")
    print(f"  - Interior cells: {len(interior_cells)}")
    print(f"  - Successfully mapped pairs: {len(root_to_cut_mapping) // 2}")
    print(f"  - Threshold factor: {threshold_factor}")


if __name__ == "__main__":
    main()
