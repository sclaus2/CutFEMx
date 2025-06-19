"""
Demo: Badly Cut Cell Identification and Visualization

This demo showcases the new discrete extension functionality by:
1. Creating a mesh with a circular level set function
2. Cutting the mesh to generate cut cells
3. Identifying badly cut cells using volume threshold
4. Visualizing the results with PyVista
"""

import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.extensions import get_badly_cut_parent_indices

from dolfinx import fem, mesh, plot
from dolfinx.io import VTKFile

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

try:
    import cutcells
except ModuleNotFoundError:
    print("cutcells is required for this demo")
    exit(0)


def main():
    print("=== Badly Cut Cell Identification Demo ===")

    # Create a triangular mesh
    N = 31
    print(f"Creating {N}x{N} triangular mesh...")

    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(N, N),
        cell_type=mesh.CellType.triangle,
    )

    gdim = 2
    tdim = msh.topology.dim

    # Create function space for level set
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

    # Estimate mesh size (using approximate cell diameter)
    h_estimate = 2.0 / N  # Approximate mesh size for rectangular domain

    print(f"Estimated mesh size h = {h_estimate:.4f}")

    # Test different threshold factors and store results
    threshold_factors = [0.05, 0.1, 0.2, 0.5]
    badly_cut_results = {}

    for threshold in threshold_factors:
        print(f"\n--- Testing threshold factor: {threshold} ---")

        # Identify badly cut cells
        badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold)

        threshold_ch = threshold * h_estimate
        print(f"Badly cut cells (Ch threshold = {threshold_ch:.6f}): {len(badly_cut_parents)}")
        percentage = 100 * len(badly_cut_parents) / len(cut_cell_entities)
        print(f"Percentage of badly cut cells: {percentage:.1f}%")

        if len(badly_cut_parents) > 0:
            # Show first 10
            print(f"Parent indices of badly cut cells: {badly_cut_parents[:10]}...")

        # Store results for visualization
        badly_cut_results[threshold] = badly_cut_parents

    # Visualize all threshold values with PyVista
    print("\n=== Visualization ===")

    # Create pyvista mesh from dolfinx mesh
    topology, cell_types, x = plot.vtk_mesh(msh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Create level set data for visualization
    level_set_values = level_set.x.array
    grid.point_data["Level_Set"] = level_set_values

    # Create multi-plot: level set + all threshold results
    num_plots = len(threshold_factors) + 1
    cols = 3
    rows = (num_plots + cols - 1) // cols  # Ceiling division

    plotter = pyvista.Plotter(shape=(rows, cols), window_size=(1800, 600 * rows))

    # Plot level set function in first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(grid, scalars="Level_Set", show_edges=True, cmap="RdBu", clim=[-1, 1])
    plotter.add_mesh(grid.contour([0]), color="black", line_width=3, label="Interface")
    plotter.add_title("Level Set Function")
    plotter.add_scalar_bar(title="Level Set Value")

    # Set 2D view for level set plot
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot each threshold result
    colors = ["lightgray", "lightblue", "red"]
    labels = ["Regular cells", "Well-cut cells", "Badly cut cells"]

    for i, threshold in enumerate(threshold_factors):
        row = (i + 1) // cols
        col = (i + 1) % cols

        plotter.subplot(row, col)

        # Create cell data for this threshold
        num_cells = msh.topology.index_map(tdim).size_local
        cell_colors = np.zeros(num_cells)

        # Mark cut cells
        cell_colors[cut_cell_entities] = 1  # Cut cells = 1

        # Mark badly cut cells for this threshold
        badly_cut_parents = badly_cut_results[threshold]
        if len(badly_cut_parents) > 0:
            cell_colors[badly_cut_parents] = 2  # Badly cut cells = 2

        # Create a copy of the grid for this threshold
        grid_copy = grid.copy()
        grid_copy.cell_data["Cell_Type"] = cell_colors

        plotter.add_mesh(
            grid_copy, scalars="Cell_Type", show_edges=True, categories=True, cmap=colors
        )

        # Set 2D view for this plot
        plotter.view_xy()
        plotter.camera.parallel_projection = True

        threshold_ch = threshold * h_estimate
        if len(cut_cell_entities) > 0:
            percentage = 100 * len(badly_cut_parents) / len(cut_cell_entities)
        else:
            percentage = 0
        title = (
            f"Threshold: {threshold}\n(Ch = {threshold_ch:.6f})\n"
            f"{len(badly_cut_parents)} badly cut ({percentage:.1f}%)"
        )
        plotter.add_title(title)

    # Add legend to the last used subplot
    for i, (color, label) in enumerate(zip(colors, labels)):
        plotter.add_mesh(
            pyvista.Sphere(radius=0.001, center=(10, 10, 10)), color=color, label=label
        )
    plotter.add_legend()

    # Ensure all plots have 2D view
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < num_plots:
                plotter.subplot(i, j)
                plotter.view_xy()
                plotter.camera.parallel_projection = True

    plotter.show()

    print("\n=== Demo completed successfully! ===")
    print("Summary:")
    num_cells = msh.topology.index_map(tdim).size_local
    print(f"  - Total cells: {num_cells}")
    print(f"  - Cut cells: {len(cut_cell_entities)}")

    print("  - Badly cut cells by threshold:")
    for threshold in threshold_factors:
        badly_cut_count = len(badly_cut_results[threshold])
        if len(cut_cell_entities) > 0:
            percentage = 100 * badly_cut_count / len(cut_cell_entities)
        else:
            percentage = 0
        threshold_ch = threshold * h_estimate
        print(f"    * {threshold} (Ch={threshold_ch:.6f}): {badly_cut_count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()
