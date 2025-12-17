"""
Focused Demo: DOF-to-Root Mapping Visualization

This demo focuses specifically on visualizing the DOF-to-root mapping,
showing DOF locations and their corresponding root cells with matching colors.
"""

import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh, plot

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.extensions import (
    get_badly_cut_parent_indices,
    build_root_mapping,
    build_dof_to_cells_mapping,
    build_dof_to_root_mapping,
)

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)


def main():
    print("=== DOF-to-Root Mapping Visualization Demo ===")

    # Color palette options for 35+ distinct colors
    # Based on colormap analysis:
    # 'hsv' - EXCELLENT (35/35 distinct colors) - Hue-Saturation-Value cycle
    # 'jet' - EXCELLENT (34/35 distinct colors) - Classic jet colormap
    # 'gist_ncar' - EXCELLENT (33/35 distinct colors) - NCAR graphics
    # 'nipy_spectral' - GOOD (26/35 distinct colors) - Spectral colors
    COLOR_PALETTE = "hsv"  # Best for 35 distinct colors (perfect coverage)

    print(f"Using colormap: {COLOR_PALETTE} (PERFECT coverage for 35 distinct colors)")

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

    # Create function space
    V = fem.functionspace(msh, ("Lagrange", 1))

    # Define circular level set function (radius = 0.6)
    def circle_level_set(x):
        return np.sqrt((x[0]) ** 2 + (x[1]) ** 2) - 0.6

    print("Creating level set function...")
    level_set = fem.Function(V)
    level_set.interpolate(circle_level_set)

    # Locate and cut entities
    print("Locating cut entities...")
    cut_cell_entities = locate_entities(level_set, tdim, "phi=0")
    print(f"Found {len(cut_cell_entities)} cut cells")

    if len(cut_cell_entities) == 0:
        print("No cut cells found!")
        return

    print("Cutting entities...")
    dof_coordinates = V.tabulate_dof_coordinates()
    cut_cells = cut_entities(level_set, dof_coordinates, cut_cell_entities, tdim, "phi<0")

    # Identify badly cut cells and interior cells
    h_estimate = 2.0 / N
    threshold_factor = 0.4
    badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold_factor)
    interior_cell_entities = locate_entities(level_set, tdim, "phi<0")
    interior_cells = [cell for cell in interior_cell_entities if cell not in cut_cell_entities]

    print(f"Badly cut cells: {len(badly_cut_parents)}")
    print(f"Interior cells: {len(interior_cells)}")

    # Build mappings
    print("Building root mapping...")
    root_to_cut_mapping = build_root_mapping(badly_cut_parents, interior_cells, msh, cut_cells)

    print("Building dof-to-cells mapping...")
    dof_to_cells_mapping = build_dof_to_cells_mapping(cut_cells, root_to_cut_mapping, V)

    print("Building dof-to-root mapping...")
    dof_to_root_mapping = build_dof_to_root_mapping(
        dof_to_cells_mapping, root_to_cut_mapping, V, msh
    )

    print(f"DOF-to-root mapping created for {len(dof_to_root_mapping)} DOFs")

    # Prepare visualization data
    num_cells = msh.topology.index_map(tdim).size_local
    topology, cell_types, x = plot.vtk_mesh(msh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)

    # Add level set data to grid
    level_set_values = level_set.x.array.real
    grid.point_data["Level_Set"] = level_set_values

    print("\n=== Creating Visualization ===")

    # Create color mapping for DOF-to-root visualization
    unique_root_cells = list(set(dof_to_root_mapping.values()))
    print(f"Unique root cells used: {len(unique_root_cells)}")

    # Assign each unique root cell a color index starting from 1
    root_cell_to_color = {root_cell: i + 1 for i, root_cell in enumerate(unique_root_cells)}

    # Create arrays for visualization
    dof_colors = []
    dof_coords_for_viz = []
    root_cell_colors = np.zeros(num_cells)

    # Color DOFs based on their assigned root cells - use SAME color integer as root
    for dof_id, root_cell in dof_to_root_mapping.items():
        color_value = root_cell_to_color[root_cell]  # Same integer as root cell
        dof_colors.append(color_value)

        # Get DOF coordinates
        dof_coord = V.tabulate_dof_coordinates()[dof_id]
        dof_coords_for_viz.append(dof_coord)

    # Color root cells - use SAME color integers as DOFs
    for root_cell in unique_root_cells:
        if root_cell < len(root_cell_colors):
            color_value = root_cell_to_color[root_cell]  # Same integer as DOFs
            root_cell_colors[root_cell] = color_value

    # Create DOF point cloud
    dof_coords_array = np.array(dof_coords_for_viz)
    dof_point_cloud = pyvista.PolyData(dof_coords_array)
    dof_point_cloud["Root_Cell_Color"] = np.array(dof_colors)

    # Add root cell colors to grid
    grid.cell_data["Root_Cell_Colors"] = root_cell_colors

    # Create visualization
    plotter = pyvista.Plotter(shape=(1, 2), window_size=(1600, 800))

    # Plot 1: Overview with all elements
    plotter.subplot(0, 0)

    # Add mesh with root cells colored
    plotter.add_mesh(
        grid,
        scalars="Root_Cell_Colors",
        show_edges=True,
        cmap=COLOR_PALETTE,  # Use the selected colormap
        opacity=0.8,
        clim=[1, len(unique_root_cells)],  # Explicitly set color range starting from 1
    )

    # Add level set contour
    contour_mesh = grid.contour([0], scalars="Level_Set")
    plotter.add_mesh(contour_mesh, color="black", line_width=3)

    # Add DOF points with matching colors
    plotter.add_mesh(
        dof_point_cloud,
        scalars="Root_Cell_Color",
        point_size=25,
        render_points_as_spheres=True,
        cmap=COLOR_PALETTE,  # Same colormap as cells
        clim=[1, len(unique_root_cells)],  # Same color range as cells
    )

    plotter.add_title(f"DOF-Root Mapping: {len(unique_root_cells)} root cells")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    # Plot 2: Focus on DOFs and nearby root cells only
    plotter.subplot(0, 1)

    # Create a mask to show only relevant cells (root cells + some neighbors)
    relevant_cells = set(unique_root_cells)

    # Add neighboring cells for context
    mesh_coords = msh.geometry.x
    cells = msh.topology.connectivity(tdim, 0).array.reshape(-1, 3)

    for root_cell in unique_root_cells:
        if root_cell < len(cells):
            # Add cells within a small radius of root cell
            root_vertices = cells[root_cell]
            root_coords = mesh_coords[root_vertices]
            root_center = np.mean(root_coords, axis=0)

            for cell_id in range(num_cells):
                if cell_id < len(cells):
                    cell_vertices = cells[cell_id]
                    cell_coords = mesh_coords[cell_vertices]
                    cell_center = np.mean(cell_coords, axis=0)

                    # If cell center is close to root center, include it
                    distance = np.linalg.norm(cell_center - root_center)
                    if distance < 0.3:  # Adjust threshold as needed
                        relevant_cells.add(cell_id)

    # Create subset colors
    subset_colors = np.zeros(num_cells)
    for cell_id in relevant_cells:
        if cell_id in unique_root_cells:
            color_value = root_cell_to_color[cell_id]  # Same integer as DOFs
            subset_colors[cell_id] = color_value
        else:
            subset_colors[cell_id] = len(unique_root_cells) + 2  # Different value for context cells

    grid_subset = grid.copy()
    grid_subset.cell_data["Subset_Colors"] = subset_colors

    # Show only relevant cells
    plotter.add_mesh(
        grid_subset,
        scalars="Subset_Colors",
        show_edges=True,
        cmap=COLOR_PALETTE,  # Use the selected colormap
        opacity=0.9,
        clim=[1, len(unique_root_cells)],  # Same color range
    )

    # Add contour
    contour_mesh = grid_subset.contour([0], scalars="Level_Set")
    plotter.add_mesh(contour_mesh, color="black", line_width=3)

    # Add DOF points
    plotter.add_mesh(
        dof_point_cloud,
        scalars="Root_Cell_Color",
        point_size=30,
        render_points_as_spheres=True,
        cmap=COLOR_PALETTE,  # Same colormap as cells
        clim=[1, len(unique_root_cells)],  # Same color range as cells
    )

    plotter.add_title("DOF-Root Details: Zoomed View")
    plotter.view_xy()
    plotter.camera.parallel_projection = True

    print("\n=== Color Mapping Debug ===")
    print(f"Root cell to color mapping: {dict(list(root_cell_to_color.items())[:5])}...")
    print(f"DOF color range: {min(dof_colors)} to {max(dof_colors)}")
    root_colors_nonzero = root_cell_colors[root_cell_colors > 0]
    print(f"Root cell color range: {min(root_colors_nonzero)} to {max(root_cell_colors)}")
    print(f"Background cells (value 0): {np.sum(root_cell_colors == 0)} cells")
    print(f"Colored root cells: {np.sum(root_cell_colors > 0)} cells")

    print("\n=== Example DOF-Root Assignments ===")
    for i, (dof_id, root_cell) in enumerate(dof_to_root_mapping.items()):
        if i >= 10:  # Show first 10 examples
            break
        color_value = root_cell_to_color[root_cell]
        dof_coord = V.tabulate_dof_coordinates()[dof_id]
        print(
            f"DOF {dof_id:3d} at ({dof_coord[0]:6.3f}, {dof_coord[1]:6.3f}) "
            f"→ Root cell {root_cell:3d} (color {color_value:2d})"
        )

    # Verify color consistency
    print("\n=== Color Consistency Verification ===")
    for i, (dof_id, root_cell) in enumerate(dof_to_root_mapping.items()):
        if i >= 3:  # Check first 3 examples
            break
        dof_color = root_cell_to_color[root_cell]
        root_color = root_cell_colors[root_cell]
        dof_coord = V.tabulate_dof_coordinates()[dof_id]

        if dof_color == root_color:
            status = "✓ MATCH"
        else:
            status = "✗ MISMATCH"

        print(
            f"DOF {dof_id} → Root {root_cell}: "
            f"DOF_color={dof_color}, Root_color={root_color} {status}"
        )

    print(
        f"\nShowing visualization with {len(unique_root_cells)} different colors (starting from 1)"
    )
    print(f"• Using '{COLOR_PALETTE}' colormap - PERFECT for 35 distinct colors")
    print("• Each color represents one root cell")
    print("• DOFs have the EXACT SAME color integer as their assigned root cell")
    print("• This shows which DOFs will get values extended from which root cells")
    print("\nNote: Top colormaps for 35+ colors (tested):")
    print("  - 'hsv': Hue-Saturation-Value (35/35 distinct - PERFECT)")
    print("  - 'jet': Classic jet colormap (34/35 distinct)")
    print("  - 'gist_ncar': NCAR graphics (33/35 distinct)")
    print("  - 'nipy_spectral': Spectral colors (26/35 distinct)")

    plotter.show()


if __name__ == "__main__":
    main()
