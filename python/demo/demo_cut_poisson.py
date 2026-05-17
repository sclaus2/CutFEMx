# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Incremental CutFEMx Poisson demo.

This file is intentionally developed one migration step at a time.  The final
target is the cut Poisson example; the current migration slice verifies the new
persistent MeshCutter workflow.
"""

from __future__ import annotations

import argparse

import numpy as np
from mpi4py import MPI

import cutfemx
from dolfinx import fem, mesh


def circle_level_set(x: np.ndarray) -> np.ndarray:
    """Negative inside a circle centered at the origin."""
    return np.sqrt(x[0] ** 2 + x[1] ** 2) - 0.5


def build_background_mesh(comm: MPI.Comm, n: int = 21) -> mesh.Mesh:
    return mesh.create_rectangle(
        comm=comm,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(n, n),
        cell_type=mesh.CellType.triangle,
    )


def build_level_set(msh: mesh.Mesh) -> fem.Function:
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi)
    phi.interpolate(circle_level_set)
    return phi


def global_count(comm: MPI.Comm, local_size: int) -> int:
    return comm.allreduce(local_size, op=MPI.SUM)


def _points_as_pyvista_coordinates(points: np.ndarray) -> np.ndarray:
    """Return point coordinates in the three-component layout PyVista expects."""
    point_coordinates = np.asarray(points).T
    if point_coordinates.shape[1] == 3:
        return np.ascontiguousarray(point_coordinates)
    if point_coordinates.shape[1] == 2:
        coordinates_3d = np.zeros(
            (point_coordinates.shape[0], 3), dtype=point_coordinates.dtype
        )
        coordinates_3d[:, :2] = point_coordinates
        return coordinates_3d
    raise ValueError("Can only plot quadrature points with geometry dimension 2 or 3.")


def _point_parent_index(rules) -> np.ndarray:
    """Return one parent cell index per quadrature point."""
    if rules.parent_map is None or rules.offsets is None:
        return np.empty(0, dtype=np.int32)
    counts = np.diff(rules.offsets).astype(np.int64, copy=False)
    return np.repeat(rules.parent_map, counts).astype(np.int32, copy=False)


def plot_cut_mesh(
    cut_mesh: cutfemx.mesh_cutter.CutMesh,
    phi_cut: fem.Function,
    quadrature_rules,
) -> None:
    """Show the cut mesh, the cut level set, and physical quadrature points."""
    try:
        import pyvista
        from dolfinx import plot
    except ImportError:
        if MPI.COMM_WORLD.rank == 0:
            print("pyvista is required for interactive plotting")
        return

    if cut_mesh.mesh is None:
        return

    plotter = pyvista.Plotter(shape=(1, 3))

    cells, types, x = plot.vtk_mesh(cut_mesh.mesh)
    mesh_grid = pyvista.UnstructuredGrid(cells, types, x)
    mesh_grid.cell_data["parent_index"] = np.asarray(cut_mesh.parent_index)
    mesh_grid.cell_data["is_cut_cell"] = np.asarray(cut_mesh.is_cut_cell)

    plotter.subplot(0, 0)
    plotter.add_text("cut mesh", font_size=12)
    plotter.add_mesh(
        mesh_grid,
        scalars="is_cut_cell",
        show_edges=True,
        cmap="coolwarm",
    )

    cells, types, x = plot.vtk_mesh(phi_cut.function_space)
    phi_grid = pyvista.UnstructuredGrid(cells, types, x)
    phi_values = np.asarray(phi_cut.x.array)
    if np.iscomplexobj(phi_values):
        phi_values = phi_values.real
    phi_grid.point_data["phi"] = phi_values[: phi_grid.n_points]

    plotter.subplot(0, 1)
    plotter.add_text("cut level set", font_size=12)
    plotter.add_mesh(
        phi_grid,
        scalars="phi",
        show_edges=True,
        cmap="viridis",
    )

    plotter.subplot(0, 2)
    plotter.add_text("runtime quadrature points", font_size=12)
    plotter.add_mesh(
        mesh_grid,
        style="wireframe",
        color="lightgray",
        line_width=1,
    )
    if quadrature_rules.physical_points is not None:
        quadrature_points = _points_as_pyvista_coordinates(
            quadrature_rules.physical_points
        )
        point_cloud = pyvista.PolyData(quadrature_points)
        point_cloud.point_data["weight"] = np.asarray(quadrature_rules.weights)
        point_cloud.point_data["parent_index"] = _point_parent_index(quadrature_rules)
        plotter.add_mesh(
            point_cloud,
            scalars="weight",
            cmap="magma",
            render_points_as_spheres=True,
            point_size=9,
        )

    plotter.link_views()
    plotter.show()


def main(show_plot: bool = True) -> None:
    comm = MPI.COMM_WORLD

    msh = build_background_mesh(comm)
    phi = build_level_set(msh)
    mesh_cutter = cutfemx.mesh_cutter(phi)

    inside_cells = mesh_cutter.locate_entities("phi < 0")
    cut_cells = mesh_cutter.locate_entities("phi = 0")
    outside_cells = mesh_cutter.locate_entities("phi > 0")

    # Subset support is part of the public MeshCutter API and will be reused by
    # runtime quadrature and cut-mesh creation.
    tdim = msh.topology.dim
    first_local_cells = np.arange(
        min(10, msh.topology.index_map(tdim).size_local), dtype=np.int32
    )
    cut_cells_in_subset = mesh_cutter.locate_entities(
        "phi = 0", entities=first_local_cells, entity_dim=tdim
    )

    cut_mesh = mesh_cutter.create_cut_mesh("phi < 0", mode="full")
    phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)
    cut_quadrature = mesh_cutter.runtime_quadrature("phi < 0", order=2)

    if comm.rank == 0:
        print("CutFEMx Poisson migration demo")
        print("stage: mesh_cutter.runtime_quadrature")

    counts = {
        "inside cells": inside_cells.size,
        "cut cells": cut_cells.size,
        "outside cells": outside_cells.size,
        "cut cells in first subset": cut_cells_in_subset.size,
    }
    for label, count in counts.items():
        total = global_count(comm, count)
        if comm.rank == 0:
            print(f"{label}: {total}")

    cut_mesh_cells = global_count(comm, cut_mesh.parent_index.size)
    cut_output_cells = global_count(comm, np.count_nonzero(cut_mesh.is_cut_cell))
    uncut_output_cells = global_count(
        comm, cut_mesh.is_cut_cell.size - np.count_nonzero(cut_mesh.is_cut_cell)
    )
    if comm.rank == 0:
        print(f"visualisation mesh cells: {cut_mesh_cells}")
        print(f"visualisation cut cells: {cut_output_cells}")
        print(f"visualisation uncut cells: {uncut_output_cells}")
        print(f"runtime quadrature rules: {cut_quadrature.parent_map.size}")
        print(f"runtime quadrature points: {cut_quadrature.weights.size}")

    if show_plot and comm.rank == 0:
        plot_cut_mesh(cut_mesh, phi_cut, cut_quadrature.with_physical_points())

    # Next migration slices will extend this same file:
    # 1. runintgen-backed UFL form compilation
    # 2. CutFEMx CutForm and assembler call
    # 3. full cut Poisson solve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the demo without opening the interactive PyVista window.",
    )
    args = parser.parse_args()
    main(show_plot=not args.no_plot)
