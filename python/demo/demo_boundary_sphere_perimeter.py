# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Runtime quadrature on cut exterior facets.

The sphere centre is outside the unit box, so the zero level set cuts the box
boundary in a circle. The demo cuts the exterior facets once, then uses the
returned CutData for classification, cut meshes, runtime quadrature, assembly,
and plotting.
"""

from __future__ import annotations

import argparse

import numpy as np
import ufl
from mpi4py import MPI

import cutfemx
from dolfinx import fem, mesh
from runintgen import dsq


def _assemble_scalar(form) -> float:
    value = cutfemx.fem.assemble_scalar(cutfemx.fem.form(form))
    return float(MPI.COMM_WORLD.allreduce(value, op=MPI.SUM))


def _boundary_circle_radius(centre: np.ndarray, sphere_radius: float) -> float:
    distance = abs(centre[0])
    if distance >= sphere_radius:
        return 0.0
    return float(np.sqrt(sphere_radius**2 - distance**2))


def _validate_left_face_cut(centre: np.ndarray, radius: float) -> None:
    other_face_distance = min(
        1.0 - centre[0],
        centre[1],
        1.0 - centre[1],
        centre[2],
        1.0 - centre[2],
    )
    if not (centre[0] < 0.0 and -centre[0] < radius < other_face_distance):
        raise ValueError(
            "This demo expects centre[0] < 0 and a sphere cutting only the "
            "left box face."
        )


def _grid(cut_mesh: cutfemx.CutMesh):
    import pyvista
    from dolfinx import plot

    if cut_mesh.mesh is None:
        return pyvista.UnstructuredGrid()
    return pyvista.UnstructuredGrid(*plot.vtk_mesh(cut_mesh.mesh))


def _points(rules):
    import pyvista

    return pyvista.PolyData(rules.with_physical_points().physical_points.T)


def _plot(
    msh,
    curve_mesh: cutfemx.CutMesh,
    patch_mesh: cutfemx.CutMesh,
    curve_rules,
    patch_rules,
    screenshot: str | None,
    show: bool,
) -> None:
    try:
        import pyvista
        from dolfinx import plot
    except ModuleNotFoundError:
        if MPI.COMM_WORLD.rank == 0:
            print("pyvista and dolfinx.plot are required for --plot/--screenshot")
        return

    if MPI.COMM_WORLD.rank != 0:
        return

    plotter = pyvista.Plotter(off_screen=screenshot is not None and not show)
    plotter.add_mesh(
        pyvista.UnstructuredGrid(*plot.vtk_mesh(msh)),
        style="wireframe",
        color="lightgray",
        line_width=1,
        opacity=0.06,
    )
    plotter.add_mesh(_grid(patch_mesh), color="#2ca25f", opacity=0.65,
                     show_edges=True, label="phi<0")
    plotter.add_mesh(_grid(curve_mesh), color="#d7191c", line_width=5,
                     label="phi=0")
    plotter.add_points(_points(patch_rules), color="black", point_size=6,
                       render_points_as_spheres=True, label="area pts")
    plotter.add_points(_points(curve_rules), color="#2c7bb6", point_size=10,
                       render_points_as_spheres=True, label="curve pts")
    plotter.add_title("Sphere cut on box boundary", font_size=13)
    plotter.add_legend(size=(0.20, 0.17), loc="upper right",
                       background_opacity=0.0)
    plotter.view_isometric()
    plotter.add_axes()

    if screenshot is not None:
        plotter.screenshot(screenshot)
        print(f"Wrote screenshot to {screenshot}")
    if show:
        plotter.show()
    else:
        plotter.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a sphere cut through the exterior boundary."
    )
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.34)
    parser.add_argument("--center", type=float, nargs=3,
                        default=(-0.11, 0.5, 0.5))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--screenshot", type=str, default=None)
    args = parser.parse_args()

    if args.n < 2:
        raise ValueError("--n must be at least 2.")
    if args.radius <= 0.0:
        raise ValueError("--radius must be positive.")

    centre = np.asarray(args.center, dtype=np.float64)
    _validate_left_face_cut(centre, args.radius)

    comm = MPI.COMM_WORLD
    msh = mesh.create_box(
        comm,
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (args.n, args.n, args.n),
        cell_type=mesh.CellType.tetrahedron,
    )
    fdim = msh.topology.dim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, msh.topology.dim)

    V = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V, name="phi")
    phi.interpolate(
        lambda x: (x[0] - centre[0]) ** 2
        + (x[1] - centre[1]) ** 2
        + (x[2] - centre[2]) ** 2
        - args.radius**2
    )

    exterior_facets = mesh.exterior_facet_indices(msh.topology)
    cut_data = cutfemx.cut(phi, exterior_facets, fdim)

    negative_facets = cutfemx.locate_entities(cut_data, "phi<0")
    cut_facets = cutfemx.locate_entities(cut_data, "phi=0")
    curve_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", args.order)
    patch_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", args.order)
    curve_mesh = cutfemx.create_cut_mesh(cut_data, "phi=0", mode="cut_only")
    patch_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="cut_only")

    one = fem.Constant(msh, np.float64(1.0))
    standard_tags = mesh.meshtags(
        msh, fdim, negative_facets, np.full(negative_facets.size, 1, dtype=np.int32)
    )
    standard_area = _assemble_scalar(
        one * ufl.Measure("ds", domain=msh, subdomain_data=standard_tags)(1)
    )
    cut_patch_area = _assemble_scalar(
        one * dsq(domain=msh, quadrature_provider=patch_rules)
    )
    perimeter = _assemble_scalar(
        one * dsq(domain=msh, quadrature_provider=curve_rules)
    )

    circle_radius = _boundary_circle_radius(centre, args.radius)
    exact_perimeter = 2.0 * np.pi * circle_radius
    exact_area = np.pi * circle_radius**2

    if comm.rank == 0:
        print("CutFEMx sphere/box-boundary cut demo")
        print(f"  exterior facets: {exterior_facets.size}")
        print(f"  phi<0 full exterior facets: {negative_facets.size}")
        print(f"  cut exterior facets: {cut_facets.size}")
        print(f"  phi=0 runtime perimeter: {perimeter:.12f}")
        print(f"  exact perimeter: {exact_perimeter:.12f}")
        print(f"  phi<0 runtime area on cut facets: {cut_patch_area:.12f}")
        print(f"  phi<0 standard + runtime area: {standard_area + cut_patch_area:.12f}")
        print(f"  exact disk area: {exact_area:.12f}")

    if args.plot or args.screenshot is not None:
        _plot(
            msh,
            curve_mesh,
            patch_mesh,
            curve_rules,
            patch_rules,
            args.screenshot,
            args.plot,
        )


if __name__ == "__main__":
    main()
