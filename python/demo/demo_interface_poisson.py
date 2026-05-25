# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

# # Two-domain unfitted Poisson interface problem
#
# This demo solves a two-phase Poisson problem on a circular unfitted
# interface. The inside and outside fields are represented by two scalar
# background spaces, coupled through symmetric Nitsche terms on the interface,
# and assembled as a 2x2 block system.

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, io, la, mesh
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve


def _points_as_pyvista_coordinates(points: np.ndarray) -> np.ndarray:
    """Return point coordinates in the three-component layout PyVista expects."""
    point_coordinates = np.asarray(points).T
    if point_coordinates.shape[1] == 3:
        return np.ascontiguousarray(point_coordinates)
    if point_coordinates.shape[1] == 2:
        coordinates_3d = np.zeros((point_coordinates.shape[0], 3), dtype=point_coordinates.dtype)
        coordinates_3d[:, :2] = point_coordinates
        return coordinates_3d
    raise ValueError("Can only plot points with geometry dimension 2 or 3.")


def _grid_from_function(u: fem.Function):
    """Build a PyVista grid carrying a scalar finite element function."""
    import pyvista
    from dolfinx import plot

    cells, types, x = plot.vtk_mesh(u.function_space)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    values = np.asarray(u.x.array)
    if np.iscomplexobj(values):
        values = values.real
    if values.size < grid.n_points:
        raise RuntimeError("Function has fewer values than PyVista points.")
    grid.point_data["u"] = values[: grid.n_points]
    grid.set_active_scalars("u")
    return grid


def _cut_solution_functions(cut_data, u1_h: fem.Function, u2_h: fem.Function):
    """Restrict both background fields to their physical cut domains."""
    inside_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
    outside_mesh = cutfemx.create_cut_mesh(cut_data, "phi>0", mode="full")
    if inside_mesh.mesh is None or outside_mesh.mesh is None:
        raise RuntimeError("Cannot export an empty inside or outside cut mesh.")
    return (
        inside_mesh,
        outside_mesh,
        cutfemx.fem.cut_function(u1_h, inside_mesh),
        cutfemx.fem.cut_function(u2_h, outside_mesh),
    )


def _prefixed_path(prefix: str | Path, suffix: str) -> Path:
    prefix_path = Path(prefix)
    return prefix_path.with_name(f"{prefix_path.name}_{suffix}.xdmf")


def _write_solution_xdmf(cut_data, u1_h: fem.Function, u2_h: fem.Function, prefix: str) -> None:
    """Write the two physical-domain solutions to XDMF files."""
    inside_mesh, outside_mesh, u1_cut, u2_cut = _cut_solution_functions(cut_data, u1_h, u2_h)
    u1_path = _prefixed_path(prefix, "u1")
    u2_path = _prefixed_path(prefix, "u2")

    with io.XDMFFile(u1_cut.function_space.mesh.comm, u1_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(inside_mesh.mesh)
        xdmf.write_function(u1_cut)
    with io.XDMFFile(u2_cut.function_space.mesh.comm, u2_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(outside_mesh.mesh)
        xdmf.write_function(u2_cut)

    if u1_h.function_space.mesh.comm.rank == 0:
        print(f"Wrote XDMF files to {u1_path} and {u2_path}")


def _plot_solution(
    msh: mesh.Mesh,
    cut_data,
    interface_rules,
    u1_h: fem.Function,
    u2_h: fem.Function,
    *,
    screenshot: str | None,
    show: bool,
) -> None:
    """Plot the two-domain solution on its active cut meshes."""
    try:
        import pyvista
        from dolfinx import plot
    except ModuleNotFoundError:
        if msh.comm.rank == 0:
            print("pyvista and dolfinx.plot are required for --plot/--screenshot")
        return

    if msh.comm.rank != 0:
        return

    _, _, u1_cut, u2_cut = _cut_solution_functions(cut_data, u1_h, u2_h)
    inside_grid = _grid_from_function(u1_cut)
    outside_grid = _grid_from_function(u2_cut)
    background_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(msh))

    values = np.concatenate(
        [
            np.asarray(inside_grid.point_data["u"]),
            np.asarray(outside_grid.point_data["u"]),
        ]
    )
    clim = (float(np.min(values)), float(np.max(values)))

    plotter = pyvista.Plotter(off_screen=screenshot is not None and not show)
    plotter.add_mesh(
        background_grid,
        style="wireframe",
        color="lightgray",
        opacity=0.10,
        line_width=1,
    )
    plotter.add_mesh(
        outside_grid,
        scalars="u",
        cmap="viridis",
        clim=clim,
        show_edges=True,
        opacity=0.88,
        scalar_bar_args={"title": "u"},
    )
    plotter.add_mesh(
        inside_grid,
        scalars="u",
        cmap="viridis",
        clim=clim,
        show_edges=True,
        opacity=0.95,
        scalar_bar_args={"title": "u"},
    )

    points = _points_as_pyvista_coordinates(interface_rules.with_physical_points().physical_points)
    if points.size > 0:
        plotter.add_points(
            pyvista.PolyData(points),
            color="black",
            point_size=8,
            render_points_as_spheres=True,
        )

    plotter.add_title("Two-domain interface Poisson solution", font_size=13)
    plotter.view_xy()
    plotter.add_axes()

    if screenshot is not None:
        plotter.screenshot(screenshot)
        print(f"Wrote screenshot to {screenshot}")
    if show:
        plotter.show()
    else:
        plotter.close()


def _assemble_matrix_block(form) -> la.MatrixCSR:
    """Assemble one CutFEMx bilinear block."""
    A = cutfemx.fem.assemble_matrix(form)
    A.scatter_reverse()
    return A


def _assemble_vector_block(form) -> la.Vector:
    """Assemble one CutFEMx linear block."""
    b = cutfemx.fem.assemble_vector(form)
    b.scatter_reverse(la.InsertMode.add)
    return b


def solve_block_system(
    a_blocks,
    L_blocks,
    domain1: cutfemx.fem.ActiveDomain,
    domain2: cutfemx.fem.ActiveDomain,
    V1: fem.FunctionSpace,
    V2: fem.FunctionSpace,
) -> tuple[fem.Function, fem.Function]:
    """Assemble, deactivate, and solve the serial 2x2 block system."""
    if V1.mesh.comm.size != 1:
        raise RuntimeError("The SciPy block solve path in this demo is serial-only.")

    A_matrix_blocks = [[_assemble_matrix_block(a_blocks[i][j]) for j in range(2)] for i in range(2)]
    b_vector_blocks = [_assemble_vector_block(L_blocks[i]) for i in range(2)]
    cutfemx.fem.deactivate_outside_blocks(A_matrix_blocks, [domain1, domain2], b_vector_blocks)
    zero_rows = cutfemx.fem.zero_block_rows(A_matrix_blocks)
    if any(rows.size > 0 for rows in zero_rows):
        detail = ", ".join(f"block row {i}: {rows.tolist()}" for i, rows in enumerate(zero_rows))
        raise RuntimeError(f"Zero matrix rows remain after deactivation ({detail}).")

    A = bmat(
        [[A.to_scipy().tocsr() for A in row] for row in A_matrix_blocks],
        format="csr",
    )
    b = np.concatenate([block.array.copy() for block in b_vector_blocks])
    n1 = b_vector_blocks[0].array.size

    solution = spsolve(A, b)

    u1_h = fem.Function(V1, name="u1_h")
    u2_h = fem.Function(V2, name="u2_h")
    u1_h.x.array[:] = solution[:n1]
    u2_h.x.array[:] = solution[n1:]
    u1_h.x.scatter_forward()
    u2_h.x.scatter_forward()
    return u1_h, u2_h


def main(
    *,
    n: int = 24,
    order: int = 4,
    radius: float = 0.53,
    center_x: float = 0.05,
    center_y: float = -0.03,
    kappa_1: float = 1.0,
    kappa_2: float = 8.0,
    gamma_interface: float = 40.0,
    gamma_boundary: float = 40.0,
    gamma_ghost: float = 0.1,
    plot: bool = False,
    screenshot: str | None = None,
    xdmf_prefix: str | None = None,
) -> None:
    comm = MPI.COMM_WORLD
    if n < 4:
        raise ValueError("--n must be at least 4.")
    if order < 1:
        raise ValueError("--order must be at least 1.")
    if radius <= 0.0:
        raise ValueError("--radius must be positive.")
    if kappa_1 <= 0.0 or kappa_2 <= 0.0:
        raise ValueError("Diffusivities must be positive.")

    msh = mesh.create_rectangle(
        comm,
        ((-1.0, -1.0), (1.0, 1.0)),
        (n, n),
        cell_type=mesh.CellType.triangle,
    )
    fdim = msh.topology.dim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    exterior_facets = mesh.exterior_facet_indices(msh.topology)
    center = (center_x, center_y)

    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(lambda x: np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius)

    cut_data = cutfemx.cut(phi)
    inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
    outside_cells = cutfemx.locate_entities(cut_data, "phi>0")
    inside_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    outside_rules = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
    ghost_facets_1 = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
    ghost_facets_2 = cutfemx.ghost_penalty_facets(cut_data, "phi>0")

    dx1 = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=[inside_cells, inside_rules],
    )
    dx2 = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=2,
        subdomain_data=[outside_cells, outside_rules],
    )
    dgamma = ufl.Measure("dx", domain=msh, subdomain_id=3, subdomain_data=interface_rules)
    dS_ghost_1 = ufl.Measure("dS", domain=msh, subdomain_id=4, subdomain_data=ghost_facets_1)
    dS_ghost_2 = ufl.Measure("dS", domain=msh, subdomain_id=5, subdomain_data=ghost_facets_2)
    ds_outer = ufl.Measure("ds", domain=msh, subdomain_id=6, subdomain_data=exterior_facets)

    V1 = fem.functionspace(msh, ("Lagrange", 1))
    V2 = fem.functionspace(msh, ("Lagrange", 1))
    W_block = ufl.MixedFunctionSpace(V1, V2)
    u1, u2 = ufl.TrialFunctions(W_block)
    v1, v2 = ufl.TestFunctions(W_block)

    x = ufl.SpatialCoordinate(msh)
    r2 = (x[0] - center_x) ** 2 + (x[1] - center_y) ** 2
    ratio = kappa_1 / kappa_2
    u1_exact = r2
    u2_exact = ratio * r2 + radius**2 * (1.0 - ratio)
    f1 = fem.Constant(msh, default_scalar_type(-4.0 * kappa_1))
    f2 = fem.Constant(msh, default_scalar_type(-4.0 * kappa_1))

    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)
    kappa_h = 2.0 * kappa_1 * kappa_2 / (kappa_1 + kappa_2)
    eta_interface = gamma_interface * kappa_h / h
    eta_boundary = gamma_boundary * kappa_2 / h
    w1 = kappa_2 / (kappa_1 + kappa_2)
    w2 = kappa_1 / (kappa_1 + kappa_2)

    def jump_pair(side1, side2):
        return side1 - side2

    def avg_flux_pair(side1, side2):
        return w1 * kappa_1 * ufl.dot(ufl.grad(side1), n_gamma) + w2 * kappa_2 * ufl.dot(
            ufl.grad(side2), n_gamma
        )

    jump_u = jump_pair(u1, u2)
    jump_v = jump_pair(v1, v2)
    flux_u = avg_flux_pair(u1, u2)
    flux_v = avg_flux_pair(v1, v2)

    a = kappa_1 * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1
    a += kappa_2 * ufl.inner(ufl.grad(u2), ufl.grad(v2)) * dx2
    a += (-flux_u * jump_v - flux_v * jump_u + eta_interface * jump_u * jump_v) * dgamma

    if ghost_facets_1.size > 0 and gamma_ghost != 0.0:
        a += (
            gamma_ghost
            * kappa_1
            * h_avg
            * ufl.inner(
                ufl.jump(ufl.grad(u1), n_facet),
                ufl.jump(ufl.grad(v1), n_facet),
            )
            * dS_ghost_1
        )
    if ghost_facets_2.size > 0 and gamma_ghost != 0.0:
        a += (
            gamma_ghost
            * kappa_2
            * h_avg
            * ufl.inner(
                ufl.jump(ufl.grad(u2), n_facet),
                ufl.jump(ufl.grad(v2), n_facet),
            )
            * dS_ghost_2
        )

    # Strong Dirichlet conditions are awkward for this first MatrixCSR block
    # prototype, so impose the fitted outer boundary data for u2 by Nitsche.
    a += (
        -kappa_2 * ufl.dot(ufl.grad(u2), n_facet) * v2
        - kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2
        + eta_boundary * u2 * v2
    ) * ds_outer

    L = f1 * v1 * dx1 + f2 * v2 * dx2
    L += (
        -kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2_exact + eta_boundary * u2_exact * v2
    ) * ds_outer

    a_blocks = ufl.extract_blocks(a)
    L_blocks = ufl.extract_blocks(L)
    a_forms = tuple(tuple(cutfemx.fem.form(a_blocks[i][j]) for j in range(2)) for i in range(2))
    L_forms = tuple(cutfemx.fem.form(L_blocks[i]) for i in range(2))

    domain1 = cutfemx.fem.active_domain(a_forms[0][0])
    domain2 = cutfemx.fem.active_domain(a_forms[1][1])
    u1_h, u2_h = solve_block_system(a_forms, L_forms, domain1, domain2, V1, V2)

    e1 = cutfemx.fem.assemble_scalar(cutfemx.fem.form((u1_h - u1_exact) ** 2 * dx1))
    e2 = cutfemx.fem.assemble_scalar(cutfemx.fem.form((u2_h - u2_exact) ** 2 * dx2))
    jump_error = cutfemx.fem.assemble_scalar(cutfemx.fem.form((u1_h - u2_h) ** 2 * dgamma))
    e1 = comm.allreduce(e1, op=MPI.SUM)
    e2 = comm.allreduce(e2, op=MPI.SUM)
    jump_error = comm.allreduce(jump_error, op=MPI.SUM)

    if comm.rank == 0:
        print("CutFEMx two-domain interface Poisson demo")
        print(f"  cells per side: {n}")
        print(f"  inside standard cells: {inside_cells.size}")
        print(f"  outside standard cells: {outside_cells.size}")
        print(f"  interface cut cells: {interface_rules.parent_map.size}")
        print(f"  ghost facets inside/outside: {ghost_facets_1.size}/{ghost_facets_2.size}")
        print(f"  inactive dofs u1/u2: {domain1.inactive_dofs.size}/{domain2.inactive_dofs.size}")
        print(f"  L2 error u1 on Omega_1: {np.sqrt(max(e1, 0.0)):.6e}")
        print(f"  L2 error u2 on Omega_2: {np.sqrt(max(e2, 0.0)):.6e}")
        print(f"  interface jump norm: {np.sqrt(max(jump_error, 0.0)):.6e}")

    if plot or screenshot is not None:
        _plot_solution(
            msh,
            cut_data,
            interface_rules,
            u1_h,
            u2_h,
            screenshot=screenshot,
            show=plot,
        )

    if xdmf_prefix is not None:
        _write_solution_xdmf(cut_data, u1_h, u2_h, xdmf_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve a two-domain unfitted Poisson interface problem."
    )
    parser.add_argument("--n", type=int, default=24, help="Background cells per side.")
    parser.add_argument("--order", type=int, default=4, help="Runtime quadrature order.")
    parser.add_argument("--radius", type=float, default=0.53)
    parser.add_argument("--center-x", type=float, default=0.05)
    parser.add_argument("--center-y", type=float, default=-0.03)
    parser.add_argument("--kappa-1", type=float, default=1.0)
    parser.add_argument("--kappa-2", type=float, default=8.0)
    parser.add_argument("--gamma-interface", type=float, default=40.0)
    parser.add_argument("--gamma-boundary", type=float, default=40.0)
    parser.add_argument("--gamma-ghost", type=float, default=0.1)
    parser.add_argument("--plot", action="store_true", help="Show a PyVista plot.")
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Write a PyVista screenshot to this path.",
    )
    parser.add_argument(
        "--xdmf-prefix",
        type=str,
        default="interface_poisson",
        help="Write physical-domain solution files '<prefix>_u1.xdmf' and '<prefix>_u2.xdmf'.",
    )
    args = parser.parse_args()
    main(
        n=args.n,
        order=args.order,
        radius=args.radius,
        center_x=args.center_x,
        center_y=args.center_y,
        kappa_1=args.kappa_1,
        kappa_2=args.kappa_2,
        gamma_interface=args.gamma_interface,
        gamma_boundary=args.gamma_boundary,
        gamma_ghost=args.gamma_ghost,
        plot=args.plot,
        screenshot=args.screenshot,
        xdmf_prefix=args.xdmf_prefix,
    )
