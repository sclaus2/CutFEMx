# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Linear elasticity on a cut plate with level-set holes.

The physical domain is a rectangular background mesh with circular holes
removed by one level set. The left edge is clamped and the right edge is pulled
in the x-direction. The von Mises stress is evaluated only after the
displacement has been restricted to the physical cut mesh, which avoids sampling
stresses in inactive or stabilization-only background cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np

import ufl
from dolfinx import default_scalar_type, fem, io, la, mesh


def _holes_level_set(
    holes: tuple[tuple[tuple[float, float], float], ...],
):
    """Return the union-of-holes level set, negative inside any circle."""

    def level_set(x: np.ndarray) -> np.ndarray:
        values = [
            np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius
            for center, radius in holes
        ]
        return np.minimum.reduce(values)

    return level_set


def _epsilon(u):
    return ufl.sym(ufl.grad(u))


def _sigma(u, mu: float, lmbda: float, gdim: int):
    eps = _epsilon(u)
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)


def _von_mises_plane_strain(u, mu: float, lmbda: float):
    stress = _sigma(u, mu, lmbda, 2)
    stress_zz = lmbda * ufl.tr(_epsilon(u))
    return ufl.sqrt(
        0.5
        * (
            (stress[0, 0] - stress[1, 1]) ** 2
            + (stress[1, 1] - stress_zz) ** 2
            + (stress_zz - stress[0, 0]) ** 2
        )
        + 3.0 * stress[0, 1] ** 2
    )


def _solve_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
    bcs: list[fem.DirichletBC],
) -> tuple[fem.Function, cutfemx.fem.ActiveDomain]:
    """Assemble, deactivate, and solve the serial MatrixCSR system."""
    from scipy.sparse.linalg import spsolve

    if V.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy solve path in this demo is serial-only.")

    A = cutfemx.fem.assemble_matrix(a, bcs=bcs)
    A.scatter_reverse()
    b = cutfemx.fem.assemble_vector(L)
    cutfemx.fem.apply_lifting(b.array, [a], [bcs])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, bcs)

    active = cutfemx.fem.active_domain(a)
    cutfemx.fem.deactivate_outside(A, b, active)
    zero_rows = cutfemx.fem.zero_rows(A)
    if zero_rows.size > 0:
        raise RuntimeError(f"Zero matrix rows remain after deactivation: {zero_rows.tolist()}")

    uh = fem.Function(V, name="u")
    uh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    uh.x.scatter_forward()
    return uh, active


def _function_cell_values(u: fem.Function) -> np.ndarray:
    """Return one scalar DG0 value per local cell."""
    msh = u.function_space.mesh
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    values = np.empty(num_cells, dtype=u.x.array.dtype)
    for cell in range(num_cells):
        dofs = u.function_space.dofmap.cell_dofs(cell)
        values[cell] = u.x.array[dofs[0]]
    return values


def _compute_von_mises_on_cut_mesh(
    u_cut: fem.Function,
    *,
    mu: float,
    lmbda: float,
) -> fem.Function:
    """Interpolate von Mises stress on the physical cut mesh only."""
    cut_msh = u_cut.function_space.mesh
    Q = fem.functionspace(cut_msh, ("Discontinuous Lagrange", 0))
    von_mises = fem.Function(Q, name="von_mises")
    interpolation_points = Q.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    expr = fem.Expression(
        _von_mises_plane_strain(u_cut, mu, lmbda),
        interpolation_points,
    )
    von_mises.interpolate(expr)
    return von_mises


def _warp_factor(points: np.ndarray, displacement: np.ndarray) -> float:
    max_displacement = float(np.max(np.linalg.norm(displacement[:, :2], axis=1)))
    if not np.isfinite(max_displacement) or max_displacement <= np.finfo(float).eps:
        return 1.0

    extents = np.ptp(points[:, :2], axis=0)
    width = float(np.max(extents)) if extents.size > 0 else 1.0
    if not np.isfinite(width) or width <= np.finfo(float).eps:
        width = 1.0
    return 0.18 * width / max_displacement


def _grid_from_cut_fields(u_cut: fem.Function, von_mises: fem.Function):
    """Build a PyVista grid with vector displacement and cellwise stress."""
    import pyvista

    from dolfinx import plot

    cells, types, x = plot.vtk_mesh(u_cut.function_space)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    gdim = u_cut.function_space.mesh.geometry.dim

    values = np.asarray(u_cut.x.array)
    if np.iscomplexobj(values):
        values = values.real
    displacement = np.zeros((grid.n_points, 3), dtype=values.dtype)
    displacement[:, :gdim] = values.reshape((-1, gdim))[: grid.n_points]
    grid.point_data["displacement"] = displacement
    grid.point_data["|u|"] = np.linalg.norm(displacement[:, :gdim], axis=1)

    stress_values = _function_cell_values(von_mises)
    if np.iscomplexobj(stress_values):
        stress_values = stress_values.real
    if stress_values.size < grid.n_cells:
        raise RuntimeError("von Mises field has fewer cell values than the PyVista grid.")
    grid.cell_data["von_mises"] = stress_values[: grid.n_cells]
    return grid


def _plot_solution(
    msh: mesh.Mesh,
    cut_mesh: cutfemx.CutMesh,
    u_cut: fem.Function,
    von_mises: fem.Function,
    *,
    screenshot: str | None,
    show: bool,
) -> None:
    try:
        import pyvista

        from dolfinx import plot
    except ModuleNotFoundError:
        if msh.comm.rank == 0:
            print("pyvista and dolfinx.plot are required for plotting")
        return

    if msh.comm.rank != 0 or cut_mesh.mesh is None:
        return

    grid = _grid_from_cut_fields(u_cut, von_mises)
    warped = grid.warp_by_vector(
        "displacement",
        factor=_warp_factor(grid.points, grid["displacement"]),
    )
    background = pyvista.UnstructuredGrid(*plot.vtk_mesh(msh))

    plotter = pyvista.Plotter(shape=(1, 2), off_screen=screenshot is not None and not show)
    plotter.subplot(0, 0)
    plotter.add_title("Deformation", font_size=12)
    plotter.add_mesh(
        background,
        style="wireframe",
        color="lightgray",
        opacity=0.10,
        line_width=1,
    )
    plotter.add_mesh(
        warped,
        scalars="|u|",
        cmap="viridis",
        show_edges=True,
        scalar_bar_args={"title": "|u|"},
    )
    plotter.view_xy()

    plotter.subplot(0, 1)
    plotter.add_title("von Mises stress", font_size=12)
    plotter.add_mesh(
        warped,
        scalars="von_mises",
        cmap="magma",
        show_edges=True,
        scalar_bar_args={"title": "sigma_vm"},
    )
    plotter.view_xy()
    plotter.link_views()

    if screenshot is not None:
        plotter.screenshot(screenshot)
        print(f"Wrote screenshot to {screenshot}")
    if show:
        plotter.show()
    else:
        plotter.close()


def _write_xdmf(
    cut_mesh: cutfemx.CutMesh,
    u_cut: fem.Function,
    von_mises: fem.Function,
    prefix: str | Path,
) -> None:
    if cut_mesh.mesh is None:
        return

    prefix_path = Path(prefix)
    path = prefix_path.with_suffix(".xdmf")
    with io.XDMFFile(cut_mesh.mesh.comm, path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(u_cut)
        xdmf.write_function(von_mises)
    if cut_mesh.mesh.comm.rank == 0:
        print(f"Wrote XDMF file to {path}")


def main(
    *,
    n: int = 32,
    order: int = 4,
    young_modulus: float = 1.0e3,
    poisson_ratio: float = 0.3,
    pull: float = 0.12,
    gamma_ghost: float = 0.05,
    plot: bool = True,
    screenshot: str | None = None,
    xdmf_prefix: str | None = None,
) -> None:
    comm = MPI.COMM_WORLD
    if n < 8:
        raise ValueError("--n must be at least 8.")
    if order < 1:
        raise ValueError("--order must be at least 1.")
    if young_modulus <= 0.0:
        raise ValueError("--young-modulus must be positive.")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("--poisson-ratio must be in (-1, 0.5).")

    msh = mesh.create_rectangle(
        comm,
        ((-1.5, -0.6), (1.5, 0.6)),
        (3 * n, n),
        cell_type=mesh.CellType.triangle,
    )

    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    holes = (
        ((-0.62, 0.0), 0.18),
        ((0.0, 0.18), 0.16),
        ((0.53, -0.12), 0.15),
    )
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(_holes_level_set(holes))

    cut_data = cutfemx.cut(phi)
    selector = "phi>0"
    solid_cells = cutfemx.locate_entities(cut_data, selector)
    solid_rules = cutfemx.runtime_quadrature(cut_data, selector, order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, selector)

    dx_solid = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[solid_cells, solid_rules],
    )
    dS_ghost = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=1,
        subdomain_data=ghost_facets,
    )

    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    gdim = msh.geometry.dim
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )

    a = ufl.inner(_sigma(u, mu, lmbda, gdim), _epsilon(v)) * dx_solid
    if ghost_facets.size > 0 and gamma_ghost != 0.0:
        n_facet = ufl.FacetNormal(msh)
        h_avg = ufl.avg(ufl.CellDiameter(msh))
        a += (
            gamma_ghost
            * mu
            * h_avg
            * ufl.inner(
                ufl.jump(ufl.grad(u), n_facet),
                ufl.jump(ufl.grad(v), n_facet),
            )
            * dS_ghost
        )

    zero_force = fem.Constant(msh, default_scalar_type((0.0, 0.0)))
    L = ufl.inner(zero_force, v) * dx_solid

    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], -1.5)
    )
    right_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.5)
    )
    u_left = fem.Function(V)
    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    bcs = [fem.dirichletbc(u_left, left_dofs)]

    Vx, _ = V.sub(0).collapse()
    u_right_x = fem.Function(Vx)
    u_right_x.interpolate(lambda x: pull * np.ones_like(x[0]))
    right_x_dofs = fem.locate_dofs_topological((V.sub(0), Vx), fdim, right_facets)
    bcs.append(fem.dirichletbc(u_right_x, right_x_dofs, V.sub(0)))

    a_form = cutfemx.fem.form(a)
    L_form = cutfemx.fem.form(L)
    uh, active = _solve_scipy(a_form, L_form, V, bcs)

    cut_mesh = cutfemx.create_cut_mesh(cut_data, selector, mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("The selected cut domain is empty.")
    u_cut = cutfemx.fem.cut_function(uh, cut_mesh)
    von_mises = _compute_von_mises_on_cut_mesh(u_cut, mu=mu, lmbda=lmbda)

    displacement_norm = float(np.max(np.linalg.norm(uh.x.array.reshape((-1, gdim)), axis=1)))
    max_stress = float(np.max(von_mises.x.array)) if von_mises.x.array.size else 0.0
    if comm.rank == 0:
        print("CutFEMx elasticity demo")
        print(f"  cells: {3 * n} x {n}")
        print("  level set: phi = min(phi_0, phi_1, phi_2)")
        print(f"  solid standard cells: {solid_cells.size}")
        print(f"  cut cells: {solid_rules.parent_map.size}")
        print(f"  ghost facets: {ghost_facets.size}")
        print(f"  inactive dofs: {active.inactive_dofs.size}")
        print(f"  max |u| on background dofs: {displacement_norm:.6e}")
        print(f"  max von Mises on cut mesh: {max_stress:.6e}")

    if plot or screenshot is not None:
        _plot_solution(
            msh,
            cut_mesh,
            u_cut,
            von_mises,
            screenshot=screenshot,
            show=plot,
        )

    if xdmf_prefix is not None:
        _write_xdmf(cut_mesh, u_cut, von_mises, xdmf_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve linear elasticity on a cut plate with level-set holes."
    )
    parser.add_argument("--n", type=int, default=32, help="Vertical background cells.")
    parser.add_argument("--order", type=int, default=4, help="Runtime quadrature order.")
    parser.add_argument("--young-modulus", type=float, default=1.0e3)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--pull", type=float, default=0.12, help="Right-edge x displacement.")
    parser.add_argument("--gamma-ghost", type=float, default=0.05)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run without opening an interactive PyVista window.",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Write a PyVista screenshot to this path.",
    )
    parser.add_argument(
        "--xdmf-prefix",
        type=str,
        default=None,
        help="Write cut displacement and von Mises fields to '<prefix>.xdmf'.",
    )
    args = parser.parse_args()
    main(
        n=args.n,
        order=args.order,
        young_modulus=args.young_modulus,
        poisson_ratio=args.poisson_ratio,
        pull=args.pull,
        gamma_ghost=args.gamma_ghost,
        plot=not args.no_plot,
        screenshot=args.screenshot,
        xdmf_prefix=args.xdmf_prefix,
    )
