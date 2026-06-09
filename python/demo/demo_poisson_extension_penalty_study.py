# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Compare cut Poisson stabilization by conditioning and convergence.

The study solves a manufactured Poisson problem on a circular cut domain with
continuous P1 elements and symmetric Nitsche boundary conditions. It compares:

* no small-cut stabilization,
* a classical face ghost penalty,
* the Badia-style aggregation extension penalty over full bad cells.

The script is serial by construction because it computes dense active-system
condition numbers for diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from mpi4py import MPI

import cutfemx
import numpy as np
import scipy.sparse as sp

import ufl
from dolfinx import fem, la, mesh


def _circle_level_set(radius: float, center: tuple[float, float]):
    def level_set(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius

    return level_set


def _cell_dofs(V: fem.FunctionSpace, cells: np.ndarray) -> np.ndarray:
    dofs: list[int] = []
    for cell in np.asarray(cells, dtype=np.int32):
        dofs.extend(int(dof) for dof in V.dofmap.cell_dofs(int(cell)))
    return np.asarray(sorted(set(dofs)), dtype=np.int32)


def _active_condition_number(A: sp.spmatrix, active_dofs: np.ndarray) -> tuple[float, float, float]:
    dense = A[active_dofs, :][:, active_dofs].toarray()
    dense = 0.5 * (dense + dense.T)
    eig = np.linalg.eigvalsh(dense)
    abs_eig = np.abs(eig)
    positive = abs_eig[abs_eig > 1.0e-14 * max(1.0, float(abs_eig.max(initial=0.0)))]
    if positive.size == 0:
        return float("inf"), float(np.min(eig)), float(np.max(eig))
    return (
        float(positive.max() / positive.min()),
        float(np.min(eig)),
        float(np.max(eig)),
    )


def _solve_active_system(
    A: sp.spmatrix,
    b: np.ndarray,
    V: fem.FunctionSpace,
    active_dofs: np.ndarray,
) -> fem.Function:
    dense = A[active_dofs, :][:, active_dofs].toarray()
    rhs = b[active_dofs]
    uh = fem.Function(V, name="u_h")
    try:
        active_values = np.linalg.solve(dense, rhs)
    except np.linalg.LinAlgError:
        active_values = np.linalg.lstsq(dense, rhs, rcond=None)[0]
    uh.x.array[:] = 0.0
    uh.x.array[active_dofs] = active_values
    uh.x.scatter_forward()
    return uh


def _solution_warp_factor(points: np.ndarray, values: np.ndarray) -> float:
    value_range = float(np.ptp(values)) if values.size > 0 else 0.0
    if not np.isfinite(value_range) or value_range <= np.finfo(float).eps:
        return 1.0
    extent = np.ptp(points[:, :2], axis=0)
    width = float(np.max(extent)) if extent.size else 1.0
    if not np.isfinite(width) or width <= np.finfo(float).eps:
        width = 1.0
    return 0.18 * width / value_range


def _plot_final_solutions(
    records: list[dict[str, object]],
    *,
    screenshot: str | None,
    show: bool,
) -> None:
    try:
        import pyvista
        from dolfinx import plot
    except ModuleNotFoundError:
        print("pyvista and dolfinx.plot are required for solution plotting")
        return

    if not records:
        return

    prepared = []
    u_min = np.inf
    u_max = -np.inf
    for record in records:
        uh = record["solution"]
        V = uh.function_space
        cells, types, points = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, points)
        values = np.asarray(uh.x.array, dtype=float)
        grid.point_data["u"] = values[: grid.n_points]
        active_grid = grid.extract_cells(np.asarray(record["active_cells"], dtype=np.int32))
        active_values = np.asarray(active_grid.point_data["u"], dtype=float)
        u_min = min(u_min, float(np.min(active_values)))
        u_max = max(u_max, float(np.max(active_values)))
        prepared.append((record["mode"], active_grid))

    clim = (u_min, u_max)
    plotter = pyvista.Plotter(
        shape=(1, len(prepared)),
        off_screen=screenshot is not None and not show,
        window_size=(520 * len(prepared), 430),
    )
    for i, (mode, grid) in enumerate(prepared):
        plotter.subplot(0, i)
        plotter.add_text(str(mode), position="upper_left", font_size=11)
        warped = grid.warp_by_scalar(
            "u",
            factor=_solution_warp_factor(grid.points, np.asarray(grid.point_data["u"])),
        )
        plotter.add_mesh(
            warped,
            scalars="u",
            clim=clim,
            cmap="viridis",
            show_edges=True,
            show_scalar_bar=i == len(prepared) - 1,
            scalar_bar_args={"title": "u_h"},
        )
        plotter.view_isometric()
        plotter.camera.zoom(1.28)
    plotter.link_views()

    if screenshot is not None:
        path = Path(screenshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(path.as_posix())
        print(f"Wrote solution comparison screenshot to {path}")
    if show:
        plotter.show()
    else:
        plotter.close()


def _mode_rows(rows: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    return [row for row in rows if row["mode"] == mode]


def _save_condition_plot(rows: list[dict[str, object]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for mode in ("none", "ghost", "extension"):
        data = _mode_rows(rows, mode)
        if not data:
            continue
        h = np.asarray([float(row["h"]) for row in data])
        cond = np.asarray([float(row["condition"]) for row in data])
        ax.loglog(h, cond, marker="o", label=mode)

    ax.set_xlabel("h")
    ax.set_ylabel("active-system condition number")
    ax.set_title("Cut Poisson conditioning")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    ax.legend()

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote condition-number plot to {output}")


def _fit_log_slope(h: np.ndarray, error: np.ndarray) -> float | None:
    mask = np.isfinite(h) & np.isfinite(error) & (h > 0.0) & (error > 0.0)
    if np.count_nonzero(mask) < 2:
        return None
    slope, _ = np.polyfit(np.log(h[mask]), np.log(error[mask]), 1)
    return float(slope)


def _add_slope_guide(ax, h: np.ndarray, error: np.ndarray, slope: float) -> None:
    mask = np.isfinite(h) & np.isfinite(error) & (h > 0.0) & (error > 0.0)
    if np.count_nonzero(mask) < 2:
        return
    h_valid = h[mask]
    error_valid = error[mask]
    h0 = h_valid[-1]
    e0 = error_valid[-1]
    guide = e0 * (h_valid / h0) ** slope
    ax.loglog(h_valid, guide, linestyle="--", color="0.45", linewidth=0.9)
    ax.text(
        h_valid[0],
        guide[0],
        f"slope {slope:.2f}",
        fontsize=8,
        color="0.35",
        va="bottom",
    )


def _save_error_plot(rows: list[dict[str, object]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    quantities = (("l2_error", "L2 error"), ("h1_error", "H1 seminorm error"))
    for ax, (key, title) in zip(axes, quantities, strict=True):
        for mode in ("none", "ghost", "extension"):
            data = _mode_rows(rows, mode)
            if not data:
                continue
            h = np.asarray([float(row["h"]) for row in data])
            error = np.asarray([float(row[key]) for row in data])
            slope = _fit_log_slope(h, error)
            label = mode if slope is None else f"{mode} ({slope:.2f})"
            ax.loglog(h, error, marker="o", label=label)
            if slope is not None:
                _add_slope_guide(ax, h, error, slope)

        ax.set_xlabel("h")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()
        ax.legend(fontsize=8)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Wrote error convergence plot to {output}")


def _assemble_case(
    *,
    msh: mesh.Mesh,
    phi: fem.Function,
    mode: str,
    order: int,
    nitsche_gamma: float,
    ghost_gamma: float,
    extension_gamma: float,
    aggregation_threshold: float,
) -> tuple[fem.Function, float, float, float, float, float, float, float, dict[str, int]]:
    cut_data = cutfemx.cut(phi)
    inside_cells = cutfemx.locate_entities(cut_data, "phi < 0")
    cut_cells = cutfemx.locate_entities(cut_data, "phi = 0")
    cut_quadrature = cutfemx.runtime_quadrature(cut_data, "phi < 0", order=order)
    interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi = 0", order=order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi < 0")

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * u_exact
    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[inside_cells, cut_quadrature],
    )
    dx_gamma = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=interface_quadrature,
    )
    dS_ghost = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=2,
        subdomain_data=ghost_facets,
    )

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += (
        -ufl.dot(ufl.grad(u), n_gamma) * v
        - ufl.dot(ufl.grad(v), n_gamma) * u
        + nitsche_gamma / h * u * v
    ) * dx_gamma
    if mode == "ghost" and ghost_facets.size > 0 and ghost_gamma != 0.0:
        a += (
            ghost_gamma
            * h_avg
            * ufl.inner(
                ufl.jump(ufl.grad(u), n_facet),
                ufl.jump(ufl.grad(v), n_facet),
            )
            * dS_ghost
        )

    L = f * v * dx_omega
    L += (
        -ufl.dot(ufl.grad(v), n_gamma) * u_exact
        + nitsche_gamma / h * u_exact * v
    ) * dx_gamma

    a_form = cutfemx.fem.form(a)
    L_form = cutfemx.fem.form(L)

    b = cutfemx.fem.assemble_vector(L_form)
    b.scatter_reverse(la.InsertMode.add)
    extension_terms = []
    ill_posed_cells = np.empty(0, dtype=np.int32)
    if mode == "extension" and extension_gamma != 0.0:
        aggregation = cutfemx.extensions.create_cell_aggregation(
            cut_data,
            "phi < 0",
            aggregation_threshold,
            root_policy="interior_or_well_cut",
        )
        ill_posed_cells = aggregation.ill_posed_cells
        cell_h = 2.0 / round(np.sqrt(msh.topology.index_map(msh.topology.dim).size_local / 2))
        num_cells = cut_data.mesh.topology.index_map(cut_data.mesh.topology.dim).size_local
        beta = np.full(
            num_cells,
            extension_gamma / cell_h**2,
            dtype=msh.geometry.x.dtype,
        )
        extension_terms.append(
            cutfemx.extensions.ExtensionPenaltyTerm(
                V,
                beta=beta,
                quadrature_degree=order,
                cut_data=cut_data,
                aggregation=aggregation,
            )
        )

    A = cutfemx.fem.assemble_matrix(a_form, extension_terms=extension_terms)
    A.scatter_reverse()
    A_scipy = A.to_scipy().tocsr()

    active_cells = np.union1d(inside_cells, cut_cells)
    active_dofs = _cell_dofs(V, active_cells)
    condition, lambda_min, lambda_max = _active_condition_number(A_scipy, active_dofs)
    uh = _solve_active_system(A_scipy, b.array, V, active_dofs)

    error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_omega)
    error_sq = cutfemx.fem.assemble_scalar(error_form)
    l2_error = float(np.sqrt(max(msh.comm.allreduce(error_sq, op=MPI.SUM), 0.0)))

    h1_error_form = cutfemx.fem.form(
        ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * dx_omega
    )
    h1_error_sq = cutfemx.fem.assemble_scalar(h1_error_form)
    h1_error = float(np.sqrt(max(msh.comm.allreduce(h1_error_sq, op=MPI.SUM), 0.0)))

    boundary_error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_gamma)
    boundary_error_sq = cutfemx.fem.assemble_scalar(boundary_error_form)
    boundary_l2_error = float(
        np.sqrt(max(msh.comm.allreduce(boundary_error_sq, op=MPI.SUM), 0.0))
    )
    boundary_measure_form = cutfemx.fem.form(1.0 * dx_gamma)
    boundary_measure = float(
        msh.comm.allreduce(cutfemx.fem.assemble_scalar(boundary_measure_form), op=MPI.SUM)
    )
    boundary_rms_error = boundary_l2_error / np.sqrt(boundary_measure)

    info = {
        "dofs": int(V.dofmap.index_map.size_local * V.dofmap.index_map_bs),
        "active_dofs": int(active_dofs.size),
        "inside_cells": int(inside_cells.size),
        "cut_cells": int(cut_cells.size),
        "ghost_facets": int(ghost_facets.size),
        "ill_cells": int(ill_posed_cells.size),
        "active_cells": active_cells,
    }
    return (
        uh,
        l2_error,
        h1_error,
        boundary_l2_error,
        boundary_rms_error,
        condition,
        lambda_min,
        lambda_max,
        info,
    )


def run(
    *,
    n0: int,
    levels: int,
    order: int,
    radius: float,
    center: tuple[float, float],
    nitsche_gamma: float,
    ghost_gamma: float,
    extension_gamma: float,
    aggregation_threshold: float,
    plot: bool = False,
    screenshot: str | None = None,
    condition_plot: str | None = None,
    error_plot: str | None = None,
) -> list[dict[str, float | int | str | None]]:
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("This diagnostic study is serial-only.")

    rows: list[dict[str, float | int | str | None]] = []
    previous_l2_error: dict[str, tuple[float, float]] = {}
    previous_h1_error: dict[str, tuple[float, float]] = {}
    final_solution_records: list[dict[str, object]] = []

    for level in range(levels):
        n = n0 * 2**level
        msh = mesh.create_rectangle(
            comm,
            points=((-1.0, -1.0), (1.0, 1.0)),
            n=(n, n),
            cell_type=mesh.CellType.triangle,
        )
        V_phi = fem.functionspace(msh, ("Lagrange", 2))
        phi = fem.Function(V_phi, name="phi")
        phi.interpolate(_circle_level_set(radius, center))
        h = 2.0 / float(n)

        for mode in ("none", "ghost", "extension"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                (
                    uh,
                    l2_error,
                    h1_error,
                    boundary_l2_error,
                    boundary_rms_error,
                    condition,
                    lambda_min,
                    lambda_max,
                    info,
                ) = _assemble_case(
                    msh=msh,
                    phi=phi,
                    mode=mode,
                    order=order,
                    nitsche_gamma=nitsche_gamma,
                    ghost_gamma=ghost_gamma,
                    extension_gamma=extension_gamma,
                    aggregation_threshold=aggregation_threshold,
                )

            l2_rate = None
            if mode in previous_l2_error:
                h_old, error_old = previous_l2_error[mode]
                if error_old > 0.0 and l2_error > 0.0:
                    l2_rate = float(np.log(error_old / l2_error) / np.log(h_old / h))
            previous_l2_error[mode] = (h, l2_error)

            h1_rate = None
            if mode in previous_h1_error:
                h_old, error_old = previous_h1_error[mode]
                if error_old > 0.0 and h1_error > 0.0:
                    h1_rate = float(np.log(error_old / h1_error) / np.log(h_old / h))
            previous_h1_error[mode] = (h, h1_error)

            row = {
                "mode": mode,
                "n": n,
                "h": h,
                "l2_error": l2_error,
                "h1_error": h1_error,
                "boundary_l2_error": boundary_l2_error,
                "boundary_rms_error": boundary_rms_error,
                "l2_rate": l2_rate,
                "h1_rate": h1_rate,
                "condition": condition,
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                **info,
            }
            rows.append(row)

            if level == levels - 1 and (plot or screenshot is not None):
                final_solution_records.append(
                    {
                        "mode": mode,
                        "solution": uh,
                        "active_cells": info["active_cells"],
                    }
                )

    if MPI.COMM_WORLD.rank == 0 and (plot or screenshot is not None):
        _plot_final_solutions(final_solution_records, screenshot=screenshot, show=plot)
    if MPI.COMM_WORLD.rank == 0 and condition_plot is not None:
        _save_condition_plot(rows, condition_plot)
    if MPI.COMM_WORLD.rank == 0 and error_plot is not None:
        _save_error_plot(rows, error_plot)

    return rows


def _print_rows(rows: list[dict[str, float | int | str | None]]) -> None:
    print(
        "mode        n      h          active  cut  ill  ghost   "
        "cond(A)        lambda_min     L2 error      L2 rate   "
        "H1 error      H1 rate   bdy L2       bdy RMS"
    )
    for row in rows:
        l2_rate = row["l2_rate"]
        h1_rate = row["h1_rate"]
        l2_rate_text = "   -" if l2_rate is None else f"{float(l2_rate):6.2f}"
        h1_rate_text = "   -" if h1_rate is None else f"{float(h1_rate):6.2f}"
        print(
            f"{row['mode']:<9} "
            f"{int(row['n']):4d} "
            f"{float(row['h']):10.3e} "
            f"{int(row['active_dofs']):7d} "
            f"{int(row['cut_cells']):4d} "
            f"{int(row['ill_cells']):4d} "
            f"{int(row['ghost_facets']):6d} "
            f"{float(row['condition']):12.4e} "
            f"{float(row['lambda_min']):12.4e} "
            f"{float(row['l2_error']):12.4e} "
            f"{l2_rate_text} "
            f"{float(row['h1_error']):12.4e} "
            f"{h1_rate_text} "
            f"{float(row['boundary_l2_error']):12.4e} "
            f"{float(row['boundary_rms_error']):12.4e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare no stabilization, ghost penalty, and aggregation extension penalty for cut Poisson."
    )
    parser.add_argument("--n0", type=int, default=8, help="Initial cells per side.")
    parser.add_argument("--levels", type=int, default=3, help="Number of mesh refinements.")
    parser.add_argument("--order", type=int, default=4, help="Runtime quadrature order.")
    parser.add_argument("--radius", type=float, default=0.61)
    parser.add_argument("--center-x", type=float, default=0.08)
    parser.add_argument("--center-y", type=float, default=-0.04)
    parser.add_argument("--nitsche-gamma", type=float, default=40.0)
    parser.add_argument("--ghost-gamma", type=float, default=0.1)
    parser.add_argument("--extension-gamma", type=float, default=1.0)
    parser.add_argument("--aggregation-threshold", type=float, default=0.2)
    parser.add_argument("--plot", action="store_true", help="Show a PyVista warped solution comparison.")
    parser.add_argument("--screenshot", type=str, default=None, help="Write the solution comparison plot to an image.")
    parser.add_argument(
        "--condition-plot",
        type=str,
        default=None,
        help="Write a matplotlib condition-number comparison plot to this path.",
    )
    parser.add_argument(
        "--error-plot",
        type=str,
        default=None,
        help="Write a matplotlib L2/H1 convergence plot with fitted slopes to this path.",
    )
    args = parser.parse_args()

    rows = run(
        n0=args.n0,
        levels=args.levels,
        order=args.order,
        radius=args.radius,
        center=(args.center_x, args.center_y),
        nitsche_gamma=args.nitsche_gamma,
        ghost_gamma=args.ghost_gamma,
        extension_gamma=args.extension_gamma,
        aggregation_threshold=args.aggregation_threshold,
        plot=args.plot,
        screenshot=args.screenshot,
        condition_plot=args.condition_plot,
        error_plot=args.error_plot,
    )
    if MPI.COMM_WORLD.rank == 0:
        _print_rows(rows)


if __name__ == "__main__":
    main()
