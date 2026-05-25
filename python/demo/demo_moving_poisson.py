# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Poisson problem on a moving unfitted circular domain.

The demo rebuilds the CutFEMx cut data and runtime quadrature rules for each
circle position. This is intentionally explicit: it shows the current migration
workflow for moving geometries without relying on the legacy ``dC/cutcell`` UFL
integral type.
"""

from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import default_real_type, fem, la, mesh
from scipy.sparse.linalg import spsolve


N = 11
STEPS = 3
ORDER = 2
RADIUS = 0.5
GAMMA = 100.0
GAMMA_G = 0.1
WARP_FACTOR = 10.0


@dataclass
class StepResult:
    center: tuple[float, float]
    solution: fem.Function
    inside_cells: int
    cut_cells: int
    ghost_facets: int


def _circle_level_set(center: tuple[float, float], radius: float):
    def marker(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius

    return marker


def _solve_step(
    msh: mesh.Mesh,
    V: fem.FunctionSpace,
    level_set: fem.Function,
    *,
    center: tuple[float, float],
    radius: float,
    order: int,
    gamma: float,
    gamma_g: float,
) -> StepResult:
    level_set.interpolate(_circle_level_set(center, radius))
    level_set.x.scatter_forward()

    cut_data = cutfemx.cut(level_set)
    inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
    cut_cells = cutfemx.locate_entities(cut_data, "phi=0")
    inside_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

    if inside_cells.size == 0 and inside_rules.parent_map.size == 0:
        raise RuntimeError(f"No active cells for center={center}.")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, default_real_type(1.0))
    g = fem.Constant(msh, default_real_type(0.0))
    n_gamma = cutfemx.normal(level_set)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=[inside_cells, inside_rules],
    )
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=2, subdomain_data=interface_rules)
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=3, subdomain_data=ghost_facets)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += -ufl.dot(ufl.grad(u), n_gamma) * v * dx_gamma
    a += -ufl.dot(ufl.grad(v), n_gamma) * u * dx_gamma
    a += gamma / h * u * v * dx_gamma
    if ghost_facets.size > 0 and gamma_g != 0.0:
        a += (
            gamma_g
            * h_avg
            * ufl.dot(
                ufl.jump(ufl.grad(u), n_facet),
                ufl.jump(ufl.grad(v), n_facet),
            )
            * dS_ghost
        )

    L = f * v * dx_omega
    L += -ufl.dot(ufl.grad(v), n_gamma) * g * dx_gamma
    L += gamma / h * v * g * dx_gamma

    a_cut = cutfemx.fem.form(a)
    L_cut = cutfemx.fem.form(L)

    b = cutfemx.fem.assemble_vector(L_cut)
    b.scatter_reverse(la.InsertMode.add)
    A = cutfemx.fem.assemble_matrix(a_cut)
    A.scatter_reverse()
    cutfemx.fem.deactivate_outside(A, b, cutfemx.fem.active_domain(a_cut))

    uh = fem.Function(V, name=f"u_center_{center[0]:.2f}")
    uh.x.array[:] = spsolve(A.to_scipy(), b.array)
    uh.x.scatter_forward()

    return StepResult(
        center=center,
        solution=uh,
        inside_cells=int(inside_cells.size),
        cut_cells=int(cut_cells.size),
        ghost_facets=int(ghost_facets.size),
    )


def _plot_results(
    msh: mesh.Mesh,
    results: list[StepResult],
) -> None:
    if msh.comm.rank != 0:
        return

    try:
        import pyvista
        from dolfinx import plot
    except ModuleNotFoundError:
        print("pyvista is required for this demo")
        return

    columns = max(1, len(results))
    plotter = pyvista.Plotter(shape=(1, columns))
    cells, types, x = plot.vtk_mesh(msh)

    values = np.concatenate([np.asarray(result.solution.x.array).real for result in results])
    clim = (float(np.min(values)), float(np.max(values)))

    for i, result in enumerate(results):
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = np.asarray(result.solution.x.array).real[: grid.n_points]
        grid.set_active_scalars("u")
        warped = grid.warp_by_scalar(scale_factor=WARP_FACTOR)
        plotter.subplot(0, i)
        plotter.add_title(f"center=({result.center[0]:.1f}, {result.center[1]:.1f})")
        plotter.add_mesh(warped, scalars="u", cmap="viridis", clim=clim, show_edges=True)

    plotter.show()


def main() -> None:
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0, -1.0), (4.0, 1.0)),
        n=(5 * N, N),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V, name="phi")

    results: list[StepResult] = []
    for step in range(STEPS):
        center = (float(step), 0.0)
        result = _solve_step(
            msh,
            V,
            level_set,
            center=center,
            radius=RADIUS,
            order=ORDER,
            gamma=GAMMA,
            gamma_g=GAMMA_G,
        )
        results.append(result)
        if comm.rank == 0:
            print(
                f"step {step}: center={result.center}, "
                f"inside={result.inside_cells}, cut={result.cut_cells}, "
                f"ghost={result.ghost_facets}"
            )

    _plot_results(msh, results)


if __name__ == "__main__":
    main()
