# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

# # Cut DG Poisson problem on an implicitly defined domain
#
# This demo solves a manufactured Poisson problem on a domain described by a
# level-set function. The background mesh is the square [-1, 1]^2, while the
# physical domain is the negative side of a circle:
#
#     Omega = {x : phi(x) < 0},     Gamma = {x : phi(x) = 0}.
#
# The finite element space is discontinuous, so the interior skeleton of the
# active mesh also needs quadrature. The script keeps the construction visible:
# first cells are classified, then the active interior facets are classified,
# and finally the UFL measures are assembled from the standard and runtime
# quadrature pieces.
#
# The method is the symmetric interior penalty DG method on the cut domain,
# combined with Nitsche's method on the embedded boundary and a ghost penalty
# around cut cells. For the manufactured solution
#
#     u = sin(pi x) sin(pi y),     f = -Delta u = 2 pi^2 u,
#
# the boundary data on Gamma is simply the exact solution evaluated on the
# level-set boundary.

from __future__ import annotations

import argparse

from mpi4py import MPI

import cutfemx
import numpy as np
from runintgen import dSq, dxq

import ufl
from dolfinx import fem, la, mesh

# ## Linear solver
#
# The demo assembles CutFEMx runtime forms into DOLFINx MatrixCSR objects and
# solves the resulting serial system with SciPy. This mirrors the simple solve
# path in ``demo_poisson.py`` and keeps the focus on the variational problem.


def solve_runtime_system_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
) -> tuple[fem.Function, la.MatrixCSR, la.Vector, object]:
    """Assemble a MatrixCSR runtime system and solve it with SciPy."""
    from scipy.sparse.linalg import spsolve

    if V.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy solve path in this demo is serial-only.")

    b = cutfemx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    A = cutfemx.fem.assemble_matrix(a)
    A.scatter_reverse()

    active_domain = cutfemx.fem.active_domain(a)
    cutfemx.fem.deactivate_outside(A, b, active_domain)

    uh = fem.Function(V, name="u_h")
    uh.x.array[:] = spsolve(A.to_scipy(), b.array)
    uh.x.scatter_forward()
    return uh, A, b, active_domain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve a cut-domain SIPG Poisson problem with runtime quadrature."
    )
    parser.add_argument("--n", type=int, default=12, help="Initial mesh cells per side.")
    parser.add_argument("--levels", type=int, default=2, help="Number of refinements.")
    parser.add_argument("--degree", type=int, default=1, help="DG polynomial degree.")
    parser.add_argument("--order", type=int, default=4, help="Runtime quadrature order.")
    parser.add_argument("--radius", type=float, default=0.61)
    parser.add_argument("--center-x", type=float, default=0.08)
    parser.add_argument("--center-y", type=float, default=-0.04)
    parser.add_argument("--penalty", type=float, default=20.0)
    parser.add_argument("--interface-penalty", type=float, default=40.0)
    parser.add_argument("--ghost-penalty", type=float, default=0.1)
    parser.add_argument(
        "--check-convergence",
        action="store_true",
        help="Fail if any observed L2 rate is below --min-rate.",
    )
    parser.add_argument("--min-rate", type=float, default=1.0)
    args = parser.parse_args()

    if args.n < 2:
        raise ValueError("--n must be at least 2.")
    if args.levels < 1:
        raise ValueError("--levels must be at least 1.")
    if args.degree < 1:
        raise ValueError("--degree must be at least 1.")
    if args.order < 1:
        raise ValueError("--order must be at least 1.")

    comm = MPI.COMM_WORLD
    center = (args.center_x, args.center_y)
    rows: list[dict[str, float | int | None]] = []

    for level in range(args.levels):
        n_cells = args.n * 2**level

        # ## Background mesh and level set
        #
        # The mesh is not fitted to the circle. The level-set function is
        # interpolated into a continuous P1 space and drives all classifications.
        msh = mesh.create_rectangle(
            comm,
            ((-1.0, -1.0), (1.0, 1.0)),
            (n_cells, n_cells),
            cell_type=mesh.CellType.triangle,
        )
        V_phi = fem.functionspace(msh, ("Lagrange", 1))
        phi = fem.Function(V_phi, name="phi")

        def circle_level_set(x: np.ndarray) -> np.ndarray:
            return (
                np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
                - args.radius
            )

        phi.interpolate(circle_level_set)

        # ## Runtime quadrature on cells
        #
        # Cells completely inside Omega can use ordinary cell quadrature. Cut
        # cells receive runtime quadrature rules for the part phi < 0. The
        # inclusive selector phi <= 0 marks the full active cell set: ordinary
        # inside cells plus cells intersected by Gamma. The same cell cut also
        # gives the embedded-boundary rules on phi = 0.
        cell_cut = cutfemx.cut(phi)
        inside_cells = cutfemx.locate_entities(cell_cut, "phi<0")
        cell_rules = cutfemx.runtime_quadrature(cell_cut, "phi<0", args.order)
        interface_rules = cutfemx.runtime_quadrature(cell_cut, "phi=0", args.order)
        cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
        active_cells = cutfemx.locate_entities(cell_cut, "phi<=0")

        if active_cells.size == 0:
            raise RuntimeError("The selected cut domain has no active cells.")

        # ## Runtime quadrature on the DG skeleton
        #
        # The SIPG terms live on interior facets of the active mesh. Facets fully
        # inside Omega use standard dS data, while facets sliced by Gamma use
        # runtime dS quadrature on the portion where phi < 0.
        fdim = msh.topology.dim - 1
        skeleton_facets = cutfemx.interior_facets_for_cells(msh, active_cells)
        skeleton_cut = cutfemx.cut(phi, skeleton_facets, fdim)
        inside_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi<0")
        skeleton_rules = cutfemx.runtime_quadrature(skeleton_cut, "phi<0", args.order)
        cut_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi=0")

        # The ghost penalty is a standard interior-facet integral on a narrow
        # band surrounding the cut cells. It stabilizes small cut-cell slivers.
        ghost_facets = cutfemx.ghost_penalty_facets(cell_cut, "phi<0")

        # ## Measures
        #
        # ``dxq`` and ``dSq`` combine standard entity arrays with runtime
        # quadrature rules. This is the essential mixed-quadrature pattern for
        # cut DG: ordinary entities stay ordinary, and only genuinely cut pieces
        # carry generated runtime quadrature.
        dx_omega = dxq(0, domain=msh, subdomain_data=[inside_cells, cell_rules])
        dS_omega = dSq(0, domain=msh, subdomain_data=[inside_skeleton_facets, skeleton_rules])
        dx_gamma = dxq(1, domain=msh, subdomain_data=interface_rules)
        dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)
        n_gamma = cutfemx.normal(phi)

        # ## Variational formulation
        #
        # The volume term is integrated over Omega. Across active interior
        # facets we add the symmetric interior penalty DG terms:
        #
        #   - <avg(grad u), jump(v n)> - <avg(grad v), jump(u n)>
        #   + sigma / avg(h) <jump(u n), jump(v n)>.
        #
        # On Gamma, the boundary condition is imposed weakly with the symmetric
        # Nitsche terms. The ghost penalty acts only on the selected facet band.
        V = fem.functionspace(msh, ("DG", args.degree))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(msh)
        n = ufl.FacetNormal(msh)
        h = ufl.CellDiameter(msh)
        h_avg = ufl.avg(h)

        u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        f = 2.0 * np.pi**2 * u_exact
        sigma = args.penalty * args.degree**2
        sigma_gamma = args.interface_penalty * args.degree**2

        jump_u = ufl.jump(u, n)
        jump_v = ufl.jump(v, n)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
        a += -ufl.inner(ufl.avg(ufl.grad(u)), jump_v) * dS_omega
        a += -ufl.inner(ufl.avg(ufl.grad(v)), jump_u) * dS_omega
        a += sigma / h_avg * ufl.inner(jump_u, jump_v) * dS_omega
        a += (
            -ufl.dot(ufl.grad(u), n_gamma) * v
            - ufl.dot(ufl.grad(v), n_gamma) * u
            + sigma_gamma / h * u * v
        ) * dx_gamma

        if ghost_facets.size > 0 and args.ghost_penalty != 0.0:
            a += (
                args.ghost_penalty
                * h_avg
                * ufl.inner(
                    ufl.jump(ufl.grad(u), n),
                    ufl.jump(ufl.grad(v), n),
                )
                * dS_ghost
            )

        L = f * v * dx_omega
        L += (
            -ufl.dot(ufl.grad(v), n_gamma) * u_exact
            + sigma_gamma / h * u_exact * v
        ) * dx_gamma

        a_form = cutfemx.fem.form(a)
        L_form = cutfemx.fem.form(L)
        uh, A, b, active_domain = solve_runtime_system_scipy(a_form, L_form, V)

        error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_omega)
        error_sq = cutfemx.fem.assemble_scalar(error_form)
        error_sq = comm.allreduce(error_sq, op=MPI.SUM)
        l2_error = float(np.sqrt(max(error_sq, 0.0)))

        h_value = 2.0 / float(n_cells)
        rate = None
        if rows:
            previous = rows[-1]
            previous_error = float(previous["l2_error"])
            previous_h = float(previous["h"])
            if previous_error > 0.0 and l2_error > 0.0:
                rate = float(np.log(previous_error / l2_error) / np.log(previous_h / h_value))

        rows.append(
            {
                "n": n_cells,
                "h": h_value,
                "dofs": V.dofmap.index_map.size_local * V.dofmap.index_map_bs,
                "active_cells": active_domain.active_cells.size,
                "cut_cells": cut_cells.size,
                "skeleton_facets": inside_skeleton_facets.size,
                "cut_skeleton_facets": cut_skeleton_facets.size,
                "ghost_facets": ghost_facets.size,
                "interface_cells": cut_cells.size,
                "l2_error": l2_error,
                "rate": rate,
                "matrix_norm_sq": float(A.squared_norm()),
                "rhs_norm": float(np.linalg.norm(b.array)),
            }
        )

    if comm.rank == 0:
        print("CutFEMx cut DG Poisson convergence demo")
        print(
            "  n    h          dofs  active  cut  skel  cut-skel  ghost  "
            "L2-error     rate"
        )
        for row in rows:
            rate = "   -" if row["rate"] is None else f"{float(row['rate']):5.2f}"
            print(
                f"{int(row['n']):4d}  {float(row['h']):8.3e}  "
                f"{int(row['dofs']):5d}  {int(row['active_cells']):6d}  "
                f"{int(row['cut_cells']):3d}  {int(row['skeleton_facets']):4d}  "
                f"{int(row['cut_skeleton_facets']):8d}  {int(row['ghost_facets']):5d}  "
                f"{float(row['l2_error']):10.3e}  {rate}"
            )
        final = rows[-1]
        print(f"  final matrix squared norm: {float(final['matrix_norm_sq']):.6e}")
        print(f"  final rhs norm: {float(final['rhs_norm']):.6e}")
        print(f"  final interface cells: {int(final['interface_cells'])}")

    if args.check_convergence:
        if len(rows) < 2:
            raise RuntimeError("--check-convergence requires at least two levels.")
        rates = [row["rate"] for row in rows[1:]]
        if not all(
            rate is not None and np.isfinite(rate) and rate >= args.min_rate for rate in rates
        ):
            formatted = ", ".join("nan" if rate is None else f"{float(rate):.3f}" for rate in rates)
            raise RuntimeError(
                f"CutDG convergence check failed: observed rates [{formatted}] "
                f"below minimum {args.min_rate:.3f}."
            )


if __name__ == "__main__":
    main()
