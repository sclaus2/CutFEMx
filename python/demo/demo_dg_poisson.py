# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Cut DG Poisson problem on an implicitly defined disk.

The physical domain is Omega = {phi < 0}. The method is symmetric interior
penalty DG in the cut domain, with Nitsche terms on Gamma = {phi = 0} and a
ghost penalty on facets adjacent to cut cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
from _pyvista_solution_plot import add_plot_arguments, plot_scalar_cut_solution

import ufl
from dolfinx import fem, io, la, mesh


def circle_level_set(center: tuple[float, float], radius: float):
    """Return a level-set function that is negative inside a circle."""

    def phi(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius

    return phi


def solve_runtime_system_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
) -> fem.Function:
    """Assemble, deactivate inactive dofs, and solve the serial MatrixCSR system."""
    from scipy.sparse.linalg import spsolve

    if V.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy solve path is serial-only.")

    A = cutfemx.fem.assemble_matrix(a)
    A.scatter_reverse()
    b = cutfemx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    cutfemx.fem.deactivate_outside(A, b, cutfemx.fem.active_domain(a))

    uh = fem.Function(V, name="u_h")
    uh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    uh.x.scatter_forward()
    return uh


def interpolate_for_xdmf(u: fem.Function, name: str) -> fem.Function:
    """Interpolate a scalar field to the background mesh geometry degree."""
    msh = u.function_space.mesh
    cmaps = getattr(msh.geometry, "cmaps", None)
    degree = int(cmaps[0].degree if cmaps is not None else msh.geometry.cmap.degree)
    V_out = fem.functionspace(msh, ("Lagrange", degree))
    u_out = fem.Function(V_out, name=name)
    u_out.interpolate(u)
    u_out.x.scatter_forward()
    return u_out


def write_xdmf(
    output_dir: Path,
    cut_data: cutfemx.CutData,
    phi: fem.Function,
    uh: fem.Function,
    u_exact: fem.Function,
    error: fem.Function,
) -> None:
    """Write background and cut-domain XDMF files."""
    comm = uh.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    phi_out = interpolate_for_xdmf(phi, "phi")
    uh_out = interpolate_for_xdmf(uh, "u_h")
    exact_out = interpolate_for_xdmf(u_exact, "u_exact")
    error_out = interpolate_for_xdmf(error, "u_error")

    background_path = output_dir / "dg_poisson_background.xdmf"
    with io.XDMFFile(comm, background_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(uh.function_space.mesh)
        xdmf.write_function(phi_out)
        xdmf.write_function(uh_out)
        xdmf.write_function(exact_out)
        xdmf.write_function(error_out)

    cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("Cannot write output for an empty cut mesh.")

    phi_cut = cutfemx.fem.cut_function(phi_out, cut_mesh)
    phi_cut.name = "phi"
    uh_cut = cutfemx.fem.cut_function(uh_out, cut_mesh)
    uh_cut.name = "u_h"
    exact_cut = cutfemx.fem.cut_function(exact_out, cut_mesh)
    exact_cut.name = "u_exact"
    error_cut = cutfemx.fem.cut_function(error_out, cut_mesh)
    error_cut.name = "u_error"

    cut_path = output_dir / "dg_poisson_cut_domain.xdmf"
    with io.XDMFFile(comm, cut_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(phi_cut)
        xdmf.write_function(uh_cut)
        xdmf.write_function(exact_cut)
        xdmf.write_function(error_cut)

    if comm.rank == 0:
        print(f"Wrote background solution to {background_path}")
        print(f"Wrote cut-domain solution to {cut_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="Initial cells per side.")
    parser.add_argument(
        "--levels",
        type=int,
        default=1,
        help="Number of uniform refinement levels for convergence checks.",
    )
    parser.add_argument("--degree", type=int, default=1, help="DG polynomial degree.")
    parser.add_argument(
        "--order", type=int, default=4, help="Runtime quadrature order."
    )
    parser.add_argument("--radius", type=float, default=0.61, help="Disk radius.")
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        default=(0.08, -0.04),
        metavar=("X", "Y"),
        help="Disk centre.",
    )
    parser.add_argument(
        "--penalty", type=float, default=20.0, help="Interior penalty scale."
    )
    parser.add_argument(
        "--interface-penalty",
        type=float,
        default=40.0,
        help="Nitsche interface penalty scale.",
    )
    parser.add_argument(
        "--ghost-penalty", type=float, default=0.1, help="Ghost penalty scale."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dg_poisson_xdmf"),
        help="Directory for XDMF output in single-case mode.",
    )
    parser.add_argument(
        "--check-convergence",
        action="store_true",
        help="Run a refinement sweep and fail if the final rate is too small.",
    )
    parser.add_argument(
        "--min-rate",
        type=float,
        default=1.0,
        help="Minimum final L2 convergence rate for --check-convergence.",
    )
    add_plot_arguments(parser)
    return parser.parse_args()


def run_case(
    *,
    comm: MPI.Comm,
    n: int,
    degree: int,
    order: int,
    radius: float,
    center: tuple[float, float],
    penalty: float,
    interface_penalty: float,
    ghost_penalty: float,
    output_dir: Path | None = None,
    write_output: bool = False,
    plot_solution: bool = False,
) -> float:
    """Solve one manufactured cut DG Poisson case and return its L2 error."""
    msh = mesh.create_rectangle(
        comm,
        ((-1.0, -1.0), (1.0, 1.0)),
        (n, n),
        cell_type=mesh.CellType.triangle,
    )

    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(circle_level_set(center, radius))
    phi.x.scatter_forward()

    cell_cut = cutfemx.cut(phi)
    inside_cells = cutfemx.locate_entities(cell_cut, "phi<0")
    active_cells = cutfemx.locate_entities(cell_cut, "phi<=0")
    cell_rules = cutfemx.runtime_quadrature(cell_cut, "phi<0", order)
    interface_rules = cutfemx.runtime_quadrature(cell_cut, "phi=0", order)
    cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")

    facet_dim = msh.topology.dim - 1
    skeleton_facets = cutfemx.interior_facets_for_cells(msh, active_cells)
    skeleton_cut = cutfemx.cut(phi, skeleton_facets, facet_dim)
    omega_interior_facets = cutfemx.locate_entities(skeleton_cut, "phi<0")
    omega_cut_facet_rules = cutfemx.runtime_quadrature(skeleton_cut, "phi<0", order)
    cut_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi=0")
    ghost_facets = cutfemx.ghost_penalty_facets(cell_cut, "phi<0")

    dx_omega = ufl.Measure(
        "dx", domain=msh, subdomain_id=0, subdomain_data=[inside_cells, cell_rules]
    )
    dS_omega = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[omega_interior_facets, omega_cut_facet_rules],
    )
    dx_gamma = ufl.Measure(
        "dx", domain=msh, subdomain_id=1, subdomain_data=interface_rules
    )
    dS_ghost = ufl.Measure(
        "dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets
    )

    V = fem.functionspace(msh, ("DG", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    n_facet = ufl.FacetNormal(msh)
    n_gamma = cutfemx.normal(phi)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * u_exact
    sigma = penalty * degree**2
    sigma_gamma = interface_penalty * degree**2
    jump_u = ufl.jump(u, n_facet)
    jump_v = ufl.jump(v, n_facet)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += -ufl.inner(ufl.avg(ufl.grad(u)), jump_v) * dS_omega
    a += -ufl.inner(ufl.avg(ufl.grad(v)), jump_u) * dS_omega
    a += sigma / h_avg * ufl.inner(jump_u, jump_v) * dS_omega
    a += (
        -ufl.dot(ufl.grad(u), n_gamma) * v
        - ufl.dot(ufl.grad(v), n_gamma) * u
        + sigma_gamma / h * u * v
    ) * dx_gamma
    if ghost_facets.size > 0:
        a += (
            ghost_penalty
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
        + sigma_gamma / h * u_exact * v
    ) * dx_gamma

    uh = solve_runtime_system_scipy(cutfemx.fem.form(a), cutfemx.fem.form(L), V)

    u_exact_bg = fem.Function(V, name="u_exact")
    u_exact_bg.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u_exact_bg.x.scatter_forward()

    error_bg = fem.Function(V, name="u_error")
    error_bg.x.array[:] = uh.x.array - u_exact_bg.x.array
    error_bg.x.scatter_forward()

    error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_omega)
    error_sq = cutfemx.fem.assemble_scalar(error_form)
    error_sq = comm.allreduce(error_sq, op=MPI.SUM)
    error = float(np.sqrt(max(error_sq, 0.0)))

    if comm.rank == 0:
        print(f"Cut DG Poisson problem on a disk, n={n}, degree={degree}")
        print(f"active cells        = {active_cells.size}")
        print(f"cut cells           = {cut_cells.size}")
        print(f"interior skeleton facets = {omega_interior_facets.size}")
        print(f"cut skeleton facets = {cut_skeleton_facets.size}")
        print(f"ghost facets        = {ghost_facets.size}")
        print(f"L2 error            = {error:.6e}")

    if write_output:
        if output_dir is None:
            raise ValueError("write_output=True requires output_dir.")
        write_xdmf(output_dir, cell_cut, phi, uh, u_exact_bg, error_bg)

    if comm.rank == 0 and plot_solution:
        plot_scalar_cut_solution(
            cell_cut,
            "phi<0",
            uh,
            title="Cut DG Poisson solution",
            field_name="u_h",
        )

    return error


def run_convergence(args: argparse.Namespace, comm: MPI.Comm) -> None:
    """Run a refinement sweep and enforce the requested final rate."""
    if args.levels < 2:
        raise ValueError("--check-convergence requires --levels >= 2.")

    previous_error: float | None = None
    previous_h: float | None = None
    final_rate: float | None = None

    for level in range(args.levels):
        n = int(args.n * 2**level)
        h = 2.0 / float(n)
        error = run_case(
            comm=comm,
            n=n,
            degree=args.degree,
            order=args.order,
            radius=args.radius,
            center=tuple(args.center),
            penalty=args.penalty,
            interface_penalty=args.interface_penalty,
            ghost_penalty=args.ghost_penalty,
        )
        if previous_error is not None and previous_h is not None:
            final_rate = float(np.log(previous_error / error) / np.log(previous_h / h))
            if comm.rank == 0:
                print(f"level {level}: h={h:.6e} L2={error:.6e} rate={final_rate:.3f}")
        elif comm.rank == 0:
            print(f"level {level}: h={h:.6e} L2={error:.6e}")

        previous_error = error
        previous_h = h

    if final_rate is None or final_rate < args.min_rate:
        raise RuntimeError(
            f"Cut DG Poisson final L2 rate {final_rate} is below {args.min_rate}."
        )


def main() -> None:
    """Run the demo."""
    args = parse_args()
    comm = MPI.COMM_WORLD

    if args.n < 1:
        raise ValueError("--n must be positive.")
    if args.degree < 1:
        raise ValueError("--degree must be positive.")
    if args.order < 1:
        raise ValueError("--order must be positive.")
    if args.check_convergence:
        run_convergence(args, comm)
    else:
        run_case(
            comm=comm,
            n=args.n,
            degree=args.degree,
            order=args.order,
            radius=args.radius,
            center=tuple(args.center),
            penalty=args.penalty,
            interface_penalty=args.interface_penalty,
            ghost_penalty=args.ghost_penalty,
            output_dir=args.output_dir,
            write_output=True,
            plot_solution=not args.no_plot,
        )


if __name__ == "__main__":
    main()
