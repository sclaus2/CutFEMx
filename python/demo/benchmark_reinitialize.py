# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Time level-set reinitialization on this machine.

Reinitializes ``phi = |x|^2 - r^2`` in a box as the signed distance to the
sphere ``|x| = r``, on the whole mesh and in a band of a few cells around the
sphere, and prints the minimum time of the distance solve, the maximum error
against ``|x| - r`` (inside the band for the band solve) and the solver's
round and relaxation counts (one relaxation per vertex is the fast marching
optimum). Runs in parallel with ``mpirun``.
"""

from __future__ import annotations

import argparse
import time

from mpi4py import MPI

import numpy as np

import dolfinx.common
from cutfemx.distance import reinitialize
from dolfinx import fem, log, mesh


def _solver_seconds() -> float:
    """Accumulated time of the distance solver, without setup and cut."""
    try:
        elapsed = dolfinx.common.timing("FIM solve")[1]
    except RuntimeError:  # not run yet
        return 0.0
    return elapsed.total_seconds() if hasattr(elapsed, "total_seconds") else elapsed


def _solve(V, radius, repeat, max_distance):
    comm = V.mesh.comm
    times = []
    for i in range(repeat):
        phi = fem.Function(V)
        phi.interpolate(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - radius**2)
        comm.Barrier()
        before = _solver_seconds()
        if i == 0:
            # Logs the solver's round and relaxation counts
            log.set_log_level(log.LogLevel.INFO)
        reinitialize(phi, max_iter=1000, tol=1e-10, max_distance=max_distance)
        log.set_log_level(log.LogLevel.WARNING)
        times.append(comm.allreduce(_solver_seconds() - before, op=MPI.MAX))
    return min(times), phi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=48, help="cells per direction")
    parser.add_argument(
        "--cell", choices=("tetrahedron", "hexahedron"), default="hexahedron"
    )
    parser.add_argument("--band", type=float, default=3.0, help="band width in cells")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    radius = 0.3
    msh = mesh.create_box(
        comm,
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        [args.n] * 3,
        getattr(mesh.CellType, args.cell),
        ghost_mode=mesh.GhostMode.shared_facet,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    exact = np.linalg.norm(V.tabulate_dof_coordinates(), axis=1) - radius
    num_vertices = msh.topology.index_map(0).size_global

    if comm.rank == 0:
        print(f"{args.cell}, {num_vertices} vertices, {comm.size} processes")
    for label, max_distance in (("full", None), ("band", args.band / args.n)):
        seconds, phi = _solve(V, radius, args.repeat, max_distance)
        error = np.abs(phi.x.array - exact)
        if max_distance is not None:
            error = error[np.abs(exact) < 0.8 * max_distance]
        error = comm.allreduce(error.max(initial=0.0), op=MPI.MAX)
        if comm.rank == 0:
            print(f"{label:4s}: {seconds:8.3f} s, max error {error:.3e}")


if __name__ == "__main__":
    main()
