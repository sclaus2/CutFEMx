# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Compare the runtime (cut-cell) kernel variants on this machine.

Assembles bilinear forms over the cut cells of a sphere, with

* ``previous``: runintgen's per-point tensor updates
  (``runintgen_chunked_contraction=False``, ``runintgen_affine_geometry=False``)
* ``current``: the defaults (chunked contraction, Jacobian once per affine cell)

and prints the minimum assembly time of each and whether the matrices agree.
Pass the compiler flags you use, e.g. ``--cflags "-O2 -march=native"``; the
default is DOLFINx's ``-O2``. Runs in parallel with ``mpirun``.
"""

from __future__ import annotations

import argparse
import tempfile
import time

from mpi4py import MPI

import cutfemx
import cutfemx.fem
import numpy as np

import basix.ufl
import ufl
from dolfinx import fem, mesh

_PREVIOUS = {"runintgen_chunked_contraction": False, "runintgen_affine_geometry": False}


def _cut_cell_measure(cell_type: mesh.CellType, n: int, order: int):
    msh = mesh.create_box(MPI.COMM_WORLD, [[-1, -1, -1], [1, 1, 1]], [n, n, n], cell_type)
    phi = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
    phi.interpolate(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) - 0.71)
    rules = cutfemx.runtime_quadrature(cutfemx.cut(phi), "phi<0", order)
    return msh, ufl.Measure("dx", domain=msh, subdomain_data=rules), rules.weights.size


def _form(name: str, scale: int):
    kind, degree, cell = name.split()
    degree = int(degree[1])
    tet = cell == "tet"
    n = scale * ((24 if degree == 1 else 16) if tet else (16 if degree == 1 else 10))
    cell_type = mesh.CellType.tetrahedron if tet else mesh.CellType.hexahedron
    msh, dx, num_points = _cut_cell_measure(cell_type, n, 2 * degree)
    if kind == "stokes":
        W = fem.functionspace(
            msh,
            basix.ufl.mixed_element(
                [
                    basix.ufl.element("Lagrange", msh.basix_cell(), degree, shape=(3,)),
                    basix.ufl.element("Lagrange", msh.basix_cell(), degree - 1),
                ]
            ),
        )
        (u, p), (v, q) = ufl.TrialFunctions(W), ufl.TestFunctions(W)
        return (
            ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - q * ufl.div(u)
        ) * dx, num_points
    shape = (3,) if kind == "elasticity" else ()
    V = fem.functionspace(msh, ("Lagrange", degree, shape))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    if kind == "poisson":
        return ufl.inner(ufl.grad(u), ufl.grad(v)) * dx, num_points
    if kind == "coefficient":
        c = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
        c.interpolate(lambda x: 1.0 + x[0] ** 2)
        return c * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx, num_points
    eps = lambda w: ufl.sym(ufl.grad(w))  # noqa: E731
    return (2.0 * ufl.inner(eps(u), eps(v)) + 1.5 * ufl.div(u) * ufl.div(v)) * dx, num_points


def _assemble(a, options, jit_options, repeat):
    a_form = cutfemx.fem.form(a, form_compiler_options=options, jit_options=jit_options)
    comm = MPI.COMM_WORLD
    best = np.inf
    for _ in range(repeat):
        comm.Barrier()
        start = time.perf_counter()
        A = cutfemx.fem.assemble_matrix(a_form)
        A.scatter_reverse()
        best = min(best, comm.allreduce(time.perf_counter() - start, op=MPI.MAX))
    return best, A.to_scipy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--scale", type=int, default=1, help="mesh refinement factor")
    parser.add_argument("--cflags", default=None, help='e.g. "-O2 -march=native"')
    parser.add_argument("--cache-dir", default=None, help="JIT cache (default: temporary)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    cache_dir = args.cache_dir or comm.bcast(tempfile.mkdtemp() if comm.rank == 0 else None)
    jit_options = {"cache_dir": cache_dir}
    if args.cflags:
        jit_options["cffi_extra_compile_args"] = args.cflags.split()

    names = [
        "poisson P1 tet",
        "poisson P2 tet",
        "coefficient P2 tet",
        "elasticity P1 tet",
        "elasticity P2 tet",
        "elasticity Q2 hex",
        "stokes P2 tet",
    ]
    if comm.rank == 0:
        print(f"{comm.size} processes, flags {args.cflags or 'DOLFINx default'}")
        header = ("form", "points", "previous", "current", "speedup")
        print("{:20s} {:>9s} {:>10s} {:>10s} {:>8s}  matrices".format(*header))
    for name in names:
        a, num_points = _form(name, args.scale)
        previous, A0 = _assemble(a, _PREVIOUS, jit_options, args.repeat)
        current, A1 = _assemble(a, None, jit_options, args.repeat)
        scale = comm.allreduce(abs(A0).max() if A0.nnz else 0.0, op=MPI.MAX)
        difference = comm.allreduce(abs(A1 - A0).max() if A0.nnz else 0.0, op=MPI.MAX)
        num_points = comm.allreduce(num_points, op=MPI.SUM)
        if comm.rank == 0:
            agree = "agree" if difference <= 1e-12 * scale else f"DIFFER ({difference / scale:.1e})"
            print(
                f"{name:20s} {num_points:9d} {previous * 1e3:8.1f}ms {current * 1e3:8.1f}ms"
                f" {previous / current:8.2f}  {agree}"
            )


if __name__ == "__main__":
    main()
