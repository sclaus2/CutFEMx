# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Poisson problem on an implicitly defined flower-shaped cut domain.

The physical domain is Omega = {phi < 0}, where phi describes a polar flower
embedded in a triangular background mesh. Dirichlet data are imposed weakly on
the embedded boundary Gamma = {phi = 0} with symmetric Nitsche terms.
"""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import fem, io, la, mesh


def flower_level_set(base_radius: float, amplitude: float, petals: int):
    """Return a level-set function that is negative inside a polar flower."""

    def phi(x: np.ndarray) -> np.ndarray:
        theta = np.arctan2(x[1], x[0])
        boundary_radius = base_radius + amplitude * np.cos(petals * theta)
        return np.sqrt(x[0] ** 2 + x[1] ** 2) - boundary_radius

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
    """Write background and physical-domain XDMF files."""
    comm = uh.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    phi_out = interpolate_for_xdmf(phi, "phi")
    uh_out = interpolate_for_xdmf(uh, "u_h")
    exact_out = interpolate_for_xdmf(u_exact, "u_exact")
    error_out = interpolate_for_xdmf(error, "u_error")

    background_path = output_dir / "poisson_background.xdmf"
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

    cut_path = output_dir / "poisson_cut_domain.xdmf"
    with io.XDMFFile(comm, cut_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(phi_cut)
        xdmf.write_function(uh_cut)
        xdmf.write_function(exact_cut)
        xdmf.write_function(error_cut)

    if comm.rank == 0:
        print(f"Wrote background solution to {background_path}")
        print(f"Wrote cut-domain solution to {cut_path}")


comm = MPI.COMM_WORLD

n = 32
order = 4
base_radius = 0.46
amplitude = 0.15
petals = 6
gamma = 40.0
gamma_g = 0.1
output_dir = Path("poisson_xdmf")

msh = mesh.create_rectangle(
    comm,
    ((-1.0, -1.0), (1.0, 1.0)),
    (n, n),
    cell_type=mesh.CellType.triangle,
)

V_phi = fem.functionspace(msh, ("Lagrange", 1))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(flower_level_set(base_radius, amplitude, petals))
phi.x.scatter_forward()

cut_data = cutfemx.cut(phi)
inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
volume_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

dx_omega = ufl.Measure(
    "dx", domain=msh, subdomain_id=0, subdomain_data=[inside_cells, volume_rules]
)
dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=interface_rules)
dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)

V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
n_gamma = cutfemx.normal(phi)
n_facet = ufl.FacetNormal(msh)
h = ufl.CellDiameter(msh)
h_avg = ufl.avg(h)

u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
f = 2.0 * np.pi**2 * u_exact

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
a += (
    -ufl.dot(ufl.grad(u), n_gamma) * v
    - ufl.dot(ufl.grad(v), n_gamma) * u
    + gamma / h * u * v
) * dx_gamma
if ghost_facets.size > 0:
    a += (
        gamma_g
        * h_avg
        * ufl.inner(
            ufl.jump(ufl.grad(u), n_facet),
            ufl.jump(ufl.grad(v), n_facet),
        )
        * dS_ghost
    )

L = f * v * dx_omega
L += (-ufl.dot(ufl.grad(v), n_gamma) * u_exact + gamma / h * u_exact * v) * dx_gamma

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

if comm.rank == 0:
    print(f"Cut Poisson problem on a {petals}-petal flower, n={n}")
    print(f"inside cells = {inside_cells.size}")
    print(f"cut cells    = {interface_rules.parent_map.size}")
    print(f"ghost facets = {ghost_facets.size}")
    print(f"L2 error     = {np.sqrt(max(error_sq, 0.0)):.6e}")

write_xdmf(output_dir, cut_data, phi, uh, u_exact_bg, error_bg)
