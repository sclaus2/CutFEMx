# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Surface Poisson DG problem on an implicitly defined sphere.

The surface is Gamma = {phi = 0}. This demo solves

    -Delta_Gamma u + u = f

with a SIPG formulation on a cut surface. The manufactured solution is
u = x - x_c on the sphere.
"""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import fem, io, la, mesh


def squared_distance(x: np.ndarray, center: tuple[float, float, float]) -> np.ndarray:
    """Return |x - center|^2 for interpolation on the background mesh."""
    return (
        (x[0] - center[0]) ** 2
        + (x[1] - center[1]) ** 2
        + (x[2] - center[2]) ** 2
    )


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

    active_domain = cutfemx.fem.active_domain(a)
    cutfemx.fem.deactivate_outside(A, b, active_domain)

    uh = fem.Function(V, name="u_h")
    uh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    uh.x.scatter_forward()
    return uh


def interpolate_for_xdmf(u: fem.Function, name: str) -> fem.Function:
    """Interpolate a scalar function into an XDMF-compatible Lagrange space."""
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
    cell_cut: cutfemx.CutData,
    phi: fem.Function,
    uh: fem.Function,
    u_exact: fem.Function,
    error: fem.Function,
) -> None:
    """Write XDMF previews and a direct DG VTK file on the cut surface."""
    comm = uh.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    phi_out = interpolate_for_xdmf(phi, "phi")
    uh_out = interpolate_for_xdmf(uh, "u_h")
    exact_out = interpolate_for_xdmf(u_exact, "u_exact")
    error_out = interpolate_for_xdmf(error, "u_error")

    background_path = output_dir / "surface_poisson_dg_background.xdmf"
    with io.XDMFFile(comm, background_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(uh.function_space.mesh)
        xdmf.write_function(phi_out)
        xdmf.write_function(uh_out)
        xdmf.write_function(exact_out)
        xdmf.write_function(error_out)

    gamma_mesh = cutfemx.create_cut_mesh(cell_cut, "phi=0", mode="cut_only")
    if gamma_mesh.mesh is None:
        raise RuntimeError("Cannot write surface output for an empty cut mesh.")

    phi_gamma = cutfemx.fem.cut_function(phi_out, gamma_mesh)
    phi_gamma.name = "phi"
    uh_gamma = cutfemx.fem.cut_function(uh_out, gamma_mesh)
    uh_gamma.name = "u_h"
    exact_gamma = cutfemx.fem.cut_function(exact_out, gamma_mesh)
    exact_gamma.name = "u_exact"
    error_gamma = cutfemx.fem.cut_function(error_out, gamma_mesh)
    error_gamma.name = "u_error"

    gamma_path = output_dir / "surface_poisson_dg_gamma.xdmf"
    with io.XDMFFile(comm, gamma_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(gamma_mesh.mesh)
        xdmf.write_function(phi_gamma)
        xdmf.write_function(uh_gamma)
        xdmf.write_function(exact_gamma)
        xdmf.write_function(error_gamma)

    uh_gamma_dg = cutfemx.fem.cut_function(uh, gamma_mesh)
    uh_gamma_dg.name = "u_h"
    exact_gamma_dg = cutfemx.fem.cut_function(u_exact, gamma_mesh)
    exact_gamma_dg.name = "u_exact"
    error_gamma_dg = cutfemx.fem.cut_function(error, gamma_mesh)
    error_gamma_dg.name = "u_error"

    gamma_dg_path = output_dir / "surface_poisson_dg_gamma_dg.pvd"
    with io.VTKFile(comm, gamma_dg_path.as_posix(), "w") as vtk:
        vtk.write_mesh(gamma_mesh.mesh)
        vtk.write_function([uh_gamma_dg, exact_gamma_dg, error_gamma_dg])

    if comm.rank == 0:
        print(f"Wrote background solution to {background_path}")
        print(f"Wrote cut-surface XDMF preview to {gamma_path}")
        print(f"Wrote DG cut-surface solution to {gamma_dg_path}")


comm = MPI.COMM_WORLD

n = 10
degree = 1
levelset_degree = 2
quadrature_order = 4
radius = 0.62
center = (0.05, -0.03, 0.02)
penalty = 20.0
ghost_stabilization = 0.1
output_dir = Path("surface_poisson_dg_xdmf")

msh = mesh.create_box(
    comm,
    (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
    (n, n, n),
    cell_type=mesh.CellType.hexahedron,
)

V_phi = fem.functionspace(msh, ("Lagrange", levelset_degree))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(lambda x: squared_distance(x, center) - radius**2)
phi.x.scatter_forward()

cell_cut = cutfemx.cut(phi)
cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
gamma_rules = cutfemx.runtime_quadrature(
    cell_cut, "phi=0", quadrature_order, backend="algoim"
)

facet_dim = msh.topology.dim - 1
skeleton_facets = cutfemx.interior_facets_for_cells(msh, cut_cells)
facet_cut = cutfemx.cut(phi, skeleton_facets, facet_dim)
skeleton_rules = cutfemx.runtime_quadrature(
    facet_cut, "phi=0", quadrature_order, backend="algoim"
)
surface_skeleton_facets = cutfemx.locate_entities(facet_cut, "phi=0")
ghost_facets = surface_skeleton_facets

dx_gamma = ufl.Measure("dx", domain=msh, subdomain_data=gamma_rules)
dS_gamma = ufl.Measure("dS", domain=msh, subdomain_data=skeleton_rules)
dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)

V = fem.functionspace(msh, ("DG", degree))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
n_gamma = cutfemx.normal(phi)
mu = cutfemx.conormal(n_gamma)
n_facet = ufl.FacetNormal(msh)
h = ufl.CellDiameter(msh)
h_avg = ufl.avg(h)

I = ufl.Identity(msh.geometry.dim)
P = I - ufl.outer(n_gamma, n_gamma)
grad_G_u = ufl.dot(P, ufl.grad(u))
grad_G_v = ufl.dot(P, ufl.grad(v))

n_p = n_gamma("+")
n_m = n_gamma("-")
P_p = I - ufl.outer(n_p, n_p)
P_m = I - ufl.outer(n_m, n_m)
avg_grad_G_u = 0.5 * (
    ufl.dot(P_p, ufl.grad(u)("+")) + ufl.dot(P_m, ufl.grad(u)("-"))
)
avg_grad_G_v = 0.5 * (
    ufl.dot(P_p, ufl.grad(v)("+")) + ufl.dot(P_m, ufl.grad(v)("-"))
)
jump_u_mu = ufl.jump(u, mu)
jump_v_mu = ufl.jump(v, mu)

u_surface = x[0] - center[0]
f = (1.0 + 2.0 / radius**2) * u_surface

a = (ufl.inner(grad_G_u, grad_G_v) + u * v) * dx_gamma
a += -ufl.inner(avg_grad_G_u, jump_v_mu) * dS_gamma
a += -ufl.inner(avg_grad_G_v, jump_u_mu) * dS_gamma
a += penalty * degree**2 / h_avg * ufl.inner(jump_u_mu, jump_v_mu) * dS_gamma

if ghost_facets.size > 0:
    a += (
        ghost_stabilization
        * ufl.inner(
            ufl.jump(ufl.grad(u), n_facet),
            ufl.jump(ufl.grad(v), n_facet),
        )
        * dS_ghost
    )

L = f * v * dx_gamma

uh = solve_runtime_system_scipy(cutfemx.fem.form(a), cutfemx.fem.form(L), V)
uh.name = "u_h"

u_exact_bg = fem.Function(V, name="u_exact")
u_exact_bg.interpolate(lambda x: x[0] - center[0])
u_exact_bg.x.scatter_forward()

error_bg = fem.Function(V, name="u_error")
error_bg.x.array[:] = uh.x.array - u_exact_bg.x.array
error_bg.x.scatter_forward()

error_form = cutfemx.fem.form((uh - u_surface) ** 2 * dx_gamma)
error_sq = cutfemx.fem.assemble_scalar(error_form)
error_sq = comm.allreduce(error_sq, op=MPI.SUM)

one = fem.Constant(msh, np.float64(1.0))
surface_measure = cutfemx.fem.assemble_scalar(cutfemx.fem.form(one * dx_gamma))
surface_measure = comm.allreduce(surface_measure, op=MPI.SUM)

if comm.rank == 0:
    print(f"Surface Poisson DG on a 3D sphere, n={n}, degree={degree}")
    print(f"cut cells       = {cut_cells.size}")
    print(f"skeleton facets = {skeleton_facets.size}")
    print(f"surface facets  = {surface_skeleton_facets.size}")
    print(f"ghost facets    = {ghost_facets.size}")
    print(f"Surface measure = {surface_measure:.6e}")
    print(f"L2 error        = {np.sqrt(max(error_sq, 0.0)):.6e}")

write_xdmf(output_dir, cell_cut, phi, uh, u_exact_bg, error_bg)
