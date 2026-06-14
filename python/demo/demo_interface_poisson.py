# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Two-domain Poisson problem on an unfitted circular interface.

The interface Gamma = {phi = 0} splits the background mesh into an inside
domain Omega_1 = {phi < 0} and an outside domain Omega_2 = {phi > 0}. The two
fields are represented on separate background spaces and are coupled with
symmetric Nitsche terms on Gamma.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
from _pyvista_solution_plot import add_plot_arguments, plot_scalar_cut_domains

import ufl
from dolfinx import default_scalar_type, fem, io, la, mesh


def circle_level_set(center: tuple[float, float], radius: float):
    """Return a level-set function that is negative inside a circle."""

    def phi(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius

    return phi


def solve_block_system_scipy(
    a_forms: tuple[
        tuple[cutfemx.fem.CutForm, cutfemx.fem.CutForm],
        tuple[cutfemx.fem.CutForm, cutfemx.fem.CutForm],
    ],
    L_forms: tuple[cutfemx.fem.CutForm, cutfemx.fem.CutForm],
    V1: fem.FunctionSpace,
    V2: fem.FunctionSpace,
) -> tuple[fem.Function, fem.Function, cutfemx.fem.ActiveDomain, cutfemx.fem.ActiveDomain]:
    """Assemble, deactivate inactive dofs, and solve the serial 2x2 block system."""
    from scipy.sparse import bmat
    from scipy.sparse.linalg import spsolve

    if V1.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy block solve path is serial-only.")

    A_blocks = []
    for row in a_forms:
        A_row = []
        for form in row:
            A = cutfemx.fem.assemble_matrix(form)
            A.scatter_reverse()
            A_row.append(A)
        A_blocks.append(A_row)

    b_blocks = []
    for form in L_forms:
        b = cutfemx.fem.assemble_vector(form)
        b.scatter_reverse(la.InsertMode.add)
        b_blocks.append(b)

    domain1 = cutfemx.fem.active_domain(a_forms[0][0])
    domain2 = cutfemx.fem.active_domain(a_forms[1][1])
    cutfemx.fem.deactivate_outside_blocks(A_blocks, [domain1, domain2], b_blocks)

    zero_rows = cutfemx.fem.zero_block_rows(A_blocks)
    if any(rows.size > 0 for rows in zero_rows):
        detail = ", ".join(
            f"block row {i}: {rows.tolist()}" for i, rows in enumerate(zero_rows)
        )
        raise RuntimeError(f"Zero matrix rows remain after deactivation ({detail}).")

    A = bmat([[block.to_scipy().tocsr() for block in row] for row in A_blocks], format="csr")
    b = np.concatenate([block.array.copy() for block in b_blocks])
    n1 = b_blocks[0].array.size
    solution = spsolve(A, b)

    u1_h = fem.Function(V1, name="u1_h")
    u2_h = fem.Function(V2, name="u2_h")
    u1_h.x.array[:] = solution[:n1]
    u2_h.x.array[:] = solution[n1:]
    u1_h.x.scatter_forward()
    u2_h.x.scatter_forward()
    return u1_h, u2_h, domain1, domain2


def write_xdmf(
    output_dir: Path,
    cut_data: cutfemx.CutData,
    u1_h: fem.Function,
    u2_h: fem.Function,
) -> None:
    """Write the two physical-domain solutions to XDMF files."""
    comm = u1_h.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    inside_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
    outside_mesh = cutfemx.create_cut_mesh(cut_data, "phi>0", mode="full")
    if inside_mesh.mesh is None or outside_mesh.mesh is None:
        raise RuntimeError("Cannot write output for an empty inside or outside cut mesh.")

    u1_cut = cutfemx.fem.cut_function(u1_h, inside_mesh)
    u1_cut.name = "u1_h"
    u2_cut = cutfemx.fem.cut_function(u2_h, outside_mesh)
    u2_cut.name = "u2_h"

    u1_path = output_dir / "interface_poisson_u1.xdmf"
    with io.XDMFFile(comm, u1_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(inside_mesh.mesh)
        xdmf.write_function(u1_cut)

    u2_path = output_dir / "interface_poisson_u2.xdmf"
    with io.XDMFFile(comm, u2_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(outside_mesh.mesh)
        xdmf.write_function(u2_cut)

    if comm.rank == 0:
        print(f"Wrote inside solution to {u1_path}")
        print(f"Wrote outside solution to {u2_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    return parser.parse_args()


args = parse_args()
comm = MPI.COMM_WORLD

n = 24
order = 4
radius = 0.53
center = (0.05, -0.03)
kappa_1 = 1.0
kappa_2 = 8.0
gamma_interface = 40.0
gamma_boundary = 40.0
gamma_ghost = 0.1
output_dir = Path("interface_poisson_xdmf")

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

V_phi = fem.functionspace(msh, ("Lagrange", 1))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(circle_level_set(center, radius))
phi.x.scatter_forward()

cut_data = cutfemx.cut(phi)
inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
outside_cells = cutfemx.locate_entities(cut_data, "phi>0")
inside_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
outside_rules = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
ghost_facets_1 = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
ghost_facets_2 = cutfemx.ghost_penalty_facets(cut_data, "phi>0")

dx1 = ufl.Measure(
    "dx", domain=msh, subdomain_id=1, subdomain_data=[inside_cells, inside_rules]
)
dx2 = ufl.Measure(
    "dx", domain=msh, subdomain_id=2, subdomain_data=[outside_cells, outside_rules]
)
dgamma = ufl.Measure("dx", domain=msh, subdomain_id=3, subdomain_data=interface_rules)
dS_ghost_1 = ufl.Measure("dS", domain=msh, subdomain_id=4, subdomain_data=ghost_facets_1)
dS_ghost_2 = ufl.Measure("dS", domain=msh, subdomain_id=5, subdomain_data=ghost_facets_2)
ds_outer = ufl.Measure("ds", domain=msh, subdomain_id=6, subdomain_data=exterior_facets)

V1 = fem.functionspace(msh, ("Lagrange", 1))
V2 = fem.functionspace(msh, ("Lagrange", 1))
W = ufl.MixedFunctionSpace(V1, V2)
u1, u2 = ufl.TrialFunctions(W)
v1, v2 = ufl.TestFunctions(W)

x = ufl.SpatialCoordinate(msh)
r2 = (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2
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

jump_u = u1 - u2
jump_v = v1 - v2
flux_u = w1 * kappa_1 * ufl.dot(ufl.grad(u1), n_gamma) + w2 * kappa_2 * ufl.dot(
    ufl.grad(u2), n_gamma
)
flux_v = w1 * kappa_1 * ufl.dot(ufl.grad(v1), n_gamma) + w2 * kappa_2 * ufl.dot(
    ufl.grad(v2), n_gamma
)

a = kappa_1 * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1
a += kappa_2 * ufl.inner(ufl.grad(u2), ufl.grad(v2)) * dx2
a += (-flux_u * jump_v - flux_v * jump_u + eta_interface * jump_u * jump_v) * dgamma

if ghost_facets_1.size > 0:
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
if ghost_facets_2.size > 0:
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

# Strong Dirichlet conditions are awkward for this MatrixCSR block prototype, so
# impose the fitted outer boundary data for u2 by Nitsche.
a += (
    -kappa_2 * ufl.dot(ufl.grad(u2), n_facet) * v2
    - kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2
    + eta_boundary * u2 * v2
) * ds_outer

L = f1 * v1 * dx1 + f2 * v2 * dx2
L += (
    -kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2_exact
    + eta_boundary * u2_exact * v2
) * ds_outer

a_blocks = ufl.extract_blocks(a)
L_blocks = ufl.extract_blocks(L)
a_forms = tuple(tuple(cutfemx.fem.form(a_blocks[i][j]) for j in range(2)) for i in range(2))
L_forms = tuple(cutfemx.fem.form(L_blocks[i]) for i in range(2))
u1_h, u2_h, domain1, domain2 = solve_block_system_scipy(a_forms, L_forms, V1, V2)

error1 = cutfemx.fem.assemble_scalar(cutfemx.fem.form((u1_h - u1_exact) ** 2 * dx1))
error2 = cutfemx.fem.assemble_scalar(cutfemx.fem.form((u2_h - u2_exact) ** 2 * dx2))
jump_error = cutfemx.fem.assemble_scalar(cutfemx.fem.form((u1_h - u2_h) ** 2 * dgamma))
error1 = comm.allreduce(error1, op=MPI.SUM)
error2 = comm.allreduce(error2, op=MPI.SUM)
jump_error = comm.allreduce(jump_error, op=MPI.SUM)

if comm.rank == 0:
    print(f"CutFEMx two-domain interface Poisson demo, n={n}")
    print(f"inside cells          = {inside_cells.size}")
    print(f"outside cells         = {outside_cells.size}")
    print(f"interface cut cells   = {interface_rules.parent_map.size}")
    print(f"ghost facets in/out   = {ghost_facets_1.size}/{ghost_facets_2.size}")
    print(
        f"inactive dofs u1/u2   = "
        f"{domain1.inactive_dofs.size}/{domain2.inactive_dofs.size}"
    )
    print(f"L2 error u1           = {np.sqrt(max(error1, 0.0)):.6e}")
    print(f"L2 error u2           = {np.sqrt(max(error2, 0.0)):.6e}")
    print(f"interface jump norm   = {np.sqrt(max(jump_error, 0.0)):.6e}")

write_xdmf(output_dir, cut_data, u1_h, u2_h)

if comm.rank == 0 and not args.no_plot:
    plot_scalar_cut_domains(
        cut_data,
        (
            ("phi<0", u1_h, "inside"),
            ("phi>0", u2_h, "outside"),
        ),
        title="Interface Poisson solution",
        field_name="u_h",
    )
