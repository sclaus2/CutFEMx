# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Poisson problem on a moving unfitted circular domain."""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import default_real_type, fem, io, la, mesh


def circle_level_set(center: tuple[float, float], radius: float):
    """Return a level-set function that is negative inside a moving circle."""

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


def solve_step(
    msh: mesh.Mesh,
    V: fem.FunctionSpace,
    phi: fem.Function,
    center: tuple[float, float],
    radius: float,
    order: int,
    gamma: float,
    gamma_g: float,
) -> tuple[cutfemx.CutData, fem.Function, np.ndarray, np.ndarray, np.ndarray]:
    """Update the level set, rebuild cut data, and solve one Poisson problem."""
    phi.interpolate(circle_level_set(center, radius))
    phi.x.scatter_forward()

    cut_data = cutfemx.cut(phi)
    inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
    cut_cells = cutfemx.locate_entities(cut_data, "phi=0")
    inside_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    source = fem.Constant(msh, default_real_type(1.0))
    boundary_value = fem.Constant(msh, default_real_type(0.0))
    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    dx_omega = ufl.Measure(
        "dx", domain=msh, subdomain_id=0, subdomain_data=[inside_cells, inside_rules]
    )
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=interface_rules)
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += -ufl.dot(ufl.grad(u), n_gamma) * v * dx_gamma
    a += -ufl.dot(ufl.grad(v), n_gamma) * u * dx_gamma
    a += gamma / h * u * v * dx_gamma
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

    L = source * v * dx_omega
    L += -ufl.dot(ufl.grad(v), n_gamma) * boundary_value * dx_gamma
    L += gamma / h * v * boundary_value * dx_gamma

    uh = solve_runtime_system_scipy(cutfemx.fem.form(a), cutfemx.fem.form(L), V)
    return cut_data, uh, inside_cells, cut_cells, ghost_facets


def write_step_xdmf(
    output_dir: Path,
    step: int,
    cut_data: cutfemx.CutData,
    phi: fem.Function,
    uh: fem.Function,
) -> None:
    """Write background and cut-domain fields for one time step."""
    comm = uh.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    background_path = output_dir / f"moving_poisson_background_{step:02d}.xdmf"
    with io.XDMFFile(comm, background_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(uh.function_space.mesh)
        xdmf.write_function(phi)
        xdmf.write_function(uh)

    cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("Cannot write output for an empty cut domain.")

    phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)
    phi_cut.name = "phi"
    uh_cut = cutfemx.fem.cut_function(uh, cut_mesh)
    uh_cut.name = "u_h"

    cut_path = output_dir / f"moving_poisson_cut_domain_{step:02d}.xdmf"
    with io.XDMFFile(comm, cut_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(phi_cut)
        xdmf.write_function(uh_cut)


comm = MPI.COMM_WORLD

n = 16
steps = 4
order = 2
radius = 0.5
gamma = 100.0
gamma_g = 0.1
output_dir = Path("moving_poisson_xdmf")

msh = mesh.create_rectangle(
    comm,
    ((-1.0, -1.0), (4.0, 1.0)),
    (5 * n, n),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))
phi = fem.Function(V, name="phi")

for step in range(steps):
    center = (float(step), 0.0)
    cut_data, uh, inside_cells, cut_cells, ghost_facets = solve_step(
        msh,
        V,
        phi,
        center,
        radius,
        order,
        gamma,
        gamma_g,
    )
    uh.name = f"u_h_{step:02d}"
    write_step_xdmf(output_dir, step, cut_data, phi, uh)

    if comm.rank == 0:
        print(
            f"step {step}: center={center}, inside={inside_cells.size}, "
            f"cut={cut_cells.size}, ghost={ghost_facets.size}"
        )
