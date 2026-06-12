# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Linear elasticity on a 3D cut dogbone cylinder."""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, io, la, mesh


def dogbone_level_set(
    half_length: float,
    grip_radius: float,
    waist_radius: float,
    gauge_half_length: float,
):
    """Return a smooth dogbone-cylinder level set, negative inside the solid."""

    def phi(x: np.ndarray) -> np.ndarray:
        t = (np.abs(x[0]) - gauge_half_length) / (half_length - gauge_half_length)
        t = np.clip(t, 0.0, 1.0)
        shoulder = t * t * (3.0 - 2.0 * t)
        radius = waist_radius + (grip_radius - waist_radius) * shoulder
        return np.sqrt(x[1] ** 2 + x[2] ** 2) - radius

    return phi


def epsilon(u):
    """Return the small-strain tensor."""
    return ufl.sym(ufl.grad(u))


def sigma(u, mu: float, lmbda: float, gdim: int):
    """Return the isotropic linear-elastic stress tensor."""
    eps = epsilon(u)
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)


def von_mises_3d(u, mu: float, lmbda: float):
    """Return the 3D von Mises stress expression."""
    stress = sigma(u, mu, lmbda, 3)
    return ufl.sqrt(
        0.5
        * (
            (stress[0, 0] - stress[1, 1]) ** 2
            + (stress[1, 1] - stress[2, 2]) ** 2
            + (stress[2, 2] - stress[0, 0]) ** 2
        )
        + 3.0 * (stress[0, 1] ** 2 + stress[1, 2] ** 2 + stress[2, 0] ** 2)
    )


def solve_runtime_system_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
    bcs: list[fem.DirichletBC],
) -> tuple[fem.Function, cutfemx.fem.ActiveDomain]:
    """Assemble, deactivate inactive dofs, apply boundary conditions, and solve."""
    from scipy.sparse.linalg import spsolve

    if V.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy solve path is serial-only.")

    A = cutfemx.fem.assemble_matrix(a, bcs=bcs)
    A.scatter_reverse()
    b = cutfemx.fem.assemble_vector(L)
    cutfemx.fem.apply_lifting(b.array, [a], [bcs])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, bcs)

    active = cutfemx.fem.active_domain(a)
    cutfemx.fem.deactivate_outside(A, b, active)

    uh = fem.Function(V, name="deformation")
    uh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    uh.x.scatter_forward()
    return uh, active


def compute_von_mises_on_background_mesh(
    u: fem.Function,
    mu: float,
    lmbda: float,
) -> fem.Function:
    """Interpolate one von Mises value per background cell."""
    msh = u.function_space.mesh
    Q = fem.functionspace(msh, ("Discontinuous Lagrange", 0))
    stress = fem.Function(Q, name="von_mises")
    interpolation_points = Q.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    stress.interpolate(fem.Expression(von_mises_3d(u, mu, lmbda), interpolation_points))
    return stress


def write_xdmf(
    output_dir: Path,
    cut_data: cutfemx.CutData,
    displacement: fem.Function,
    von_mises: fem.Function,
) -> None:
    """Write background and cut-solid XDMF files."""
    comm = displacement.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    displacement_path = output_dir / "elasticity_deformation_background.xdmf"
    with io.XDMFFile(comm, displacement_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(displacement.function_space.mesh)
        xdmf.write_function(displacement)

    stress_path = output_dir / "elasticity_von_mises_background.xdmf"
    with io.XDMFFile(comm, stress_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(von_mises.function_space.mesh)
        xdmf.write_function(von_mises)

    cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("Cannot write output for an empty solid domain.")

    displacement_cut = cutfemx.fem.cut_function(displacement, cut_mesh)
    displacement_cut.name = "deformation"
    stress_cut = cutfemx.fem.cut_function(von_mises, cut_mesh)
    stress_cut.name = "von_mises"

    cut_displacement_path = output_dir / "elasticity_deformation_solid.xdmf"
    with io.XDMFFile(comm, cut_displacement_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(displacement_cut)

    cut_stress_path = output_dir / "elasticity_von_mises_solid.xdmf"
    with io.XDMFFile(comm, cut_stress_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(stress_cut)

    if comm.rank == 0:
        print(f"Wrote deformation XDMF to {displacement_path} and {cut_displacement_path}")
        print(f"Wrote von Mises XDMF to {stress_path} and {cut_stress_path}")


comm = MPI.COMM_WORLD

n = 10
order = 4
young_modulus = 1.0e3
poisson_ratio = 0.3
pull = 0.12
gamma_ghost = 0.05
output_dir = Path("elasticity_xdmf")

half_length = 1.5
grip_radius = 0.42
waist_radius = 0.22
gauge_half_length = 0.45
box_radius = 0.50

msh = mesh.create_box(
    comm,
    (
        np.array([-half_length, -box_radius, -box_radius]),
        np.array([half_length, box_radius, box_radius]),
    ),
    (4 * n, n, n),
    cell_type=mesh.CellType.tetrahedron,
)

V_phi = fem.functionspace(msh, ("Lagrange", 1))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(
    dogbone_level_set(
        half_length,
        grip_radius,
        waist_radius,
        gauge_half_length,
    )
)
phi.x.scatter_forward()

cut_data = cutfemx.cut(phi)
solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
solid_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

dx_solid = ufl.Measure(
    "dx", domain=msh, subdomain_id=0, subdomain_data=[solid_cells, solid_rules]
)
dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=1, subdomain_data=ghost_facets)

V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
gdim = msh.geometry.dim
mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
lmbda = young_modulus * poisson_ratio / (
    (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
)

a = ufl.inner(sigma(u, mu, lmbda, gdim), epsilon(v)) * dx_solid
if ghost_facets.size > 0:
    n_facet = ufl.FacetNormal(msh)
    h_avg = ufl.avg(ufl.CellDiameter(msh))
    a += (
        gamma_ghost
        * (2.0 * mu + lmbda)
        * h_avg
        * ufl.inner(
            ufl.jump(ufl.grad(u), n_facet),
            ufl.jump(ufl.grad(v), n_facet),
        )
        * dS_ghost
    )

zero_force = fem.Constant(msh, default_scalar_type((0.0, 0.0, 0.0)))
L = ufl.inner(zero_force, v) * dx_solid

facet_dim = msh.topology.dim - 1
left_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.isclose(x[0], -half_length)
)
right_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.isclose(x[0], half_length)
)
u_left = fem.Function(V)
left_dofs = fem.locate_dofs_topological(V, facet_dim, left_facets)
bcs = [fem.dirichletbc(u_left, left_dofs)]

Vx, _ = V.sub(0).collapse()
u_right_x = fem.Function(Vx)
u_right_x.interpolate(lambda x: pull * np.ones_like(x[0]))
right_x_dofs = fem.locate_dofs_topological((V.sub(0), Vx), facet_dim, right_facets)
bcs.append(fem.dirichletbc(u_right_x, right_x_dofs, V.sub(0)))

uh, active = solve_runtime_system_scipy(cutfemx.fem.form(a), cutfemx.fem.form(L), V, bcs)
von_mises = compute_von_mises_on_background_mesh(uh, mu, lmbda)

displacement_norm = float(np.max(np.linalg.norm(uh.x.array.reshape((-1, gdim)), axis=1)))
max_stress = float(np.max(von_mises.x.array)) if von_mises.x.array.size else 0.0

if comm.rank == 0:
    print(f"Cut dogbone elasticity problem, cells={4 * n} x {n} x {n}")
    print(f"solid cells   = {solid_cells.size}")
    print(f"cut cells     = {solid_rules.parent_map.size}")
    print(f"ghost facets  = {ghost_facets.size}")
    print(f"inactive dofs = {active.inactive_dofs.size}")
    print(f"max |u|       = {displacement_norm:.6e}")
    print(f"max sigma_vm  = {max_stress:.6e}")

write_xdmf(output_dir, cut_data, uh, von_mises)
