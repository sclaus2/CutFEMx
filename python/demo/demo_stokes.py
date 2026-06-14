"""Stokes flow in a channel with an unfitted circular obstacle."""

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

import cutfemx
import numpy as np
from _pyvista_solution_plot import add_plot_arguments, plot_vector_scalar_cut_solution

import basix.ufl
import ufl
from dolfinx import default_scalar_type, fem, io, la, mesh


def cylinder_level_set(center: tuple[float, float], radius: float):
    """Return a level-set function that is positive in the fluid."""

    def phi(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius

    return phi


def traction(u, p, nu, n):
    """Return (nu grad(u) - p I) n."""
    return nu * ufl.dot(ufl.grad(u), n) - p * n


def solve_runtime_system_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    W: fem.FunctionSpace,
    bcs: list[fem.DirichletBC],
) -> fem.Function:
    """Assemble, deactivate inactive dofs, apply boundary conditions, and solve."""
    from scipy.sparse.linalg import spsolve

    if W.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy solve path is serial-only.")

    A = cutfemx.fem.assemble_matrix(a, bcs=bcs)
    A.scatter_reverse()
    b = cutfemx.fem.assemble_vector(L)
    cutfemx.fem.apply_lifting(b.array, [a], [bcs])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, bcs)
    cutfemx.fem.deactivate_outside(A, b, cutfemx.fem.active_domain(a))

    wh = fem.Function(W, name="stokes_solution")
    wh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    wh.x.scatter_forward()
    return wh


def write_xdmf(
    output_dir: Path,
    cut_data: cutfemx.CutData,
    velocity: fem.Function,
    pressure: fem.Function,
) -> None:
    """Write background and fluid-domain XDMF files."""
    comm = velocity.function_space.mesh.comm
    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    velocity_path = output_dir / "stokes_velocity_background.xdmf"
    with io.XDMFFile(comm, velocity_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(velocity.function_space.mesh)
        xdmf.write_function(velocity)

    pressure_path = output_dir / "stokes_pressure_background.xdmf"
    with io.XDMFFile(comm, pressure_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(pressure.function_space.mesh)
        xdmf.write_function(pressure)

    cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi>0", mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("Cannot write output for an empty fluid domain.")

    velocity_cut = cutfemx.fem.cut_function(velocity, cut_mesh)
    velocity_cut.name = "u"
    pressure_cut = cutfemx.fem.cut_function(pressure, cut_mesh)
    pressure_cut.name = "p"

    cut_velocity_path = output_dir / "stokes_velocity_fluid.xdmf"
    with io.XDMFFile(comm, cut_velocity_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(velocity_cut)

    cut_pressure_path = output_dir / "stokes_pressure_fluid.xdmf"
    with io.XDMFFile(comm, cut_pressure_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(cut_mesh.mesh)
        xdmf.write_function(pressure_cut)

    if comm.rank == 0:
        print(f"Wrote velocity XDMF to {velocity_path} and {cut_velocity_path}")
        print(f"Wrote pressure XDMF to {pressure_path} and {cut_pressure_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    return parser.parse_args()


args = parse_args()
comm = MPI.COMM_WORLD

n = 24
order = 4
cylinder_center = (-1.2, 0.0)
cylinder_radius = 0.3
gamma_u = 100.0
gamma_p = 0.1
gamma_g = 0.1
output_dir = Path("stokes_xdmf")

msh = mesh.create_rectangle(
    comm,
    ((-3.0, -1.0), (5.0, 1.0)),
    (4 * n, n),
    cell_type=mesh.CellType.triangle,
)

P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
P1_vec = basix.ufl.element(
    "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)
)
W = fem.functionspace(msh, basix.ufl.mixed_element([P1_vec, P1]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

level_set = fem.Function(Q, name="phi")
level_set.interpolate(cylinder_level_set(cylinder_center, cylinder_radius))
level_set.x.scatter_forward()
cut_data = cutfemx.cut(level_set)

fluid_cells = cutfemx.locate_entities(cut_data, "phi>0")
cut_cells = cutfemx.locate_entities(cut_data, "phi=0")
fluid_rules = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi>0")

active_cells = np.union1d(fluid_cells, cut_cells)
pressure_facets = cutfemx.interior_facets_for_cells(msh, active_cells)

dx_omega = ufl.Measure(
    "dx", domain=msh, subdomain_id=0, subdomain_data=[fluid_cells, fluid_rules]
)
dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=interface_rules)
dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)
dS_pressure = ufl.Measure("dS", domain=msh, subdomain_id=3, subdomain_data=pressure_facets)

w = ufl.TrialFunction(W)
u, p = ufl.split(w)
v, q = ufl.split(ufl.TestFunction(W))
nu = fem.Constant(msh, default_scalar_type(1.0))
f = fem.Constant(msh, default_scalar_type((0.0, 0.0)))
n_gamma = -cutfemx.normal(level_set)
n_facet = ufl.FacetNormal(msh)
h = ufl.CellDiameter(msh)

a = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
a += -p * ufl.div(v) * dx_omega
a += ufl.div(u) * q * dx_omega
a += -ufl.inner(traction(u, p, nu, n_gamma), v) * dx_gamma
a += -ufl.inner(traction(v, q, nu, n_gamma), u) * dx_gamma
a += gamma_u * nu / h * ufl.inner(u, v) * dx_gamma
if ghost_facets.size > 0:
    a += (
        gamma_g
        * ufl.avg(h)
        * ufl.inner(
            ufl.jump(ufl.grad(u), n_facet),
            ufl.jump(ufl.grad(v), n_facet),
        )
        * dS_ghost
    )
a += (
    gamma_p
    * ufl.avg(h) ** 3
    * ufl.inner(
        ufl.jump(ufl.grad(p), n_facet),
        ufl.jump(ufl.grad(q), n_facet),
    )
    * dS_pressure
)
L = ufl.inner(f, v) * dx_omega

inflow_u = fem.Function(V)
inflow_u.interpolate(lambda x: np.stack((1.0 - x[1] ** 2, np.zeros_like(x[0]))))
walls_u = fem.Function(V)
walls_u.x.array[:] = 0.0
outflow_p = fem.Function(Q)
outflow_p.x.array[:] = 0.0

facet_dim = msh.topology.dim - 1
inflow_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.isclose(x[0], -3.0)
)
wall_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.isclose(np.abs(x[1]), 1.0)
)
outflow_facets = mesh.locate_entities_boundary(
    msh, facet_dim, lambda x: np.isclose(x[0], 5.0)
)

inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), facet_dim, inflow_facets)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), facet_dim, wall_facets)
outflow_pressure_dofs = fem.locate_dofs_topological(
    (W.sub(1), Q), facet_dim, outflow_facets
)
bcs = [
    fem.dirichletbc(inflow_u, inflow_dofs, W.sub(0)),
    fem.dirichletbc(walls_u, wall_dofs, W.sub(0)),
    fem.dirichletbc(outflow_p, outflow_pressure_dofs, W.sub(1)),
]

wh = solve_runtime_system_scipy(cutfemx.fem.form(a), cutfemx.fem.form(L), W, bcs)
velocity = wh.sub(0).collapse()
velocity.name = "u"
pressure = wh.sub(1).collapse()
pressure.name = "p"

V_out = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
velocity_out = fem.Function(V_out, name="u")
velocity_out.interpolate(velocity)
velocity_out.x.scatter_forward()

if comm.rank == 0:
    print(f"Cut Stokes channel flow, n={n}")
    print(f"fluid cells   = {fluid_cells.size}")
    print(f"cut cells     = {cut_cells.size}")
    print(f"ghost facets  = {ghost_facets.size}")
    print(f"pressure facets = {pressure_facets.size}")
    print(f"velocity dofs = {velocity.x.array.size}")

write_xdmf(output_dir, cut_data, velocity_out, pressure)

if comm.rank == 0 and not args.no_plot:
    plot_vector_scalar_cut_solution(
        cut_data,
        "phi>0",
        velocity_out,
        pressure,
        title="Cut Stokes solution",
        vector_name="u",
        scalar_name="p",
    )
