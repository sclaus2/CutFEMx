import numpy as np
from mpi4py import MPI
import os

import cutfemx
from cutfemx.fem import (
    active_domain,
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    cut_form,
    deactivate_outside,
)

from dolfinx import fem, mesh, plot, default_scalar_type, la, io
from dolfinx.fem import Constant

import ufl
from ufl import grad, inner, div, dot, avg, jump, FacetNormal, CellDiameter, Identity, sym

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

# 1. Mesh Creation (Channel)
comm = MPI.COMM_WORLD
# x in [-2, 6], y in [-2, 2]. Cylinder at (0,0) radius 1.
# Scaled: L=4, H=2.
# Let's match demo_cut_poisson scale approx for resolution.
# [-1, 3] x [-1, 1]
N = 41
msh = mesh.create_rectangle(
    comm=comm,
    points=((-3.0, -1.0), (5.0, 1.0)),
    n=(4 * N, N),
    cell_type=mesh.CellType.triangle,
)

import basix.ufl

# 2. Function Spaces (Taylor-Hood P2-P1)
P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(msh, TH)

V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

# Trial and Test Functions
w = ufl.TrialFunction(W)
(u, p) = ufl.split(w)
(v, q) = ufl.split(ufl.TestFunction(W))

# Parameters
nu = Constant(msh, default_scalar_type(1.0))
gamma_u = 100.0 # Increased Nitsche penalty
gamma_p = 0.1 # Increased Pressure ghost penalty
gamma_g = 0.1 # Velocity ghost penalty

# 3. Level Set (Circle at (0,0), radius 0.3)
def circle(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.3

level_set = fem.Function(Q) # Level set usually scalar
level_set.interpolate(circle)
cut_data = cutfemx.cut(level_set)

# Entity location
dim = msh.topology.dim
active_entities = cutfemx.locate_entities(cut_data, "phi>0")
intersected_entities = cutfemx.locate_entities(cut_data, "phi=0") # Interface

# 4. Cut Mesh and Quadrature
# Compute normal on intersected cells for Nitsche
n_K = cutfemx.normal(level_set)

# Quadrature
order = 4
fluid_quad = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
interface_quad = cutfemx.runtime_quadrature(cut_data, "phi=0", order)

# Ghost penalty facets for velocity extension near the cut boundary.
gp_ids = cutfemx.ghost_penalty_facets(cut_data, "phi>0")

# Exclude cut cells from active_entities to avoid double integration
uncut_cells = np.setdiff1d(active_entities, intersected_entities)


# Interior facets whose two adjacent cells are in the active fluid domain.
interior_entities = np.union1d(active_entities, intersected_entities)
interior_ids = cutfemx.interior_facets_for_cells(msh, interior_entities)

# Measures
dx_omega = ufl.Measure(
    "dx",
    subdomain_id=0,
    subdomain_data=[uncut_cells, fluid_quad],
    domain=msh,
)
dx_gamma = ufl.Measure(
    "dx",
    subdomain_id=1,
    subdomain_data=interface_quad,
    domain=msh,
)
dS_ghost = ufl.Measure("dS", subdomain_id=0, subdomain_data=gp_ids, domain=msh)
dS_interior = ufl.Measure("dS", subdomain_id=1, subdomain_data=interior_ids, domain=msh)

n = FacetNormal(msh) # Physical normal
h = CellDiameter(msh)

# 5. Variational Form
# Stress tensor
def sigma(u, p):
    return nu * (grad(u) + grad(u).T) - p * Identity(dim)

# Body force
f = Constant(msh, default_scalar_type((0.0, 0.0)))

# Bulk terms
a = inner(sigma(u, p), sym(grad(v))) * dx_omega + div(u) * q * dx_omega 

# Nitsche on Interface (u=0)
n_gamma = -n_K

a += -dot(dot(sigma(u, p), n_gamma), v) * dx_gamma
a += -dot(dot(sigma(v, q), n_gamma), u) * dx_gamma # Symmetric Nitsche
a += (gamma_u / h) * inner(u, v) * dx_gamma

# Velocity ghost penalty on the cut-boundary ghost facets.
a += avg(gamma_g) * avg(h) * inner(jump(grad(u), n), jump(grad(v), n)) * dS_ghost

# Pressure stabilization on all interior facets of the active domain.
a += avg(gamma_p) * avg(h)**3 * inner(jump(grad(p), n), jump(grad(q), n)) * dS_interior

L = inner(f, v) * dx_omega

# 6. Boundary Conditions (Physical Domain)
# Inflow (x = -1)
# Wall (y = +/- 1)
# Outflow (x = 3) -> Natural BC (Do nothing -> p approx 0)

inflow_u = fem.Function(V)
def inflow_profile(x):
    return np.stack((1.0 - x[1]**2, np.zeros_like(x[0])))
inflow_u.interpolate(inflow_profile)

walls_u = fem.Function(V)
walls_u.x.array[:] = 0.0

# DOFs
fdim = msh.topology.dim - 1
# Locate facets
inflow_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], -3.0))
wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(np.abs(x[1]), 1.0))
outflow_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 5.0))

# Sub-space dofs
# BCs must be on W.sub(0).
inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)

bcs = [
    fem.dirichletbc(inflow_u, inflow_dofs, W.sub(0)),
    fem.dirichletbc(walls_u, wall_dofs, W.sub(0))
]

# Fix pressure on the outflow to remove the pressure nullspace.
p_zero = fem.Function(Q)
p_zero.x.array[:] = 0.0
p_dofs = fem.locate_dofs_topological((W.sub(1), Q), fdim, outflow_facets)
bcs.append(fem.dirichletbc(p_zero, p_dofs, W.sub(1)))

# 7. Solve
# CutForm
a_cut = cut_form(a)
L_cut = cut_form(L)

A = assemble_matrix(a_cut, bcs=bcs)
A.scatter_reverse()
b = assemble_vector(L_cut)

# Apply BCs to vector
apply_lifting(b.array, [a_cut], [bcs])
b.scatter_reverse(la.InsertMode.add)
fem.set_bc(b.array, bcs)

deactivate_outside(A, b, active_domain(a_cut))

# Solve
# Direct solver (MUMPS via PETSc or Scipy)
# Use Scipy
from scipy.sparse.linalg import spsolve

if msh.comm.size != 1:
    raise RuntimeError("The MatrixCSR/SciPy solve path in this demo is serial-only.")

# Solve
A_py = A.to_scipy()
u_sol = spsolve(A_py, b.array)

w_h = fem.Function(W)
w_h.x.array[:] = u_sol
w_h.x.scatter_forward()

# Access sub-functions for plotting (requires collapse for PyVista usually)
u_vis = w_h.sub(0).collapse()
p_vis = w_h.sub(1).collapse()
V_out = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
u_out = fem.Function(V_out, name="u")
u_out.interpolate(u_vis)

# Save for Paraview
with io.XDMFFile(msh.comm, "stokes_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_out)
with io.XDMFFile(msh.comm, "stokes_p.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(p_vis)


# 8. Visualization
is_ci = os.getenv("CI") is not None
plotter = pyvista.Plotter(shape=(1, 2), off_screen=is_ci)

# Velocity
cgrid = pyvista.UnstructuredGrid(*plot.vtk_mesh(u_vis.function_space))
cgrid.point_data["u"] = u_vis.x.array.reshape(-1, 2) # Vector
cgrid.set_active_scalars("u")

plotter.subplot(0, 0)
plotter.add_title("Velocity")
plotter.add_mesh(cgrid, scalars="u", show_edges=True)
plotter.view_xy()

# Pressure
pgrid = pyvista.UnstructuredGrid(*plot.vtk_mesh(p_vis.function_space))
pgrid.point_data["p"] = p_vis.x.array
pgrid.set_active_scalars("p")

plotter.subplot(0, 1)
plotter.add_title("Pressure")
plotter.add_mesh(pgrid, scalars="p", show_edges=True)

# Visualize Interior Facets (Red Lines)
fdim = msh.topology.dim - 1
msh.topology.create_connectivity(fdim, 0)
conn = msh.topology.connectivity(fdim, 0)
viz_cells = []
# Ensure interior_ids is accessible (computed earlier)
for f in interior_ids:
    verts = conn.links(f)
    viz_cells.append(len(verts))
    viz_cells.extend(verts)

if len(interior_ids) > 0:
    # 3 is VTK_LINE
    facet_grid = pyvista.UnstructuredGrid(viz_cells, np.full(len(interior_ids), 3, dtype=np.int8), msh.geometry.x)
    plotter.add_mesh(facet_grid, color="red", line_width=5, label="Stabilization Facets")

# Visualize Intersected Cells (Green Wireframe)
viz_intersected = []
conn_cells = msh.topology.connectivity(msh.topology.dim, 0)
for c in intersected_entities:
    verts = conn_cells.links(c)
    viz_intersected.append(len(verts))
    viz_intersected.extend(verts)

if len(intersected_entities) > 0:
    # 5 is VTK_TRIANGLE
    int_grid = pyvista.UnstructuredGrid(viz_intersected, np.full(len(intersected_entities), 5, dtype=np.int8), msh.geometry.x)
    plotter.add_mesh(int_grid, color="green", show_edges=True, opacity=0.5, label="Intersected Cells")

plotter.view_xy()

if not os.getenv("CI"):
    plotter.show()
else:
    plotter.screenshot("stokes_solution.png")
