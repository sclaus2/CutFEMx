import numpy as np
from mpi4py import MPI
import os

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology, interior_facets
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import assemble_vector, cut_form, assemble_matrix, deactivate, cut_function

from dolfinx import fem, mesh, plot, default_scalar_type, la, io
from dolfinx.fem import Constant

import ufl
from ufl import dS, dx, grad, inner, div, dot, avg, jump, FacetNormal, CellDiameter, Identity, sym

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
P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim, ))
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

# Entity location
dim = msh.topology.dim
intersected_entities = locate_entities(level_set, dim, "phi=0")
inside_entities = locate_entities(level_set, dim, "phi<0") # Fluid is OUTSIDE the circle?
# Wait, "flow around a cylinder". Fluid is in domain AND r > R.
# So "phi > 0" is fluid. Circle is obstacle (phi < 0).
# Let's conform to standard: fluid is active domain.
# If circle defined as sqrt(x^2+y^2) - R
# Inside circle (obstacle): phi < 0.
# Outside circle (fluid): phi > 0.
# So we locate "phi>0".

active_entities = locate_entities(level_set, dim, "phi>0")
intersected_entities = locate_entities(level_set, dim, "phi=0") # Interface

# 4. Cut Mesh and Quadrature
# Compute normal on intersected cells for Nitsche
V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K, level_set, intersected_entities)

# cut_entities needs dof coords to determine which cells are cut for specific basis?
# Actually for mesh cutting, usually geometric checks.
# But `cut_entities` checks dof placement relative to isosurface.
# Using Q (P1) coords for level set is consistent with level_set function.
Q_coords = Q.tabulate_dof_coordinates()
cut_cells = cut_entities(level_set, Q_coords, intersected_entities, dim, "phi>0")

cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, active_entities)

# Quadrature
order = 2 # P2 elements -> order 4 needed? 2*deg.
fluid_quad = runtime_quadrature(level_set, "phi>0", order)
interface_quad = runtime_quadrature(level_set, "phi=0", order)

quad_domains = [(0, fluid_quad), (1, interface_quad)]

# Ghost Penalty Facets
gp_ids = ghost_penalty_facets(level_set, "phi>0")
gp_topo = facet_topology(msh, gp_ids)

# Exclude cut cells from active_entities to avoid double integration
uncut_cells = np.setdiff1d(active_entities, cut_cells)


# Interior Facets (Fluid + Ghost Penalty)
interior_entities = np.union1d(active_entities, intersected_entities)
interior_ids = interior_facets(level_set, interior_entities, "phi>0")
interior_topo = facet_topology(msh, interior_ids)

# Measures
dx = ufl.Measure("dx", subdomain_data=[(0, uncut_cells)], domain=msh)
dS_gp = ufl.Measure("dS", subdomain_data=[(0, gp_topo),(1, interior_topo)], domain=msh) # For Ghost Penalty For Pressure CIP
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0) # Fluid domain (uncut + cut)
dsq = dx_rt(1)       # Interface

n = FacetNormal(msh) # Physical normal
h = CellDiameter(msh)

# 5. Variational Form
# Stress tensor
def sigma(u, p):
    return nu * (grad(u) + grad(u).T) - p * Identity(dim)

# Body force
f = Constant(msh, default_scalar_type((0.0, 0.0)))

# Bulk terms
a = inner(sigma(u, p), sym(grad(v))) * dxq + div(u) * q * dxq 

# Nitsche on Interface (u=0)
n_gamma = -n_K

a += -dot(dot(sigma(u, p), n_gamma), v) * dsq
a += -dot(dot(sigma(v, q), n_gamma), u) * dsq # Symmetric Nitsche
a += (gamma_u / h) * inner(u, v) * dsq

# Ghost Penalty (Velocity - Cut Region)
a += avg(gamma_g) * avg(h) * inner(jump(grad(u), n), jump(grad(v), n)) * dS_gp(0)

# Pressure Consistent Interior Penalty (Stabilization for P1-P1) - Global
# Apply on active interior facets (dS_fluid) to allow pressure deactivation
a += avg(gamma_p) * avg(h)**3 * inner(jump(grad(p), n), jump(grad(q), n)) * dS_gp(1)

# Add L2 Regularization (soft-fix for pressure modes/nullspace)
a += 1e-8 * inner(p, q) * dxq

L = inner(f, v) * dxq

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

# Sub-space dofs
# BCs must be on W.sub(0).
inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)

bcs = [
    fem.dirichletbc(inflow_u, inflow_dofs, W.sub(0)),
    fem.dirichletbc(walls_u, wall_dofs, W.sub(0))
]

# Fix pressure at one point (to remove nullspace if Outflow is not sufficient)
p_zero = fem.Function(Q)
p_zero.x.array[:] = 0.0
# Locate a point dof for pressure at Outflow (3.0, 1.0)
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 5.0) & np.isclose(x[1], 1.0))
if len(p_dofs) > 0:
    bcs.append(fem.dirichletbc(p_zero, p_dofs, W.sub(1)))

# 7. Solve
# CutForm
a_cut = cut_form(a)
L_cut = cut_form(L)

A = assemble_matrix(a_cut, bcs=bcs)
b = assemble_vector(L_cut)

# Apply BCs to vector
fem.apply_lifting(b.array, [a_cut._form], [bcs])
b.scatter_reverse(la.InsertMode.add)
fem.set_bc(b.array, bcs)

# Deactivation Strategy:
deactivate(A, "phi<0", level_set, [W, 0], diagonal=1.0)
deactivate(A, "phi<0", level_set, [W, 1], diagonal=1.0)

# Solve
# Direct solver (MUMPS via PETSc or Scipy)
# Use Scipy
from scipy.sparse.linalg import spsolve
# Solve
A_py = A.to_scipy()
u_sol = spsolve(A_py, b.array)

w_h = fem.Function(W)
w_h.x.array[:] = u_sol

# Split
(u_h, p_h) = w_h.split() 
# Access sub-functions for plotting (requires collapse for PyVista usually)
u_vis = w_h.sub(0).collapse()
p_vis = w_h.sub(1).collapse()

# Save for Paraview
with io.VTXWriter(msh.comm, "stokes_u.bp", [u_vis], engine="BP4") as vtx:
    vtx.write(0.0)
with io.VTXWriter(msh.comm, "stokes_p.bp", [p_vis], engine="BP4") as vtx:
    vtx.write(0.0)


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
