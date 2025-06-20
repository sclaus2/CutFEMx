import numpy as np
from mpi4py import MPI
import os

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import assemble_vector, cut_form, assemble_matrix, deactivate, cut_function

from dolfinx import fem, mesh, plot
from dolfinx import default_real_type

# from dolfinx.fem.assemble import assemble_vector, assemble_matrix
from dolfinx.fem import form

import ufl
from ufl import dS, dx, grad, inner, dc, FacetNormal, CellDiameter, dot, avg, jump

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

N = 21

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle,
)
gdim = 2
tdim = msh.topology.dim

V = fem.functionspace(msh, ("Lagrange", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, default_real_type(1.0))
g = fem.Constant(msh, default_real_type(0.0))
gamma = 100
gamma_g = 1e-1

n = FacetNormal(msh)
h = CellDiameter(msh)


def circle(x):
    return np.sqrt((x[0]) ** 2 + (x[1]) ** 2) - 0.5


level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.topology.dim

intersected_entities = locate_entities(level_set, dim, "phi=0")
inside_entities = locate_entities(level_set, dim, "phi<0")

V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K, level_set, intersected_entities)

dof_coordinates = V.tabulate_dof_coordinates()

cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)

order = 1
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)

quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]

gp_ids = ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh, gp_ids)

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)
dsq = dx_rt(1)

a = inner(grad(u), grad(v)) * dxq

# cut cells on surface
a += -dot(grad(u), n_K) * v * dsq
a += -dot(grad(v), n_K) * u * dsq
a += gamma * 1.0 / h * u * v * dsq
a += avg(gamma_g) * avg(h) * dot(jump(grad(u), n), jump(grad(v), n)) * dS(0)

L = f * v * dxq
L += -dot(grad(v), n_K) * g * dsq
L += gamma * 1.0 / h * v * g * dsq

a_cut = cut_form(a)
L_cut = cut_form(L)

b = assemble_vector(L_cut)
A = assemble_matrix(a_cut)
deactivate(A, "phi>0", level_set, [V])
A_py = A.to_scipy()

from scipy.sparse.linalg import spsolve

u_sol = spsolve(A_py, b.array)

u_func = fem.Function(V)
u_func.x.array[:] = u_sol
u_cut = cut_function(u_func, cut_mesh)

# Plot runtime quadrature points on cut mesh
points_phys = physical_points(inside_quadrature, msh)

# Create subplots for comprehensive visualization
plotter = pyvista.Plotter(shape=(1, 3))

# Prepare mesh data
cells, types, x = plot.vtk_mesh(V)
facets, ftypes, fx = plot.vtk_mesh(msh, tdim - 1, gp_ids)
ccells, ctypes, cx = plot.vtk_mesh(cut_mesh._mesh)

grid = pyvista.UnstructuredGrid(cells, types, x)
fgrid = pyvista.UnstructuredGrid(facets, ftypes, fx)
cgrid = pyvista.UnstructuredGrid(ccells, ctypes, cx)

# Set scalar data
grid.point_data["u"] = u_sol
grid.set_active_scalars("u")

# Subplot 1: Cut mesh with quadrature points
plotter.subplot(0, 0)
plotter.add_mesh(cgrid, show_edges=True, show_scalar_bar=True, color="lightgray")
plotter.add_points(points_phys, render_points_as_spheres=True, color="black", point_size=8)
plotter.add_title("Cut Mesh with Quadrature Points")
plotter.view_xy()

# Subplot 2: Overview with all meshes
plotter.subplot(0, 1)
plotter.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.7, label="Background Mesh")
plotter.add_mesh(cgrid, show_edges=True, color="blue", opacity=0.9, label="Cut Mesh")
plotter.add_mesh(fgrid, show_edges=True, line_width=5, color="red", label="Ghost Penalty Facets")
plotter.add_title("Ghost Penalty Facets and Cut Mesh")
plotter.view_xy()

# Subplot 3: Full mesh solution (warped)
plotter.subplot(0, 2)
warped = grid.warp_by_scalar(scale_factor=10.0)
plotter.add_mesh(warped, scalars="u", show_edges=True, show_scalar_bar=True)
plotter.add_title("Full Mesh Solution (Warped)")
# Isometric view for better 3D perspective of mesh relationships
plotter.camera_position = "iso"


# Optional: Add interactive widgets for better 3D exploration
# plotter.add_axes()  # Add coordinate axes


plotter.show()
