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

#from dolfinx.fem.assemble import assemble_vector, assemble_matrix
from dolfinx.fem import form

import ufl
from ufl import dS, dx, grad, inner, dC, FacetNormal, CellDiameter, dot, avg, jump

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

from scipy.sparse.linalg import spsolve

N = 11

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (4.0, 1.0)),
    n=(5*N, N),
    cell_type=mesh.CellType.triangle,
)
gdim = 2
tdim = msh.topology.dim

V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, default_real_type(1.0))
g = fem.Constant(msh, default_real_type(0.0))
gamma = 100
gamma_g = 1e-1

n = FacetNormal(msh)
h = CellDiameter(msh)

xM = np.array([-1.0,0.0])

def circle(x):
  return np.sqrt((x[0]-xM[0])**2+(x[1]-xM[1])**2)-0.5

level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.topology.dim

intersected_entities = locate_entities(level_set,dim,"phi=0")
inside_entities = locate_entities(level_set,dim,"phi<0")

V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K,level_set,intersected_entities)

dof_coordinates = V.tabulate_dof_coordinates()

cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm,cut_cells,msh,inside_entities)

order = 2
inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

gp_ids =  ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh,gp_ids)

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)
dsq = dx_rt(1)

a  = inner(grad(u), grad(v))*dxq

# cut cells on surface
a += - dot(grad(u), n_K)*v*dsq
a += - dot(grad(v), n_K)*u*dsq
a += gamma*1.0/h*u*v*dsq
a += avg(gamma_g)*avg(h)*dot(jump(grad(u), n), jump(grad(v), n))*dS(0)

L  = f*v*dxq
L += - dot(grad(v), n_K)*g*dsq
L +=  gamma*1.0/h*v*g*dsq

a_cut = cut_form(a)
L_cut = cut_form(L)

T = 3.0
dt = 1.0
t = 0

u_cut = []
u_sol_array = []
steps = 0

while t<T:
  print("Step:", steps)
  dx = np.array([1.0,0.0])

  #update circle midpoint
  xM = xM + dx
  level_set.interpolate(circle)

  intersected_entities = locate_entities(level_set,dim,"phi=0")
  inside_entities = locate_entities(level_set,dim,"phi<0")

  gp_ids =  ghost_penalty_facets(level_set, "phi<0")
  gp_topo = facet_topology(msh,gp_ids)

  inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
  interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

  subdomain_data={"cell": [(0, inside_entities)], "interior_facet": [(0, gp_topo)]}
  quad_domains = {"cutcell": [(0,inside_quadrature), (1,interface_quadrature)]}

  a_cut.update_integration_domains(subdomain_data)
  a_cut.update_runtime_domains(quad_domains)

  L_cut.update_integration_domains(subdomain_data)
  L_cut.update_runtime_domains(quad_domains)

  compute_normal(n_K,level_set,intersected_entities)

  b = assemble_vector(L_cut)
  A = assemble_matrix(a_cut)
  deactivate(A,"phi>0",level_set,[V])
  A_py = A.to_scipy()

  u_sol = spsolve(A_py, b.array)

  u_sol_array.append(u_sol)

  t = t+dt
  steps = steps+1

plotter = pyvista.Plotter(shape=(1, 3))

cells, types, x = plot.vtk_mesh(msh)
grid = pyvista.UnstructuredGrid(cells, types, x)

plotter.subplot(0, 0)
grid.point_data["u"] = u_sol_array[0]
grid.set_active_scalars("u")
warped = grid.warp_by_scalar(scale_factor=10.0)
plotter.add_mesh(warped)

plotter.subplot(0, 1)
grid.point_data["u"] = u_sol_array[1]
warped2 = grid.warp_by_scalar(scale_factor=10.0)
plotter.add_mesh(warped2)

plotter.subplot(0, 2)
grid.point_data["u"] = u_sol_array[2]
warped3 = grid.warp_by_scalar(scale_factor=10.0)
plotter.add_mesh(warped3)

plotter.show()

