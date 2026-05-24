import numpy as np
from mpi4py import MPI
import os

import cutfemx
from cutfemx.fem import (
  active_domain,
  assemble_vector,
  cut_form,
  assemble_matrix,
  deactivate_outside,
  cut_function,
)

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
cut_data = cutfemx.cut(level_set)
dim = msh.topology.dim

intersected_entities = cutfemx.locate_entities(cut_data, "phi=0")
inside_entities = cutfemx.locate_entities(cut_data, "phi<0")

n_K = cutfemx.normal(level_set)

order = 2
inside_quadrature = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi=0", order)

quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

gp_ids = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_id=0, subdomain_data=gp_ids, domain=msh)
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)
dsq = dx_rt(1)

a  = inner(grad(u), grad(v))*dxq

# cut cells on surface
a += - dot(grad(u), n_K)*v*dsq
a += - dot(grad(v), n_K)*u*dsq
a += gamma*1.0/h*u*v*dsq
a += avg(gamma_g)*avg(h)*dot(jump(grad(u), n), jump(grad(v), n))*dS

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
  cut_data.update()

  intersected_entities = cutfemx.locate_entities(cut_data, "phi=0")
  inside_entities = cutfemx.locate_entities(cut_data, "phi<0")

  gp_ids = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

  inside_quadrature = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
  interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi=0", order)

  subdomain_data={"cell": [(0, inside_entities)], "interior_facet": [(0, gp_ids)]}
  quad_domains = {"cutcell": [(0,inside_quadrature), (1,interface_quadrature)]}
  subdomain_data_L={"cell": [(0, inside_entities)]}

  a_cut.update_integration_domains(subdomain_data)
  a_cut.update_runtime_domains(quad_domains)

  L_cut.update_integration_domains(subdomain_data_L)
  L_cut.update_runtime_domains(quad_domains)

  b = assemble_vector(L_cut)
  A = assemble_matrix(a_cut)
  deactivate_outside(A, active_domain(a_cut))
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
