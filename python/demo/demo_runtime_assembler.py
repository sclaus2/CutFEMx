import numpy as np
from mpi4py import MPI
import os

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import assemble_scalar, cut_form

from dolfinx import fem, mesh, plot
from dolfinx import default_real_type

import ufl

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

V = fem.functionspace(msh, ("Lagrange", 1))

def circle(x):
  return np.sqrt((x[0])**2+(x[1])**2)-0.5

level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.topology.dim

intersected_entities = locate_entities(level_set,dim,"phi=0")
inside_entities = locate_entities(level_set,dim,"phi<0")

dof_coordinates = V.tabulate_dof_coordinates()

cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm,cut_cells,msh,inside_entities)

order = 2
inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

alpha = fem.Constant(msh, default_real_type(1.0))

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)])
dx_rt = ufl.Measure("dx", metadata={"quadrature_rule":"runtime"} , subdomain_data=quad_domains)

dxq = dx_rt(0) + dx(0)
dsq = dx_rt(1)

L_area = alpha*dxq

#jit_options = {"cache_dir": os.getcwd()}
L_area_form = cut_form(L_area, jit_options=jit_options)

area = assemble_scalar(L_area_form)

print("area=", area)
print("value theoretical area=", 0.5*0.5*np.pi)

L_s = alpha*dsq
L_s_form = cut_form(L_s, jit_options=jit_options)
surface = assemble_scalar(L_s_form)

print("circumference=", surface)
print("value theoretical circumference=", 2*0.5*np.pi)

#Plot runtime quadrature points on cut mesh
points_phys = physical_points(inside_quadrature,msh)
points_phys_interface = physical_points(interface_quadrature,msh)

plotter = pyvista.Plotter()
cells, types, x = plot.vtk_mesh(cut_mesh._mesh)
grid = pyvista.UnstructuredGrid(cells, types, x)
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, color='lightgray')
plotter.add_points(points_phys, render_points_as_spheres=True, color='black')
plotter.add_points(points_phys_interface, render_points_as_spheres=True, color='red')
plotter.view_xy()
plotter.show()

