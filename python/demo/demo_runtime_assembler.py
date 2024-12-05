import numpy as np
from mpi4py import MPI
import os

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form, assemble_scalar

from dolfinx import fem, mesh, plot
from dolfinx import default_real_type

import ufl

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

from cutfemx import cutfemx_cpp as _cpp


N = 31

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
runtime_quadrature = runtime_quadrature(level_set,"phi<0",order)

alpha = fem.Constant(msh, default_real_type(1.0))

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)])
dx_rt = ufl.Measure("dx", metadata={"quadrature_rule":"runtime"} , subdomain_data=[(0, runtime_quadrature)]) #

dxq = dx_rt(0) + dx(0)

L = alpha*dxq 

jit_options = {"cache_dir": os.getcwd()}
L_form = cut_form(L, jit_options=jit_options)

val = assemble_scalar(L_form)

print("value=", val)
print("value theoretical=", 0.5*0.5*np.pi)
