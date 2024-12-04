import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points

from dolfinx import fem, mesh, plot

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

N = 15

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle,
)
gdim = 2
tdim = msh.topology.dim

V = fem.functionspace(msh, ("Lagrange", 1))

def circle(x):
  return np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)-0.2

level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.topology.dim

intersected_entities = locate_entities(level_set,dim,"phi=0")
inside_entities = locate_entities(level_set,dim,"phi<0")

dof_coordinates = V.tabulate_dof_coordinates()

cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")

num_local_cells = msh.topology.index_map(tdim).size_local
cut_mesh = create_cut_mesh(msh.comm,cut_cells,msh,inside_entities)

order = 2
runtime_quadrature = runtime_quadrature(level_set,"phi<0",order)

#Plot runtime quadrature points on cut mesh
points_phys = physical_points(runtime_quadrature,msh)

plotter = pyvista.Plotter()
cells, types, x = plot.vtk_mesh(cut_mesh._mesh)
grid = pyvista.UnstructuredGrid(cells, types, x)
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, color='lightgray')
plotter.add_points(points_phys, render_points_as_spheres=True, color='black')
plotter.view_xy()
plotter.save_graphic("quadrature.svg")
plotter.show()
