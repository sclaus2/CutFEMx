import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities
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

V = fem.functionspace(msh, ("Lagrange", 1))

def circle(x):
  return np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)-0.2

level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.topology.dim

intersected_entities = locate_entities(level_set,dim,"phi=0")
inside_entities = locate_entities(level_set,dim,"phi<0")

# To visualize the function u, we create a VTK-compatible grid to
# values of u to
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["ls"] = level_set.x.array

# The function "u" is set as the active scalar for the mesh, and
# warp in z-direction is set
grid.set_active_scalars("ls")
warped = grid.warp_by_scalar()

plotter = pyvista.Plotter(shape=(2, 2))
plotter.subplot(0, 0)
plotter.add_text("Level Set Function", font_size=14, color="black", position="upper_edge")
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)

#plot entities
cells, types, x = plot.vtk_mesh(msh, dim, intersected_entities)
sub_grid = pyvista.UnstructuredGrid(cells, types, x)
plotter.subplot(0, 1)
plotter.add_text("Intersected Elements", font_size=14, color="black", position="upper_edge")
plotter.add_mesh(sub_grid, show_edges=True, edge_color="black")
plotter.view_xy()

cells, types, x = plot.vtk_mesh(msh, dim, inside_entities)
sub_grid = pyvista.UnstructuredGrid(cells, types, x)
plotter.subplot(1, 0)
plotter.add_text("Elements Inside", font_size=14, color="black", position="upper_edge")
plotter.add_mesh(sub_grid, show_edges=True, edge_color="black")
plotter.view_xy()

plotter.show()