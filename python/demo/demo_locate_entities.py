import numpy as np
from mpi4py import MPI

from cutfemx import cutfemx_cpp as _cpp

from dolfinx.cpp.fem import Function_float64

from dolfinx import fem, io, mesh, plot

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(11, 11),
    cell_type=mesh.CellType.triangle,
)

V = fem.functionspace(msh, ("Lagrange", 1))

def circle(x):
  return np.sqrt((x[0]-0.5)**2+(x[1]-0.5)**2)-0.2

level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.geometry.dim

entities = _cpp.level_set.locate_entities(level_set._cpp_object,dim,"phi=0")

print("intersected entities=", entities)


# To visualize the function u, we create a VTK-compatible grid to
# values of u to
cells, types, x = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.point_data["ls"] = level_set.x.array

# The function "u" is set as the active scalar for the mesh, and
# warp in z-direction is set
grid.set_active_scalars("ls")
warped = grid.warp_by_scalar()

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
plotter.view_xy()

plotter.show()