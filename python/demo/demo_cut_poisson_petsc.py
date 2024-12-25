import os
import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs

from dolfinx import fem, mesh, plot

#from dolfinx.fem.assemble import assemble_vector, assemble_matrix
from dolfinx.fem import form
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode

import ufl
from ufl import dS, dx, grad, inner, dc, FacetNormal, CellDiameter, dot, avg, jump

try:
    from petsc4py import PETSc

    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
except ModuleNotFoundError:
    print("This demo requires petsc4py.")
    exit(0)

try:
    import pyvista
except ModuleNotFoundError:
    print("pyvista is required for this demo")
    exit(0)

N = 11

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle,
    ghost_mode=GhostMode.shared_facet,
)
gdim = 2
tdim = msh.topology.dim

V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, PETSc.ScalarType(1.0))
g = fem.Constant(msh, PETSc.ScalarType(0.0))
gamma = 100
gamma_g = 1e-1

n = FacetNormal(msh)
h = CellDiameter(msh)

def circle(x):
    #return x[0]+1e-8
    return np.sqrt((x[0])**2+(x[1])**2)-0.5

level_set = fem.Function(V)
level_set.interpolate(circle)
dim = msh.topology.dim

intersected_entities = locate_entities(level_set,dim,"phi=0")
inside_entities = locate_entities(level_set,dim,"phi<0")

cell_map = msh.topology.index_map(tdim)
size_local = cell_map.size_local 
num_ghosts = cell_map.num_ghosts

print("num_cells=", size_local)
print(intersected_entities)

V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K,level_set,intersected_entities)
#n_K = fem.Constant(msh, (PETSc.ScalarType(1.0),PETSc.ScalarType(0.0)))

dof_coordinates = V.tabulate_dof_coordinates()

cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm,cut_cells,msh,inside_entities)
interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
interface_mesh = create_cut_cells_mesh(msh.comm,size_local,interface_cells)

order = 2
inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

print("inside parent map=", inside_quadrature.parent_map)
print("interface parent map=", interface_quadrature.parent_map)

quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

gp_ids =  ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh,gp_ids)

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities),(2, intersected_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dx", metadata={"quadrature_rule":"runtime"} , subdomain_data=quad_domains, domain=msh)

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

jit_options = {"cache_dir": os.getcwd()}
a_cut = cut_form(a,  jit_options=jit_options)
L_cut = cut_form(L,  jit_options=jit_options)

dofs = locate_dofs("phi>0",level_set,[V])
bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V)

A = assemble_matrix(a_cut, [bc])
A.assemble()

print(A.view(viewer=PETSc.Viewer().createASCII("A.dat")))
b = assemble_vector(L_cut)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Set solver options
opts = PETSc.Options()  # type: ignore
opts["ksp_rtol"] = 1.0e-12
opts["ksp_type"] = "gmres"
opts["pc_type"] = "jacobi"

# Create PETSc Krylov solver and turn convergence monitoring on
solver = PETSc.KSP().create(msh.comm)  # type: ignore
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

uh = fem.Function(V)

# Set a monitor, solve linear system, and display the solver
# configuration
solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b, uh.x.petsc_vec)
solver.view()

# Scatter forward the solution vector to update ghost values
uh.x.scatter_forward()

u_cut = cut_function(uh,cut_mesh)

with XDMFFile(msh.comm, "results/uh.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
#    file.write_function(xi)

with XDMFFile(msh.comm, "results/uh_cut.xdmf", "w") as file:
    file.write_mesh(cut_mesh._mesh)
    file.write_function(u_cut)

with XDMFFile(msh.comm, "results/uh_I.xdmf", "w") as file:
    file.write_mesh(interface_mesh._mesh)

# with XDMFFile(msh.comm, "results/xi.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_function(xi)

deactivated_dofs = np.zeros(shape=(len(dofs),3))

for i in range(len(dofs)):
    for j in range(3):
        dof = dofs[i]
        deactivated_dofs[i][j] = dof_coordinates[dof][j]

points_phys = physical_points(interface_quadrature,msh)

tdim = msh.topology.dim
cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
cell_entities = np.arange(num_cells, dtype=np.int32)

plotter = pyvista.Plotter()
cells, types, x = plot.vtk_mesh(msh,tdim,cell_entities)#(cut_mesh._mesh)
facets, ftypes, fx = plot.vtk_mesh(msh, tdim-1, gp_ids)
ccells, ctypes, cx = plot.vtk_mesh(cut_mesh._mesh)

pset = pyvista.PointSet(points_phys)
pset2 = pyvista.PointSet(deactivated_dofs)

fname = "points_p" + str(MPI.COMM_WORLD.rank) + ".vtk"

pset.cast_to_polydata().save(fname)

fname = "dofs_p" + str(MPI.COMM_WORLD.rank) + ".vtk"
pset2.cast_to_polydata().save(fname)

grid = pyvista.UnstructuredGrid(cells, types, x)
fname = "mesh_p" + str(MPI.COMM_WORLD.rank) + ".vtk"
grid.save(fname)
# fgrid = pyvista.UnstructuredGrid(facets, ftypes, fx)
# cgrid = pyvista.UnstructuredGrid(ccells, ctypes, cx)

# grid.point_data["u"] = uh.x.array
# grid.plot(scalars='u')

# plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, color='lightgray')
# plotter.add_mesh(cgrid, show_edges=True, show_scalar_bar=True, color='blue')
# plotter.add_mesh(fgrid, show_edges=True, show_scalar_bar=True, line_width=5, color='red')
# plotter.add_points(points_phys, render_points_as_spheres=True, color='black')
# plotter.add_points(deactivated_dofs, render_points_as_spheres=True, color='red', point_size=20)

# plotter.view_xy()
# plotter.show()


