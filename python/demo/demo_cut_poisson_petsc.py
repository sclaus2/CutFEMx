import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate

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

N = 6

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
    return x[0]+1e-2
  #return np.sqrt((x[0])**2+(x[1])**2)-0.5

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

dof_coordinates = V.tabulate_dof_coordinates()

cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm,cut_cells,msh,inside_entities)
interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
interface_mesh = create_cut_cells_mesh(msh.comm,size_local,interface_cells)

order = 1
inside_quadrature = runtime_quadrature(level_set,"phi<0",order)
interface_quadrature = runtime_quadrature(level_set,"phi=0",order)

print("inside parent map=", inside_quadrature.parent_map)
print("interface parent map=", interface_quadrature.parent_map)

dofs = V.dofmap.cell_dofs(inside_quadrature.parent_map[len(inside_quadrature.parent_map)-1])

print("dofs=", dofs)

quad_domains = [(0,inside_quadrature), (1,interface_quadrature)]

gp_ids =  ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh,gp_ids)

dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dx", metadata={"quadrature_rule":"runtime"} , subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)
dsq = dx_rt(1)

a  = inner(grad(u), grad(v))*dxq

# cut cells on surface
a += - dot(grad(u), n_K)*v*dsq
a += - dot(grad(v), n_K)*u*dsq
a += gamma*1.0/h*u*v*dsq
#a += avg(gamma_g)*avg(h)*dot(jump(grad(u), n), jump(grad(v), n))*dS(0)

L  = f*v*dxq
L += - dot(grad(v), n_K)*g*dsq
L +=  gamma*1.0/h*v*g*dsq

a_cut = cut_form(a)
L_cut = cut_form(L)

A = assemble_matrix(a_cut)
xi = deactivate(A,"phi>0",level_set,[V])
A.assemble()

print(A.view(viewer=PETSc.Viewer().createASCII("A.dat")))
b = assemble_vector(L_cut)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set solver options
opts = PETSc.Options()  # type: ignore
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
#solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
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

with XDMFFile(msh.comm, "results/xi.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(xi)

with XDMFFile(msh.comm, "results/n_K.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(n_K)



