import os
import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs

from dolfinx.fem.assemble import assemble_scalar

from dolfinx import fem, mesh, plot, la

# from dolfinx.fem.assemble import assemble_vector, assemble_matrix
from dolfinx.fem import form
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode

import ufl
from ufl import dS, dx, grad, inner, dc, FacetNormal, CellDiameter, dot, avg, jump, ds
import matplotlib.pyplot as plt

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

N = 20

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0 + 1e-5, -1.0), (1.0 + 1e-5, 1.0)),
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
gamma = 50
gamma_g = 1e-1

n = FacetNormal(msh)
h = CellDiameter(msh)


# Define the function
def circle(x):
    return (x[0] - 0.0) * (x[0] - 0.0) + (x[1] - 0.0) * (x[1] - 0.0) - 0.5 * 0.5


def exact_solution(x):
    return np.sqrt(x[0] * x[0] + x[1] * x[1]) - 0.5


# Define the function to compute the sign of the level set function
# This function returns -1 for negative values, 1 for positive values, and 0 for zero
def sgn(phi_array):
    sign_array = np.zeros(phi_array.shape)
    for i in range(len(phi_array)):
        if phi_array[i] < 0:
            sign_array[i] = -1
        elif phi_array[i] > 0:
            sign_array[i] = 1
        else:
            sign_array[i] = 0
    return sign_array


level_set = fem.Function(V)
level_set.interpolate(circle)

ls_ex = fem.Function(V)
ls_ex.interpolate(exact_solution)

sign = fem.Function(V)
sign.x.array[:] = sgn(level_set.x.array)

dim = msh.topology.dim

intersected_entities = locate_entities(level_set, dim, "phi=0")
inside_entities = locate_entities(level_set, dim, "phi<0")
outside_entities = locate_entities(level_set, dim, "phi>0")

V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K, level_set, intersected_entities)
# n_K = fem.Constant(msh, (PETSc.ScalarType(1.0),PETSc.ScalarType(0.0)))


def norm(phi):
    return ufl.max_value(ufl.sqrt(ufl.dot(phi, phi)), 1e-15)


def d_1(x):
    return 1.0 / x


def classic(x):
    outarray = np.zeros(x.shape)
    i = 0
    for val in x:
        outarray[i] = d_1(val)
        i += 1
    return outarray


DG0 = fem.functionspace(msh, ("DG", 0))
norm_grad_phi_expr = fem.Expression(norm(grad(level_set)), DG0.element.interpolation_points())
norm_grad_phi_func = fem.Function(DG0)
norm_grad_phi_func.interpolate(norm_grad_phi_expr)

classic_func = fem.Function(DG0)
classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

dof_coordinates = V.tabulate_dof_coordinates()

interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
# interface_mesh = create_cut_cells_mesh(msh.comm, interface_cells)
if len(interface_cells.cut_cells) == 0:
    print("No interface cells found")
    exit(0)

order = 2
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)
outside_quadrature = runtime_quadrature(level_set, "phi>0", order)

quad_domains = [(0, inside_quadrature), (1, interface_quadrature), (2, outside_quadrature)]

gp_ids = ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh, gp_ids)

dx = ufl.Measure(
    "dx",
    subdomain_data=[(0, inside_entities), (1, intersected_entities), (2, outside_entities)],
    domain=msh,
)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)
dsq = dx_rt(1)


def aC(u, v):
    a = inner(grad(u), grad(v)) * dx

    # cut cells on surface
    a += -dot(grad(u), n_K) * v * dsq
    a += -dot(grad(v), n_K) * u * dsq
    a += gamma * 1.0 / h * u * v * dsq
    return a


def LC(func):
    L = inner(func * grad(level_set), grad(v)) * dx
    # L += -dot(func * grad(level_set), n_K) * v * dsq
    # L + -dot(grad(v) / norm(grad(v)), n_K) * level_set * dsq
    L += -dot(grad(v), n_K) * g * dsq
    L += gamma * 1.0 / h * v * g * dsq
    return L


a = aC(u, v)
L1 = LC(classic_func)


a_cut = cut_form(a)
L1_cut = cut_form(L1)

A = assemble_matrix(a_cut)
A.assemble()


def b_vector(L_cut):
    b = assemble_vector(L_cut)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return b


b1 = b_vector(L1_cut)


# Set solver options
opts = PETSc.Options()  # type: ignore
opts["ksp_rtol"] = 1.0e-12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"

# Create PETSc Krylov solver andv    turn convergence monitoring on
solver = PETSc.KSP().create(msh.comm)  # type: ignore
solver.setFromOptions()

# Set matrix operator
solver.setOperators(A)

uh1 = fem.Function(V)
uh1.name = "uh1"


# Set a monitor, solve linear system, and display the solver
# configuration
# solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b1, uh1.x.petsc_vec)
uh1.x.scatter_forward()

midline_dofsV = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))

dofs_coords = V.tabulate_dof_coordinates()
x_dofs_coords = dofs_coords[:, 0]
midline_x = x_dofs_coords[midline_dofsV]

# Predictor
uh_predictor = fem.Function(V)
uh_predictor.name = "uh_predictor"

a_predictor = inner(grad(u), grad(v)) * dx
a_predictor += -dot(grad(u), n_K) * v * dsq
a_predictor += -dot(grad(v), n_K) * u * dsq
a_predictor += gamma * 1.0 / h * u * v * dsq
a_predictor_cut = cut_form(a_predictor)

L_predictor = inner(sign, v) * dx
# L_predictor = inner(-1.0, v) * (dx_rt(0) + dx(0))
# L_predictor += inner(1.0, v) * (dx_rt(2) + dx(2))
L_predictor += -dot(grad(v), n_K) * g * dsq
L_predictor += gamma * 1.0 / h * v * g * dsq
L_predictor += inner(sign, v) * ds
L_predictor_cut = cut_form(L_predictor)

A_predictor = assemble_matrix(a_predictor_cut)
A_predictor.assemble()

b_predictor = assemble_vector(L_predictor_cut)
b_predictor.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
b_predictor.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

solver.setOperators(A_predictor)
solver.solve(b_predictor, uh_predictor.x.petsc_vec)
uh_predictor.x.scatter_forward()

# Corrector
uh_corrector = fem.Function(V)
uh_corrector.name = "uh_corrector"

level_set.x.array[:] = uh_predictor.x.array[:]
norm_grad_phi_func.interpolate(norm_grad_phi_expr)
classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

b_corrector = assemble_vector(L1_cut)
b_corrector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
b_corrector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

solver.setOperators(A)
solver.solve(b_corrector, uh_corrector.x.petsc_vec)
uh_corrector.x.scatter_forward()

plt.plot(midline_x, ls_ex.x.array[midline_dofsV], label="exact")
plt.plot(midline_x, level_set.x.array[midline_dofsV], label=r"$\phi^{0}$")
plt.plot(midline_x, uh_predictor.x.array[midline_dofsV], label="Predictor")
plt.plot(midline_x, uh_corrector.x.array[midline_dofsV], label="Corrector")

plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predictor-Corrector Method")
plt.legend()
plt.grid()
# plt.savefig("predictor_corrector_circle.svg", format="svg")
# plt.close()
plt.show()


# plotter = pyvista.Plotter(shape=(1, 1))

# cells, types, x = plot.vtk_mesh(msh)
# grid = pyvista.UnstructuredGrid(cells, types, x)

# plotter.subplot(0, 0)
# grid.point_data["u"] = uh_predictor.x.array
# grid.set_active_scalars("u")
# warped = grid.warp_by_scalar(scale_factor=1.0)
# plotter.add_mesh(warped)


# plotter.show()


# Show behavior over several time steps
# N = 5
# u_pc = np.zeros((N, len(midline_dofsV)))
# u_pc[0, :] = uh_corrector.x.array[midline_dofsV]


# for i in range(1, N):
#     level_set.x.array[:] = uh_corrector.x.array[:]
#     norm_grad_phi_func.interpolate(norm_grad_phi_expr)
#     classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

#     b_corrector = assemble_vector(L1_cut)
#     b_corrector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#     b_corrector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

#     solver.solve(b_corrector, uh_corrector.x.petsc_vec)
#     uh_corrector.x.scatter_forward()

#     u_pc[i, :] = uh_corrector.x.array[midline_dofsV]

# print("|grad(phi)|=", norm_grad_phi_func.x.array)

# plt.plot(midline_x, ls_ex.x.array[midline_dofsV], label="exact")

# for i in range(0, N):
#     N_str = str(i)
#     plt.plot(midline_x, u_pc[i, :], label="u_pc(" + N_str + ")")

# plt.axhline(y=0, color="k", linestyle="--")
# plt.axvline(x=0, color="k", linestyle="--")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Corector Steps")
# plt.legend()
# plt.grid()
# # plt.savefig("circle_corrector_steps.svg", format="svg")
# # plt.close()
# plt.show()  # removed

V3 = fem.functionspace(msh, ("Lagrange", 3))
ls_ex_3 = fem.Function(V3)
ls_ex_3.interpolate(exact_solution)

ls_3 = fem.Function(V3)
ls_3.interpolate(uh_corrector)


e_W = fem.Function(V3)
e_W.x.array[:] = ls_3.x.array - ls_ex_3.x.array

# Integrate the error
error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
error_local = assemble_scalar(error)
error_global = msh.comm.allreduce(error_local, op=MPI.SUM)
l2_norm = np.sqrt(error_global)

print("L2 norm of the difference between the exact solution and the computed solution:", l2_norm)

e_eik = fem.Function(DG0)
e_eik.x.array[:] = norm_grad_phi_func.x.array - 1.0

# Integrate the error
error_eik = fem.form(ufl.inner(e_eik, e_eik) * ufl.dx)
error_eik_local = assemble_scalar(error_eik)
error_eik_global = msh.comm.allreduce(error_eik_local, op=MPI.SUM)
l2_norm_eik = np.sqrt(error_eik_global)
print(
    "L2 norm of the difference between the exact solution and the computed solution eikonal:",
    l2_norm_eik,
)
