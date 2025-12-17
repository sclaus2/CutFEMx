import os
import numpy as np
from mpi4py import MPI

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, cut_function

from cutfemx.petsc import assemble_vector, assemble_matrix, deactivate, locate_dofs

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
    points=((-1.0 + 1e-14, -1.0 + 1e-14), (1.0, 1.0)),
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
gamma = 10
gamma_g = 1e-1

n = FacetNormal(msh)
h = CellDiameter(msh)


# Slopes
a1 = 2.0  # example value
a2 = 0.4  # example value
a3 = 0.0  # example value

# Starting value
l1 = 0.75
l2 = 0.5
l3 = 0.75

b3 = a3 * l3 + a2 * l2 + a1 * l1
y0 = -(l1 * a1 + 0.25 * a2) + 1e-10


# Define the function
def piecewise_linear(x):
    value_array = np.zeros(x[0].shape)
    i = 0
    for val in x[0]:
        if -1.2 <= val < -0.25:
            value_array[i] = y0 + a1 * (val + 1)
        elif -0.25 <= val < 0.25:
            f_m025 = y0 + a1 * (0.75)
            value_array[i] = f_m025 + a2 * (val + 0.25)
        elif 0.25 <= val <= 1.2:
            f_m025 = y0 + a1 * (0.75)
            f_025 = f_m025 + a2 * (0.5)
            value_array[i] = f_025 + a3 * (val - 0.25)
        i += 1

    return value_array


x = np.linspace(-1.0, 1.0, 100)
y = piecewise_linear((x,))
plt.plot(x, y, label="piecewise linear")
plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Piecewise Linear Function")
plt.legend()
plt.grid()
plt.savefig("piecewise_linear_function.svg", format="svg")
plt.close()
# plt.show() removed


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
level_set.interpolate(piecewise_linear)

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
    return ufl.max_value(ufl.sqrt(ufl.dot(phi, phi)), 1e-10)


def d_1(x):
    return 1.0 / x


def d_2(x):
    return 3 * x - 2 * x * x


def d_3(x):
    return 2 - x


def classic(x):
    outarray = np.zeros(x.shape)
    i = 0
    for val in x:
        outarray[i] = d_1(val)
        i += 1
    return outarray


def basting(x):
    outarray = np.zeros(x.shape)
    i = 0
    for val in x:
        if val > 1.0:
            outarray[i] = d_1(val)
        elif val <= 1.0:
            outarray[i] = d_2(val)
        i += 1
    return outarray


def adams(x):
    outarray = np.zeros(x.shape)
    i = 0
    for val in x:
        if val > 1.0:
            outarray[i] = d_1(val)
        elif val <= 1.0:
            outarray[i] = d_3(val)
        i += 1
    return outarray


x = np.linspace(0, 6.0, 100)
y1 = 1 - classic(x)
y2 = 1 - basting(x)
y3 = 1 - adams(x)
plt.plot(x, y1, label="classic")
plt.plot(x, y2, label="basting")
plt.plot(x, y3, label="adams")
plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("x")
plt.ylabel("d")
plt.title("Diffusion Function")
plt.legend()
plt.grid()
plt.ylim(-1.2, 1.2)
plt.savefig("diffusion_function.svg", format="svg")
plt.close()
# plt.show() removed

DG0 = fem.functionspace(msh, ("DG", 0))
norm_grad_phi_expr = fem.Expression(norm(grad(level_set)), DG0.element.interpolation_points())
norm_grad_phi_func = fem.Function(DG0)
norm_grad_phi_func.interpolate(norm_grad_phi_expr)

classic_func = fem.Function(DG0)
classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

basting_func = fem.Function(DG0)
basting_func.x.array[:] = basting(norm_grad_phi_func.x.array)
adams_func = fem.Function(DG0)
adams_func.x.array[:] = adams(norm_grad_phi_func.x.array)

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

a = inner(grad(u), grad(v)) * dx

# cut cells on surface
a += -dot(grad(u), n_K) * v * dsq
a += -dot(grad(v), n_K) * u * dsq
a += gamma * 1.0 / h * u * v * dsq


def RHS(func):
    L = inner(func * grad(level_set), grad(v)) * dx
    L += -dot(grad(v), n_K) * g * dsq
    L += gamma * 1.0 / h * v * g * dsq
    return L


L1 = RHS(classic_func)
L2 = RHS(basting_func)
L3 = RHS(adams_func)

a_cut = cut_form(a)

L1_cut = cut_form(L1)
L2_cut = cut_form(L2)
L3_cut = cut_form(L3)

A = assemble_matrix(a_cut)
A.assemble()


def b_vector(L_cut):
    b = assemble_vector(L_cut)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return b


b1 = b_vector(L1_cut)
b2 = b_vector(L2_cut)
b3 = b_vector(L3_cut)

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
uh2 = fem.Function(V)
uh2.name = "uh2"
uh3 = fem.Function(V)
uh3.name = "uh3"

# Set a monitor, solve linear system, and display the solver
# configuration
# solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
solver.solve(b1, uh1.x.petsc_vec)

solver.solve(b2, uh2.x.petsc_vec)

solver.solve(b3, uh3.x.petsc_vec)


# Scatter forward the solution vector to update ghost values
uh1.x.scatter_forward()

midline_dofsV = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0.0))

dofs_coords = V.tabulate_dof_coordinates()
x_dofs_coords = dofs_coords[:, 0]
midline_x = x_dofs_coords[midline_dofsV]


plt.plot(midline_x, midline_x, label="exact")
plt.plot(midline_x, level_set.x.array[midline_dofsV], label="$phi^{0}$")
plt.plot(midline_x, uh1.x.array[midline_dofsV], label="$1/|phi|$")
plt.plot(midline_x, uh2.x.array[midline_dofsV], label="Basting")
plt.plot(midline_x, uh3.x.array[midline_dofsV], label="Adams")

plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Piecewise Linear Function Test")
plt.legend()
plt.grid()
plt.savefig("piecewise_linear_function_test.svg", format="svg")
plt.close()
# plt.show() removed

# Predictor
uh_predictor = fem.Function(V)
uh_predictor.name = "uh_predictor"

a_predictor = inner(grad(u), grad(v)) * dx
a_predictor += -dot(grad(u), n_K) * v * dsq
a_predictor += -dot(grad(v), n_K) * u * dsq
a_predictor += gamma * 1.0 / h * u * v * dsq
a_predictor_cut = cut_form(a_predictor)

L_predictor = inner(sign, v) * dx
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

plt.plot(midline_x, midline_x, label="exact")
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
plt.savefig("predictor_corrector.svg", format="svg")
plt.close()
# plt.show() removed

# Show behavior over several time steps
N = 5
u_pc = np.zeros((N, len(midline_dofsV)))
u_pc[0, :] = uh_corrector.x.array[midline_dofsV]


for i in range(1, N):
    level_set.x.array[:] = uh_corrector.x.array[:]
    norm_grad_phi_func.interpolate(norm_grad_phi_expr)
    classic_func.x.array[:] = classic(norm_grad_phi_func.x.array)

    b_corrector = assemble_vector(L1_cut)
    b_corrector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_corrector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    solver.solve(b_corrector, uh_corrector.x.petsc_vec)
    uh_corrector.x.scatter_forward()

    u_pc[i, :] = uh_corrector.x.array[midline_dofsV]

plt.plot(midline_x, midline_x, label="exact")

for i in range(0, N):
    N_str = str(i)
    plt.plot(midline_x, u_pc[i, :], label="u_pc(" + N_str + ")")

plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Corector Steps")
plt.legend()
plt.grid()
plt.savefig("corrector_steps.svg", format="svg")
plt.close()
# plt.show() removed


# Test different Picard Iteration
def non_linear_d(u):
    return 1 - 1 / u


u_picard2 = fem.Function(V)
u_picard2.x.array[:] = uh_predictor.x.array[:]
level_set.x.array[:] = uh_predictor.x.array[:]
norm_grad_phi_func.interpolate(norm_grad_phi_expr)

non_lineral_d_func = fem.Function(DG0)
non_lineral_d_func.x.array[:] = non_linear_d(norm_grad_phi_func.x.array)

print("u_picard2: ", u_picard2.x.array[midline_dofsV])
print("norm_grad_phi_func: ", norm_grad_phi_func.x.array[midline_dofsV])
print("non_lineral_d_func: ", non_lineral_d_func.x.array[midline_dofsV])

plt.plot(midline_x, midline_x, label="exact")
plt.plot(midline_x, non_lineral_d_func.x.array[midline_dofsV], label=r"$\phi^{0}$")
plt.show()
plt.close()

a_picard2 = inner(non_lineral_d_func * grad(u), grad(v)) * dx
a_picard2 += -dot(non_lineral_d_func * grad(u), n_K) * v * dsq
a_picard2 += -dot(non_lineral_d_func * grad(v), n_K) * u * dsq
a_picard2 += gamma * non_lineral_d_func * 1.0 / h * u * v * dsq
L_picard2 = inner(g, v) * dx


# Solve corrector
a_picard2_cut = cut_form(a_picard2)
L_picard2_cut = cut_form(L_picard2)

u_pc = np.zeros((N, len(midline_dofsV)))
N = 1
for i in range(0, N):
    A_picard2 = assemble_matrix(a_picard2_cut)
    A_picard2.assemble()
    # print("A_picard2: ", A_picard2.view())
    b_picard2 = assemble_vector(L_picard2_cut)
    b_picard2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_picard2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    solver.setOperators(A_picard2)
    solver.solve(b_picard2, u_picard2.x.petsc_vec)
    u_picard2.x.scatter_forward()

    print("Picard Iteration: ", i)
    print("u_picard2: ", u_picard2.x.array[midline_dofsV])

    # Update norm_grad_phi_func
    level_set.x.array[:] = u_picard2.x.array[:]
    norm_grad_phi_func.interpolate(norm_grad_phi_expr)
    non_lineral_d_func.x.array[:] = non_linear_d(norm_grad_phi_func.x.array)

    u_pc[i, :] = u_picard2.x.array[midline_dofsV]

    # A_picard2.zeroEntries()
    # with b_picard2.localForm() as loc_L:
    #     loc_L.set(0)

plt.plot(midline_x, midline_x, label="exact")

for i in range(0, N):
    N_str = str(i)
    plt.plot(midline_x, u_pc[i, :], label="u_pc(" + N_str + ")")

plt.axhline(y=0, color="k", linestyle="--")
plt.axvline(x=0, color="k", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Corector Steps")
plt.legend()
plt.grid()
plt.savefig("corrector_steps_implicit.svg", format="svg")
plt.close()
