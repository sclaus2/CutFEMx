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

from dolfinx import default_scalar_type

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


def norm(phi):
    return ufl.max_value(ufl.sqrt(ufl.dot(phi, phi)), 1e-15)


def d(x):
    outarray = np.zeros(x.shape)
    i = 0
    for val in x:
        outarray[i] = 1.0 / val
        i += 1
    return outarray


def solve_predictor_corrector(N):
    # Create mesh
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0 + 1e-5, -1.0), (1.0 + 1e-5, 1.0)),
        n=(N, N),
        cell_type=mesh.CellType.triangle,
        ghost_mode=GhostMode.shared_facet,
    )

    tdim = msh.topology.dim

    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    g = fem.Constant(msh, PETSc.ScalarType(0.0))
    gamma = 1e4
    gamma_g = 1e-1

    n = FacetNormal(msh)
    h = CellDiameter(msh)

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

    DG0 = fem.functionspace(msh, ("DG", 0))
    norm_grad_phi_expr = fem.Expression(norm(grad(level_set)), DG0.element.interpolation_points())
    norm_grad_phi_func = fem.Function(DG0)
    norm_grad_phi_func.interpolate(norm_grad_phi_expr)

    classic_func = fem.Function(DG0)
    classic_func.x.array[:] = d(norm_grad_phi_func.x.array)

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
        # a += avg(gamma_g) * avg(h) * dot(jump(grad(u), n), jump(grad(v), n)) * dS
        return a

    def LC(func):
        L = inner(func * grad(level_set), grad(v)) * dx
        # L += -dot(func * grad(level_set), n_K) * v * dsq
        # L + -dot(grad(v) / norm(grad(v)), n_K) * level_set * dsq
        L += -dot(grad(v), n_K) * g * dsq
        L += gamma * 1.0 / h * v * g * dsq
        return L

    def b_vector(L_cut):
        b = assemble_vector(L_cut)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return b

    a = aC(u, v)
    L1 = LC(classic_func)

    a_cut = cut_form(a)
    L1_cut = cut_form(L1)

    A = assemble_matrix(a_cut)
    A.assemble()

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
    classic_func.x.array[:] = d(norm_grad_phi_func.x.array)

    b_corrector = assemble_vector(L1_cut)
    b_corrector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_corrector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    solver.setOperators(A)
    solver.solve(b_corrector, uh_corrector.x.petsc_vec)
    uh_corrector.x.scatter_forward()

    # Show behavior over several time steps
    iter = 100

    for i in range(1, iter):
        level_set.x.array[:] = uh_corrector.x.array[:]
        norm_grad_phi_func.interpolate(norm_grad_phi_expr)
        classic_func.x.array[:] = d(norm_grad_phi_func.x.array)

        b_corrector = assemble_vector(L1_cut)
        b_corrector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_corrector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        solver.solve(b_corrector, uh_corrector.x.petsc_vec)
        uh_corrector.x.scatter_forward()

    filename = "./results/u" + str(N) + ".bp"
    with dolfinx.io.VTXWriter(MPI.COMM_WORLD, filename, [uh_corrector], engine="BP4") as vtx:
        vtx.write(0.0)

    # Compute the error
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

    e_eik = fem.Function(DG0)
    e_eik.x.array[:] = norm_grad_phi_func.x.array - 1.0

    # Integrate the error
    error_eik = fem.form(ufl.inner(e_eik, e_eik) * ufl.dx)
    error_eik_local = assemble_scalar(error_eik)
    error_eik_global = msh.comm.allreduce(error_eik_local, op=MPI.SUM)
    l2_norm_eik = np.sqrt(error_eik_global)

    return l2_norm, l2_norm_eik


if __name__ == "__main__":
    # Set the number of elements in the mesh
    N = 10
    steps = 5
    Es_l2 = np.zeros(steps, dtype=default_scalar_type)
    Es_eik = np.zeros(steps, dtype=default_scalar_type)
    hs = np.zeros(steps, dtype=np.float64)

    for i in range(steps):
        # Call the function to solve the problem
        l2_norm, l2_norm_eik = solve_predictor_corrector(N)

        Es_l2[i] = l2_norm
        Es_eik[i] = l2_norm_eik
        hs[i] = 1.0 / N
        print(f"h: {hs[i]:.2e} Error: {Es_l2[i]:.2e}")
        print(f"h: {hs[i]:.2e} Eikonal Error: {Es_eik[i]:.2e}")

        N = int(N * 2)

    # Compute the convergence rates
    rates_l2 = np.log(Es_l2[1:] / Es_l2[:-1]) / np.log(hs[1:] / hs[:-1])
    rates_eik = np.log(Es_eik[1:] / Es_eik[:-1]) / np.log(hs[1:] / hs[:-1])

    # Print the convergence rates
    print("Convergence rates:")
    print(f"Rates {rates_l2}")
    print(f"Rates {rates_eik}")
