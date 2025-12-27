
import pytest
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, default_real_type
from dolfinx.fem import form
from cutfemx.level_set import locate_entities, compute_normal, ghost_penalty_facets, facet_topology
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import assemble_vector, cut_form, assemble_matrix, deactivate, assemble_scalar

def solve_poisson(degree, n_ele=32):
    # 1. Create Mesh
    msh = mesh.create_unit_square(MPI.COMM_WORLD, n_ele, n_ele)
    tdim = msh.topology.dim
    
    # 2. Define Level Set (Circle: r=0.3, center=(0.5, 0.5))
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    
    # phi < 0 inside circle
    phi_expr = ufl.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2) - 0.3
    
    level_set = fem.Function(V)
    expr = fem.Expression(phi_expr, V.element.interpolation_points())
    level_set.interpolate(expr)
    
    # 3. Define Analysis Function Space & Functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # 4. Analytical Solution (MMS)
    u_ex = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    f = 8 * (ufl.pi**2) * u_ex
    
    # 5. CutFEM Setup
    intersected_entities = locate_entities(level_set, tdim, "phi=0")
    inside_entities = locate_entities(level_set, tdim, "phi<0")
    
    # Compute Level Set Normal
    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
    n_K = fem.Function(V_DG)
    compute_normal(n_K, level_set, intersected_entities)
    
    # Quadrature Rules
    # Use degree dependent order? P1 -> 2, P2 -> 4?
    # Demo uses order=1 for P2? That seems low.
    # Typically 2*degree is safe.
    q_order = 2 * degree + 1 # For safety
    inside_quadrature = runtime_quadrature(level_set, "phi<0", q_order)
    interface_quadrature = runtime_quadrature(level_set, "phi=0", q_order)
    
    quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]
    
    # Ghost Penalty
    gp_ids = ghost_penalty_facets(level_set, "phi<0")
    gp_topo = facet_topology(msh, gp_ids)
    
    # Measures
    dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
    dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
    dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)
    
    dxq = dx_rt(0) + dx(0) # Cut + Interior
    dsq = dx_rt(1)       # Interface
    
    # Parameters
    gamma = 100.0
    gamma_g = 0.1
    h = ufl.CellDiameter(msh)
    n = ufl.FacetNormal(msh)
    
    # 6. Variational Form
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dxq
    
    # Nitsche on Interface (dsq)
    # u_ex on boundary (Since u_ex is defined everywhere, we use it as Dirichlet data)
    u_D = u_ex 
    
    # Standard Nitsche symmetric
    a += -ufl.dot(ufl.grad(u), n_K) * v * dsq
    a += -ufl.dot(ufl.grad(v), n_K) * u * dsq
    a += (gamma / h) * u * v * dsq
    
    # Ghost Penalty
    a += ufl.avg(gamma_g) * ufl.avg(h) * ufl.dot(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * dS(0)
    
    L = ufl.inner(f, v) * dxq
    
    # RHS Nitsche terms involving u_D
    L += -ufl.dot(ufl.grad(v), n_K) * u_D * dsq
    L += (gamma / h) * u_D * v * dsq
    
    # Assembly
    a_cut = cut_form(a)
    L_cut = cut_form(L)
    
    A = assemble_matrix(a_cut)
    b = assemble_vector(L_cut)
    
    # Deactivate outside DOFs
    deactivate(A, "phi>0", level_set, [V])
    
    # Solve
    from scipy.sparse.linalg import spsolve
    A_scipy = A.to_scipy()
    u_vec = spsolve(A_scipy, b.array)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_vec
    
    # 7. Compute Error
    # L2 Error
    # Assemble square of error, then take sqrt
    error_sq_form = ufl.inner(u_sol - u_ex, u_sol - u_ex) * dxq
    
    E_compiled = cut_form(error_sq_form)
    local_error_sq = assemble_scalar(E_compiled)
    global_error_sq = MPI.COMM_WORLD.allreduce(local_error_sq, op=MPI.SUM)
    global_error = np.sqrt(global_error_sq)
    
    return global_error

def test_p1():
    err = solve_poisson(degree=1, n_ele=32)
    print(f"P1 Error: {err}")
    assert err < 0.05 # Expect convergence, tune threshold

def test_p2():
    err = solve_poisson(degree=2, n_ele=32)
    print(f"P2 Error: {err}")
    assert err < 0.005 # P2 should be much better
    
if __name__ == "__main__":
    test_p1()
    test_p2()
