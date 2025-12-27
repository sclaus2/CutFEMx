
import pytest
import numpy as np
from mpi4py import MPI
import dolfinx.mesh as mesh
import dolfinx.fem as fem
import ufl
import cutfemx.fem as cf_fem
from cutfemx import cutfemx_cpp as _cpp
import basix
from dolfinx.fem import CoordinateElement
from dolfinx.mesh import Mesh

from cutfemx.quadrature import runtime_quadrature_mesh

def test_elasticity_assembly():
    """
    Compares Standard DOLFINx Assembly vs CutFEMx Runtime Assembly for Linear Elasticity.
    Tests VECTOR-VALUED element support.
    """
    comm = MPI.COMM_WORLD
    N = 4
    msh = mesh.create_unit_square(comm, N, N)
    
    # Vector Function Space
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    
    # Elasticity Parameters
    E, nu = 1.0e5, 0.3
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return 2*mu*epsilon(u) + lmbda*ufl.tr(epsilon(u))*ufl.Identity(len(u))
    
    # --- 1. Baseline (DOLFINx) ---
    a_ref = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    A_ref = fem.assemble_matrix(fem.form(a_ref))
    
    # --- 2. CutFEMx (Runtime Quad) ---
    # Order 2 is sufficient for P1 elasticity (gradients are const)
    qr = runtime_quadrature_mesh(msh, order=2)
    
    # Use dC measure with runtime quadrature
    dx_cut = ufl.Measure("dC", subdomain_data=[(0, qr)], domain=msh)
    
    a_cut = ufl.inner(sigma(u), epsilon(v)) * dx_cut(0)
    
    print("DEBUG: Assembling CutFEMx Elasticity matrix...")
    # CutFEMx assembly
    A_cut = cf_fem.assemble_matrix(cf_fem.cut_form(a_cut))
    print("DEBUG: Assembly complete.")
    
    # --- 3. Compare ---
    A_ref_dense = A_ref.to_dense()
    A_cut_dense = A_cut.to_dense()
    
    diff = np.linalg.norm(A_ref_dense - A_cut_dense)
    print(f"Matrix Difference Norm (Elasticity): {diff}")
    
    assert diff < 1e-9, f"Elasticity assembly mismatch! Norm: {diff}"

if __name__ == "__main__":
    test_elasticity_assembly()
