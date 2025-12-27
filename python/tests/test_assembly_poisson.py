
import pytest
import numpy as np
from mpi4py import MPI
import dolfinx.mesh as mesh
import dolfinx.fem as fem
import ufl
import cutfemx.fem as cf_fem
# import cutfemx.quadrature as cf_quad # Not using installed module for debugging helper
from cutfemx import cutfemx_cpp as _cpp
import basix
from dolfinx.fem import CoordinateElement
from dolfinx.mesh import Mesh


from cutfemx.quadrature import runtime_quadrature_mesh

def test_poisson_assembly():
    """
    Compares Standard DOLFINx Assembly vs CutFEMx Runtime Assembly (on full mesh).
    """
    comm = MPI.COMM_WORLD
    N = 4 # Small mesh for testing
    msh = mesh.create_unit_square(comm, N, N)
    
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    
    # --- 1. Baseline (DOLFINx) ---
    a_ref = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    A_ref = fem.assemble_matrix(fem.form(a_ref))
    # A_ref.assemble() not needed for MatrixCSR in recent dolfinx?
    
    # --- 2. CutFEMx (Runtime Quad) ---
    # Generate full domain quadrature (order >= 1 for P1 gradient inner product which is const)
    # P1*P1 grad dot grad is constant. Order 0 is enough? 
    # Let's use order 2 to be safe.
    qr = runtime_quadrature_mesh(msh, order=2)
    
    # Use dC measure with runtime quadrature
    dx_cut = ufl.Measure("dC", subdomain_data=[(0, qr)], domain=msh)
    
    a_cut = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_cut(0)
    
    print("DEBUG: Assembling CutFEMx matrix...")
    # CutFEMx assembly
    A_cut = cf_fem.assemble_matrix(cf_fem.cut_form(a_cut))
    # A_cut.assemble()
    print("DEBUG: Assembly complete.")
    
    # --- 3. Compare ---
    # Convert to dense for small matrices
    A_ref_dense = A_ref.to_dense()
    A_cut_dense = A_cut.to_dense()
    
    diff = np.linalg.norm(A_ref_dense - A_cut_dense)
    print(f"Matrix Difference Norm: {diff}")
    
    assert diff < 1e-12, f"Assembly mismatch! Norm: {diff}"

if __name__ == "__main__":
    test_poisson_assembly()
