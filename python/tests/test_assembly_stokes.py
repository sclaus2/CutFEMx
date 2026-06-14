# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT


import numpy as np
from mpi4py import MPI
import dolfinx.mesh as mesh
import dolfinx.fem as fem
import ufl
import cutfemx.fem as cf_fem

from quadrature_utils import runtime_quadrature_mesh


def _interior_facets(msh):
    fdim = msh.topology.dim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facet_to_cell = msh.topology.connectivity(fdim, msh.topology.dim)
    num_facets = msh.topology.index_map(fdim).size_local
    return np.asarray(
        [
            facet
            for facet in range(num_facets)
            if len(facet_to_cell.links(facet)) == 2
        ],
        dtype=np.int32,
    )


def test_stokes_assembly():
    """
    Compares Standard DOLFINx Assembly vs CutFEMx Runtime Assembly for Stokes (Mixed).
    Tests MIXED ELEMENT support.
    """
    comm = MPI.COMM_WORLD
    N = 4
    msh = mesh.create_unit_square(comm, N, N)
    
    import basix.ufl
    
    # Taylor-Hood P2/P1
    # P2 Vector Element
    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    # P1 Scalar Element
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    
    # Mixed Element
    ME = basix.ufl.mixed_element([P2, P1])
    
    V = fem.functionspace(msh, ME)
    
    # Trial/Test Functions
    # Use split form for definition but assemble on mixed space
    w = ufl.TrialFunction(V)
    q = ufl.TestFunction(V)
    
    (u, p) = ufl.split(w)
    (v, r) = ufl.split(q)
    
    # Stokes Form
    # a(u, p; v, r) = inner(grad(u), grad(v)) - div(v)*p - div(u)*r
    a_ref = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v)*p - ufl.div(u)*r) * ufl.dx
    
    print("DEBUG: Assembling Reference Stokes...")
    A_ref = fem.assemble_matrix(fem.form(a_ref))
    print("DEBUG: Reference Assembly Complete.")
    
    # --- 2. CutFEMx (Runtime Quad) ---
    # Order 3/4 needed for P2?
    # P2*P2 grad -> P1*P1 -> P2. Order 2 enough? P2*P1 -> P3?
    # Uses order 4 to be safe.
    qr = runtime_quadrature_mesh(msh, order=4)
    
    dx_cut = ufl.Measure("dx", subdomain_data=qr, domain=msh)
    
    # Same form but with dx_cut
    a_cut = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v)*p - ufl.div(u)*r) * dx_cut
    
    print("DEBUG: Assembling CutFEMx Stokes matrix...")
    # CutFEMx assembly
    A_cut = cf_fem.assemble_matrix(cf_fem.cut_form(a_cut))
    print("DEBUG: Assembly complete.")
    
    # --- 3. Compare ---
    A_ref_dense = A_ref.to_dense()
    A_cut_dense = A_cut.to_dense()
    
    diff = np.linalg.norm(A_ref_dense - A_cut_dense)
    print(f"Matrix Difference Norm (Stokes): {diff}")
    
    assert diff < 1e-9, f"Stokes assembly mismatch! Norm: {diff}"


def test_stokes_mixed_ghost_penalty_assembly():
    """Compare mixed velocity/pressure interior-facet stabilization assembly."""
    comm = MPI.COMM_WORLD
    N = 4
    msh = mesh.create_unit_square(comm, N, N)

    import basix.ufl

    P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))

    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)

    facets = _interior_facets(msh)
    facet_tags = mesh.meshtags(
        msh,
        msh.topology.dim - 1,
        facets,
        np.ones(facets.size, dtype=np.int32),
    )
    dS_ref = ufl.Measure("dS", domain=msh, subdomain_data=facet_tags)(1)
    dS_cut = ufl.Measure("dS", domain=msh, subdomain_id=1, subdomain_data=facets)

    gp = (
        ufl.avg(h)
        * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n))
    )
    gp += (
        ufl.avg(h) ** 3
        * ufl.inner(ufl.jump(ufl.grad(p), n), ufl.jump(ufl.grad(q), n))
    )

    A_ref = fem.assemble_matrix(fem.form(gp * dS_ref))
    A_cut = cf_fem.assemble_matrix(cf_fem.cut_form(gp * dS_cut))

    diff = np.linalg.norm(A_ref.to_dense() - A_cut.to_dense())
    print(f"Matrix Difference Norm (Stokes ghost penalty): {diff}")

    assert diff < 1e-9, f"Stokes ghost penalty mismatch! Norm: {diff}"

if __name__ == "__main__":
    test_stokes_assembly()
