# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI

import numpy as np
import pytest

pytest.importorskip("petsc4py")

from petsc4py import PETSc

import basix.ufl
import cutfemx
import cutfemx.petsc as cf_petsc

import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import petsc as dolfinx_petsc

pytestmark = pytest.mark.petsc4py


def _dense_array(A: PETSc.Mat) -> np.ndarray:
    dense = A.convert(PETSc.Mat.Type.DENSE)
    return dense.getDenseArray().copy()


def test_petsc_runtime_vector_and_matrix_match_matrixcsr():
    msh = mesh.create_unit_square(MPI.COMM_SELF, 3, 3)
    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(lambda x: x[0] - 0.51)

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    L = cutfemx.fem.form(v * dx_runtime)
    a = cutfemx.fem.form(u * v * dx_runtime)

    b_ref = cutfemx.fem.assemble_vector(L)
    b_ref.scatter_reverse(la.InsertMode.add)
    A_ref = cutfemx.fem.assemble_matrix(a)
    A_ref.scatter_reverse()

    b = cf_petsc.assemble_vector(L)
    b.assemble()
    A = cf_petsc.assemble_matrix(a)
    A.assemble()

    b_existing = cf_petsc.create_vector(V)
    with b_existing.localForm() as b_local:
        b_local.set(0.0)
    cf_petsc.assemble_vector(b_existing, L)
    b_existing.assemble()

    A_existing = cf_petsc.create_matrix(a)
    cf_petsc.assemble_matrix(A_existing, a)
    A_existing.assemble()

    np.testing.assert_allclose(b.getArray(), b_ref.array[: b.getSize()])
    np.testing.assert_allclose(b_existing.getArray(), b.getArray())
    np.testing.assert_allclose(_dense_array(A), A_ref.to_dense())
    np.testing.assert_allclose(_dense_array(A_existing), A_ref.to_dense())


def test_petsc_runtime_matrix_accepts_extension_terms():
    msh = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(lambda x: x[0] - 0.51)

    cut_data = cutfemx.cut(level_set)
    selector = "phi<0"
    rules = cutfemx.runtime_quadrature(cut_data, selector, order=2)
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = cutfemx.fem.form(u * v * dx_runtime)

    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        selector,
        1.0,
        root_policy="interior_only",
    )
    term = cutfemx.extensions.ExtensionPenaltyTerm(
        V, beta=2.5, quadrature_degree=2, cut_data=cut_data, aggregation=aggregation
    )

    A_ref = cutfemx.fem.assemble_matrix(a)
    A_ext = cutfemx.extensions.extension_penalty_matrix(
        V, cut_data, aggregation, beta=2.5, quadrature_degree=2
    )
    A_ref.scatter_reverse()
    A_ext.scatter_reverse()

    A = cf_petsc.assemble_matrix(a, extension_terms=[term])
    A.assemble()

    np.testing.assert_allclose(_dense_array(A), A_ref.to_dense() + A_ext.to_dense())


def test_petsc_runtime_matrix_accepts_mixed_subspace_extension_terms():
    msh = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    velocity_element = basix.ufl.element(
        "Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)
    )
    pressure_element = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(
        msh, basix.ufl.mixed_element([velocity_element, pressure_element])
    )

    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V_phi)
    level_set.interpolate(lambda x: x[0] - 0.51)
    cut_data = cutfemx.cut(level_set)
    selector = "phi<0"
    rules = cutfemx.runtime_quadrature(cut_data, selector, order=2)
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_data=rules)

    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)
    a = cutfemx.fem.form(
        (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p - ufl.div(u) * q)
        * dx_runtime
    )

    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        selector,
        1.0,
        root_policy="interior_only",
    )
    terms = [
        cutfemx.extensions.ExtensionPenaltyTerm(
            W.sub(0), 1.0, 2, cut_data=cut_data, aggregation=aggregation
        ),
        cutfemx.extensions.ExtensionPenaltyTerm(
            W.sub(1), 0.5, 2, cut_data=cut_data, aggregation=aggregation
        ),
    ]

    A = cf_petsc.assemble_matrix(a, extension_terms=terms)
    A.assemble()
    dense = _dense_array(A)

    assert dense.shape[0] == W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    assert np.linalg.norm(dense) > 0.0


def _interface_band_problem():
    msh = mesh.create_rectangle(
        MPI.COMM_SELF,
        ((-1.0, -1.0), (1.0, 1.0)),
        (8, 8),
        cell_type=mesh.CellType.quadrilateral,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V_phi)
    level_set.interpolate(lambda x: x[0] ** 2 + x[1] ** 2 - 0.6**2)

    cut_data = cutfemx.cut(level_set)
    cut_cells = cutfemx.locate_entities(cut_data, "phi=0")
    rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order=4)
    dGamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=rules)
    dx_band = ufl.Measure("dx", domain=msh, subdomain_id=2, subdomain_data=cut_cells)

    velocity = basix.ufl.element(
        "Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)
    )
    pressure = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([velocity, pressure]))
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    def mass_and_stiffness(u, p, v, q):
        return (
            ufl.inner(u, v)
            + p * q
            + ufl.inner(ufl.grad(u), ufl.grad(v))
            + ufl.inner(ufl.grad(p), ufl.grad(q))
        )

    a = mass_and_stiffness(u, p, v, q) * dGamma
    a += mass_and_stiffness(u, p, v, q) * dx_band
    f = fem.Constant(msh, np.array([1.0, -0.5]))
    L = ufl.inner(f, v) * dGamma
    return W, cutfemx.fem.form(a), cutfemx.fem.form(L)


def test_petsc_deactivate_outside_sets_diagonal_after_final_assembly():
    W, a, L = _interface_band_problem()
    domain = cutfemx.fem.active_domain(a)
    inactive = domain.inactive_dofs
    assert inactive.size > 0

    A = cf_petsc.assemble_matrix(a)
    A.assemble()
    b = cf_petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Rows outside the active cells receive no assembly contribution. The
    # scan must see the finally assembled matrix and leave it assembled.
    np.testing.assert_array_equal(cf_petsc.zero_rows(A), inactive)
    assert A.assembled

    returned = cf_petsc.deactivate_outside(A, b, domain)
    A.assemble()

    assert returned is domain
    assert cf_petsc.zero_rows(A) == []
    np.testing.assert_allclose(A.getDiagonal().getArray()[inactive], 1.0)
    np.testing.assert_allclose(b.getArray()[inactive], 0.0)

    A_ref = cutfemx.fem.assemble_matrix(a)
    A_ref.scatter_reverse()
    b_ref = cutfemx.fem.assemble_vector(L)
    b_ref.scatter_reverse(la.InsertMode.add)
    cutfemx.fem.deactivate_outside(A_ref, b_ref, domain)
    np.testing.assert_allclose(_dense_array(A), A_ref.to_dense())

    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    x = A.createVecRight()
    ksp.solve(b, x)

    assert ksp.getConvergedReason() > 0
    r = A.createVecLeft()
    A.mult(x, r)
    r.axpy(-1.0, b)
    assert r.norm() < 1.0e-10 * b.norm()
    np.testing.assert_allclose(x.getArray()[inactive], 0.0)
    ksp.destroy()


def test_petsc_deactivate_outside_rejects_matrix_without_diagonal():
    W, a, _ = _interface_band_problem()
    domain = cutfemx.fem.active_domain(a)

    # A plain DOLFINx matrix only preallocates entries of the integration
    # cells, so inactive rows have no diagonal to set.
    msh = W.mesh
    u = ufl.TrialFunction(W)
    v = ufl.TestFunction(W)
    cut_cells = domain.active_cells
    dx_cells = ufl.Measure("dx", domain=msh, subdomain_data=[(1, cut_cells)])
    A = dolfinx_petsc.assemble_matrix(fem.form(ufl.inner(u, v) * dx_cells(1)))
    A.assemble()

    with pytest.raises(RuntimeError, match="nonzero structure"):
        cf_petsc.deactivate_outside(A, domain)
    A.destroy()


def _ghost_penalty_elasticity(cells_only=False):
    """Vector Q2 on a cut square with a gradient-jump ghost penalty."""
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        ((-1.0, -1.0), (1.0, 1.0)),
        (6, 6),
        cell_type=mesh.CellType.quadrilateral,
    )
    level_set = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
    level_set.interpolate(lambda x: x[0] ** 2 + x[1] ** 2 - 0.63**2)
    cut_data = cutfemx.cut(level_set)
    inside = cutfemx.locate_entities(cut_data, "phi<0")
    rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order=3)
    ghost = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
    dx = ufl.Measure("dx", domain=msh, subdomain_id=0, subdomain_data=[inside, rules])
    dS = ufl.Measure("dS", domain=msh, subdomain_id=1, subdomain_data=ghost)

    V = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(msh)
    a = ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
    if not cells_only:
        a += ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * dS
    dofs = fem.locate_dofs_geometrical(V, lambda x: x[0] < -0.3)
    bc = fem.dirichletbc(np.zeros(msh.geometry.dim), dofs, V)
    return cutfemx.fem.form(a), [bc]


def _owned_rows(A: PETSc.Mat) -> np.ndarray:
    """Owned rows of A as a dense array with global columns."""
    indptr, indices, values = A.getValuesCSR()
    dense = np.zeros((indptr.size - 1, A.getSize()[1]), dtype=values.dtype)
    rows = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    dense[rows, indices] = values
    return dense


def _assert_same_matrix(A: PETSc.Mat, B: PETSc.Mat):
    """Compare owned rows. Off-process contributions arrive at final
    assembly in no fixed order, so sums may differ by round-off."""
    a, b = _owned_rows(A), _owned_rows(B)
    scale = MPI.COMM_WORLD.allreduce(np.abs(b).max(initial=0.0), op=MPI.MAX)
    assert scale > 0.0
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-13 * scale)


def test_petsc_create_matrix_holds_full_sparsity_pattern():
    """AIJ matrices start assembled, with explicit zeros on the whole pattern."""
    a, _ = _ghost_penalty_elasticity()
    A = cf_petsc.create_matrix(a)
    pattern = cutfemx.fem.create_matrix(a)

    bs0, bs1 = pattern.block_size
    num_rows = pattern.index_map(0).size_local
    assert A.assembled
    assert A.getInfo(PETSc.Mat.InfoType.LOCAL)["nz_used"] == (
        bs0 * bs1 * pattern.indptr[num_rows]
    )
    assert A.norm(PETSc.NormType.FROBENIUS) == 0.0


def test_petsc_assembly_into_full_pattern_matches_set_values():
    """Adding into the CSR arrays equals MatSetValues, also when re-assembling."""
    a, bcs = _ghost_penalty_elasticity()
    A = cf_petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    # One MatSetValues call leaves B unassembled, so assembling B goes
    # through MatSetValues for every element.
    B = cf_petsc.create_matrix(a)
    B.setValueLocal(0, 0, 0.0, addv=PETSc.InsertMode.ADD_VALUES)
    assert not B.assembled
    cf_petsc.assemble_matrix(B, a, bcs=bcs)
    B.assemble()

    _assert_same_matrix(A, B)

    A.zeroEntries()
    cf_petsc.assemble_matrix(A, a, bcs=bcs)
    A.assemble()
    _assert_same_matrix(A, B)


def test_petsc_assembly_outside_pattern_falls_back_to_set_values():
    """Blocks missing from an assembled matrix are inserted with MatSetValues."""
    a_cells, _ = _ghost_penalty_elasticity(cells_only=True)
    a, bcs = _ghost_penalty_elasticity()

    A = cf_petsc.create_matrix(a_cells)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    cf_petsc.assemble_matrix(A, a, bcs=bcs)
    A.assemble()
    B = cf_petsc.assemble_matrix(a, bcs=bcs)
    B.assemble()

    _assert_same_matrix(A, B)


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="checks a process-local flag")
def test_petsc_assembly_into_full_pattern_skips_set_values():
    """Assembling into the CSR arrays leaves the matrix assembled.

    Any MatSetValues call, e.g. after an element was wrongly found to be
    outside the nonzero structure, marks the matrix as unassembled. The
    Python wrapper's diagonal insertion does too, so call the binding.
    """
    a, _ = _ghost_penalty_elasticity()
    A = cf_petsc.create_matrix(a)
    assemble = getattr(cutfemx.cutfemx_cpp.fem.petsc, f"assemble_matrix_{a.type_name}")
    assemble(A, a._cpp_object, [])
    assert A.assembled
