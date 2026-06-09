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
