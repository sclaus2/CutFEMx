# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI

import basix.ufl
import cutfemx
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh


def _line_cut(n=(3, 3), offset=0.51):
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=n,
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.interpolate(lambda x: x[0] - offset)
    return msh, phi


def test_cell_aggregation_classifies_active_and_cut_cells():
    _, phi = _line_cut()
    cut_data = cutfemx.cut(phi)

    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        0.2,
        root_policy="interior_or_well_cut",
    )

    cut_cells = cutfemx.locate_entities(cut_data, "phi=0")
    interior_cells = cutfemx.locate_entities(cut_data, "phi<0")
    active_cells = np.union1d(cut_cells, interior_cells)

    assert np.array_equal(aggregation.cut_cells, cut_cells)
    assert np.array_equal(aggregation.interior_cells, interior_cells)
    assert np.array_equal(aggregation.active_cells, active_cells)
    assert set(aggregation.well_posed_cells.tolist()).issubset(
        set(active_cells.tolist())
    )
    assert set(aggregation.ill_posed_cells.tolist()).issubset(set(cut_cells.tolist()))
    assert aggregation.rootless_cells.size == 0


def test_cell_aggregation_propagates_ill_posed_cells_to_roots():
    _, phi = _line_cut()
    cut_data = cutfemx.cut(phi)

    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
    )

    assert aggregation.ill_posed_cells.size == aggregation.cut_cells.size
    assert aggregation.rootless_cells.size == 0
    for cell in aggregation.ill_posed_cells:
        root = aggregation.root_cell[cell]
        assert root in set(aggregation.interior_cells.tolist())
        assert aggregation.propagation_depth[cell] > 0


def test_cell_aggregation_opposite_volume_fractions_sum_to_one_on_cut_cells():
    _, phi = _line_cut()
    cut_data = cutfemx.cut(phi)

    negative = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        0.0,
        root_policy="interior_or_well_cut",
    )
    positive = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi > 0",
        0.0,
        root_policy="interior_or_well_cut",
    )

    assert np.array_equal(negative.cut_cells, positive.cut_cells)
    cut_cells = negative.cut_cells
    np.testing.assert_allclose(
        negative.cut_volume_fraction[cut_cells]
        + positive.cut_volume_fraction[cut_cells],
        np.ones(cut_cells.size),
        atol=1.0e-12,
    )


def test_cell_aggregation_rejects_rootless_active_component():
    _, phi = _line_cut(n=(2, 1), offset=0.5)
    cut_data = cutfemx.cut(phi)

    with pytest.raises(RuntimeError, match="without an admissible root"):
        cutfemx.extensions.create_cell_aggregation(
            cut_data,
            "phi < 0",
            1.0,
            root_policy="interior_only",
        )

    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
        allow_rootless=True,
    )
    assert aggregation.rootless_cells.size == aggregation.ill_posed_cells.size


def test_extension_penalty_matrix_is_symmetric_and_annihilates_constants():
    msh, phi = _line_cut(n=(4, 4), offset=0.51)
    cut_data = cutfemx.cut(phi)
    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    A = cutfemx.extensions.extension_penalty_matrix(
        V,
        cut_data,
        aggregation,
        beta=3.0,
        quadrature_degree=2,
    )
    A.scatter_reverse()
    dense = A.to_dense()

    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    np.testing.assert_allclose(dense @ np.ones(dense.shape[1]), 0.0, atol=1.0e-12)
    assert np.linalg.norm(dense) > 0.0


def test_extension_quadrature_uses_full_bad_cell_measure():
    n = (4, 4)
    msh, phi = _line_cut(n=n, offset=0.51)
    cut_data = cutfemx.cut(phi)
    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    q = cutfemx.extensions.extension_quadrature(
        V,
        cut_data,
        aggregation,
        quadrature_degree=2,
    )

    expected_triangle_area = 1.0 / (2.0 * n[0] * n[1])
    assert len(q.bad_cells) == aggregation.ill_posed_cells.size
    assert len(q.root_cells) == aggregation.ill_posed_cells.size
    for pair in range(len(q.bad_cells)):
        q0, q1 = q.offsets[pair], q.offsets[pair + 1]
        np.testing.assert_allclose(
            np.sum(q.weights[q0:q1]), expected_triangle_area, atol=1.0e-12
        )


def test_extension_penalty_cellwise_dg0_beta_matches_scalar_beta():
    msh, phi = _line_cut(n=(4, 4), offset=0.51)
    cut_data = cutfemx.cut(phi)
    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    Q0 = fem.functionspace(msh, ("DG", 0))
    beta = fem.Function(Q0)
    beta.x.array[:] = 2.5

    A_scalar = cutfemx.extensions.extension_penalty_matrix(
        V,
        cut_data,
        aggregation,
        beta=2.5,
        quadrature_degree=2,
    )
    A_dg0 = cutfemx.extensions.extension_penalty_matrix(
        V,
        cut_data,
        aggregation,
        beta=beta,
        quadrature_degree=2,
    )
    A_scalar.scatter_reverse()
    A_dg0.scatter_reverse()

    np.testing.assert_allclose(A_dg0.to_dense(), A_scalar.to_dense(), atol=1.0e-12)


def test_extension_penalty_term_api_matches_direct_call():
    msh, phi = _line_cut(n=(4, 4), offset=0.51)
    cut_data = cutfemx.cut(phi)
    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    term = cutfemx.extensions.ExtensionPenaltyTerm(V, beta=1.7, quadrature_degree=2)

    A_term = cutfemx.extensions.extension_penalty_matrix(term, cut_data, aggregation)
    A_direct = cutfemx.extensions.extension_penalty_matrix(
        V,
        cut_data,
        aggregation,
        beta=1.7,
        quadrature_degree=2,
    )
    A_term.scatter_reverse()
    A_direct.scatter_reverse()

    np.testing.assert_allclose(A_term.to_dense(), A_direct.to_dense(), atol=1.0e-12)


def test_extension_penalty_assembles_on_mixed_parent_subspaces_without_collapse():
    msh, phi = _line_cut(n=(4, 4), offset=0.51)
    cut_data = cutfemx.cut(phi)
    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        "phi < 0",
        1.0,
        root_policy="interior_only",
    )

    velocity_element = basix.ufl.element(
        "Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)
    )
    pressure_element = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(
        msh, basix.ufl.mixed_element([velocity_element, pressure_element])
    )

    A_u = cutfemx.extensions.extension_penalty_matrix(
        W.sub(0),
        cut_data,
        aggregation,
        beta=2.0,
        quadrature_degree=3,
    )
    A_p = cutfemx.extensions.extension_penalty_matrix(
        W.sub(1),
        cut_data,
        aggregation,
        beta=0.5,
        quadrature_degree=2,
    )
    A_u.scatter_reverse()
    A_p.scatter_reverse()
    dense_u = A_u.to_dense()
    dense_p = A_p.to_dense()

    assert dense_u.shape == dense_p.shape
    assert dense_u.shape[0] == W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    np.testing.assert_allclose(dense_u, dense_u.T, atol=1.0e-12)
    np.testing.assert_allclose(dense_p, dense_p.T, atol=1.0e-12)

    velocity_dofs = set()
    pressure_dofs = set()
    tdim = msh.topology.dim
    for cell in range(msh.topology.index_map(tdim).size_local):
        velocity_dofs.update(W.sub(0).dofmap.cell_dofs(cell).tolist())
        pressure_dofs.update(W.sub(1).dofmap.cell_dofs(cell).tolist())

    u_rows, u_cols = np.nonzero(np.abs(dense_u) > 1.0e-12)
    p_rows, p_cols = np.nonzero(np.abs(dense_p) > 1.0e-12)
    assert set(u_rows.tolist()).issubset(velocity_dofs)
    assert set(u_cols.tolist()).issubset(velocity_dofs)
    assert set(p_rows.tolist()).issubset(pressure_dofs)
    assert set(p_cols.tolist()).issubset(pressure_dofs)
    assert np.linalg.norm(dense_u) > 0.0
    assert np.linalg.norm(dense_p) > 0.0


def test_matrixcsr_runtime_matrix_accepts_extension_terms():
    msh, phi = _line_cut(n=(4, 4), offset=0.51)
    cut_data = cutfemx.cut(phi)
    selector = "phi < 0"
    rules = cutfemx.runtime_quadrature(cut_data, selector, order=2)
    dxq = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = cutfemx.fem.form(u * v * dxq)

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
    A = cutfemx.fem.assemble_matrix(a, extension_terms=[term])
    A_ref.scatter_reverse()
    A_ext.scatter_reverse()
    A.scatter_reverse()

    np.testing.assert_allclose(A.to_dense(), A_ref.to_dense() + A_ext.to_dense())


def test_matrixcsr_runtime_matrix_accepts_mixed_subspace_extension_terms():
    msh, phi = _line_cut(n=(4, 4), offset=0.51)
    cut_data = cutfemx.cut(phi)
    selector = "phi < 0"
    rules = cutfemx.runtime_quadrature(cut_data, selector, order=2)
    dxq = ufl.Measure("dx", domain=msh, subdomain_data=rules)

    velocity_element = basix.ufl.element(
        "Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)
    )
    pressure_element = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(
        msh, basix.ufl.mixed_element([velocity_element, pressure_element])
    )
    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)
    a = cutfemx.fem.form(
        (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p - ufl.div(u) * q)
        * dxq
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

    A = cutfemx.fem.assemble_matrix(a, extension_terms=terms)
    A.scatter_reverse()
    dense = A.to_dense()

    assert dense.shape[0] == W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    np.testing.assert_allclose(dense, dense.T, atol=1.0e-12)
    assert np.linalg.norm(dense) > 0.0
