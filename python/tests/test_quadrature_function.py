# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
"""Quadrature functions on runtime cell and interior-facet measures.

Several forms below have equal UFL signatures on purpose: the JIT cache then
reuses one compiled module, and each form must still evaluate its own
quadrature functions with the side requested by its restrictions.
"""

from mpi4py import MPI

import cutfemx
import numpy as np
import pytest

import ufl
from dolfinx import fem, mesh

_GRADIENT = np.array([0.05, 0.0, 1.0])


def _plane(sign=1.0):
    """Hexahedral box cut by a tilted plane; ``sign`` flips the level set."""
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
        (3, 3, 3),
        cell_type=mesh.CellType.hexahedron,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(
        lambda x: sign * (_GRADIENT[0] * x[0] + _GRADIENT[2] * x[2] - 0.1234)
    )
    return msh, level_set


def _measures(msh, level_set, order=2):
    """Return (dx on the surface, dS on surface ∩ facets, facet rules)."""
    cut = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cut, "phi=0", order)
    cells = cutfemx.locate_entities(cut, "phi=0")
    facets = cutfemx.interior_facets_for_cells(msh, cells)
    facet_cut = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    facet_rules = cutfemx.runtime_quadrature(facet_cut, "phi=0", order)
    dG = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=rules)
    dS_G = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=facet_rules)
    return dG, dS_G, facet_rules


def _assemble(expr):
    form = cutfemx.fem.form(expr)
    return form.mesh.comm.allreduce(cutfemx.fem.assemble_scalar(form), op=MPI.SUM)


def test_conormal_restrictions_are_opposite_whatever_was_compiled_before():
    msh, level_set = _plane()
    _, dS_G, _ = _measures(msh, level_set)
    n = cutfemx.normal(level_set)
    mu = cutfemx.conormal(n)
    length = _assemble(fem.Constant(msh, 1.0) * dS_G)

    # Same signature as the conormal product below: compiled first on purpose.
    n_product = _assemble(ufl.dot(n("+"), n("-")) * dS_G)
    mu_product = _assemble(ufl.dot(mu("+"), mu("-")) * dS_G)
    mu_plus = _assemble(ufl.dot(mu("+"), mu("+")) * dS_G)
    mu_sum = _assemble(ufl.dot(mu("+") + mu("-"), mu("+") + mu("-")) * dS_G)

    assert length > 0.0
    np.testing.assert_allclose(n_product, length, rtol=1.0e-12)
    np.testing.assert_allclose(mu_product, -length, rtol=1.0e-12)
    np.testing.assert_allclose(mu_plus, length, rtol=1.0e-12)
    np.testing.assert_allclose(mu_sum, 0.0, atol=1.0e-12)


def test_quadrature_function_and_function_on_one_element_keep_their_values():
    msh, level_set = _plane()
    dG, _, _ = _measures(msh, level_set)
    area = _assemble(fem.Constant(msh, 1.0) * dG)
    f = fem.Function(fem.functionspace(msh, ("DG", 0)))
    f.x.array[:] = 2.0
    q = cutfemx.QuadratureFunction(msh, lambda x: np.full(x.shape[1], 3.0))

    np.testing.assert_allclose(_assemble(f * f * dG), 4.0 * area, rtol=1.0e-12)
    np.testing.assert_allclose(_assemble(q * q * dG), 9.0 * area, rtol=1.0e-12)
    np.testing.assert_allclose(_assemble(f * f * dG), 4.0 * area, rtol=1.0e-12)


class _ParentCell:
    """Side-dependent values under a cache key that ignores the side."""

    def cache_key(self, context):
        return "parent-cell"

    def evaluate(self, context):
        rules = context.rules
        return np.repeat(rules.parent_map, np.diff(rules.offsets)).astype(np.float64)


def test_restricted_values_follow_the_side_across_forms():
    msh, level_set = _plane()
    _, dS_G, _ = _measures(msh, level_set)
    W = fem.functionspace(msh, ("DG", 0))
    cell_index = fem.Function(W)  # local cell index, ghosts included
    cell_index.x.array[W.dofmap.list[:, 0]] = np.arange(W.dofmap.list.shape[0])
    c = cutfemx.QuadratureFunction(msh, name="cell")
    c.set_evaluator(_ParentCell())

    plus = _assemble(c("+") * dS_G)
    minus = _assemble(c("-") * dS_G)

    np.testing.assert_allclose(plus, _assemble(cell_index("+") * dS_G), rtol=1.0e-12)
    np.testing.assert_allclose(minus, _assemble(cell_index("-") * dS_G), rtol=1.0e-12)
    assert abs(plus - minus) > 1.0e-3 * abs(plus)


def test_cached_forms_evaluate_the_current_quadrature_functions():
    unit_normal_z = _GRADIENT[2] / np.linalg.norm(_GRADIENT)
    msh, level_set = _plane()
    _, flipped = _plane(sign=-1.0)
    flipped = fem.Function(level_set.function_space)
    flipped.x.array[:] = -level_set.x.array
    dG, _, _ = _measures(msh, level_set)
    area = _assemble(fem.Constant(msh, 1.0) * dG)

    up = cutfemx.normal(level_set, name="up")
    down = cutfemx.normal(flipped, name="down")

    np.testing.assert_allclose(_assemble(up[2] * dG), unit_normal_z * area, rtol=1e-12)
    np.testing.assert_allclose(
        _assemble(down[2] * dG), -unit_normal_z * area, rtol=1e-12
    )


def test_cached_forms_on_a_new_mesh_evaluate_the_new_level_set():
    unit_normal_z = _GRADIENT[2] / np.linalg.norm(_GRADIENT)
    for sign in (1.0, -1.0):
        msh, level_set = _plane(sign=sign)
        dG, _, _ = _measures(msh, level_set)
        area = _assemble(fem.Constant(msh, 1.0) * dG)
        n = cutfemx.normal(level_set)
        np.testing.assert_allclose(
            _assemble(n[2] * dG), sign * unit_normal_z * area, rtol=1e-12
        )


def test_set_values_supplies_side_specific_interior_facet_values():
    msh, level_set = _plane()
    _, dS_G, facet_rules = _measures(msh, level_set)
    length = _assemble(fem.Constant(msh, 1.0) * dS_G)
    num_points = facet_rules.total_points
    s = cutfemx.QuadratureFunction(msh, name="side_sign")
    s.set_values(facet_rules, np.ones(num_points), restriction="+")
    s.set_values(facet_rules, -np.ones(num_points), restriction="-")

    np.testing.assert_allclose(_assemble(s("+") * s("-") * dS_G), -length, rtol=1e-12)
    np.testing.assert_allclose(_assemble(s("-") * dS_G), -length, rtol=1e-12)
    with pytest.raises(ValueError, match="restriction"):
        s.set_values(facet_rules, np.ones(num_points), restriction="left")


def test_conormal_of_explicit_normal_values_matches_evaluated_normal():
    msh, level_set = _plane()
    _, dS_G, facet_rules = _measures(msh, level_set)
    n_exact = _GRADIENT / np.linalg.norm(_GRADIENT)
    n_explicit = cutfemx.QuadratureFunction(msh, name="normal_values", shape=(3,))
    n_explicit.set_values(facet_rules, np.tile(n_exact, (facet_rules.total_points, 1)))
    mu_explicit = cutfemx.conormal(n_explicit)
    mu = cutfemx.conormal(cutfemx.normal(level_set))

    for side in ("+", "-"):
        difference = mu_explicit(side) - mu(side)
        error = _assemble(ufl.dot(difference, difference) * dS_G)
        np.testing.assert_allclose(error, 0.0, atol=1.0e-24)


def test_quadrature_function_is_a_dolfinx_function_without_derivatives():
    msh, level_set = _plane()
    dG, _, _ = _measures(msh, level_set)
    n = cutfemx.normal(level_set)

    assert isinstance(n, fem.Function)
    assert not n.is_cellwise_constant()
    with pytest.raises(NotImplementedError, match="Derivatives"):
        cutfemx.fem.form(ufl.div(n) * dG)


def test_quadrature_function_in_mixed_integral_is_rejected():
    msh, level_set = _plane()
    cut = cutfemx.cut(level_set)
    inside = cutfemx.locate_entities(cut, "phi<0")
    rules = cutfemx.runtime_quadrature(cut, "phi<0", 2)
    dx_mixed = ufl.Measure(
        "dx", domain=msh, subdomain_id=0, subdomain_data=[inside, rules]
    )
    phi_q = cutfemx.level_set_value(level_set)

    with pytest.raises(NotImplementedError, match="mixed standard/runtime"):
        cutfemx.fem.form(phi_q * dx_mixed)


def test_quadrature_functions_on_empty_runtime_rules_assemble_to_zero():
    """A process without cut entities receives empty rule sets."""
    msh, level_set = _plane()
    level_set.x.array[:] += 10.0  # the plane leaves the mesh
    dG, dS_G, facet_rules = _measures(msh, level_set)
    n = cutfemx.normal(level_set)
    mu = cutfemx.conormal(n)
    q = cutfemx.QuadratureFunction(msh, lambda x: np.ones(x.shape[1]))

    assert facet_rules.total_points == 0
    assert _assemble(q * n[2] * dG) == 0.0
    assert _assemble(ufl.dot(mu("+"), mu("-")) * dS_G) == 0.0
