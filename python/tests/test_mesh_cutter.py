# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

import numpy as np
import pytest
from mpi4py import MPI

import cutfemx
from dolfinx import fem, mesh
import ufl


def _line_level_set():
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(3, 3),
        cell_type=mesh.CellType.triangle,
    )

    def line(x):
        return x[0] - 0.51

    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(line)
    return msh, level_set


def test_mesh_cutter_locate_entities_default_cells():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    intersected_entities = cutter.locate_entities("phi=0")

    entities_exact = np.array([2, 4, 7, 10, 13, 15], dtype=np.int32)
    assert np.array_equal(intersected_entities, entities_exact)


def test_mesh_cutter_locate_entities_subset():
    msh, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    entities_part = np.arange(11, dtype=np.int32)
    intersected_entities = cutter.locate_entities(
        "phi=0", entities=entities_part, entity_dim=msh.topology.dim
    )

    entities_exact = np.array([2, 4, 7, 10], dtype=np.int32)
    assert np.array_equal(intersected_entities, entities_exact)


def test_mesh_cutter_locate_entities_requires_entity_dim_with_subset():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    with pytest.raises(ValueError, match="entity_dim must be supplied"):
        cutter.locate_entities("phi=0", entities=np.arange(11, dtype=np.int32))


def test_mesh_cutter_create_cut_mesh_full():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    cut_mesh = cutter.create_cut_mesh("phi<0", mode="full")

    assert cut_mesh.mesh is not None
    assert cut_mesh.parent_index.size == cut_mesh.is_cut_cell.size
    assert np.count_nonzero(cut_mesh.is_cut_cell) > 0
    assert np.count_nonzero(cut_mesh.is_cut_cell == 0) > 0


def test_mesh_cutter_create_cut_mesh_rejects_interface_full_mode():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    with pytest.raises(ValueError, match="mode='full'"):
        cutter.create_cut_mesh("phi=0", mode="full")


def test_cut_function_uses_cut_mesh_parent_map():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    cut_mesh = cutter.create_cut_mesh("phi<0", mode="full")
    cut_level_set = cutfemx.fem.cut_function(level_set, cut_mesh)

    dof_coordinates = cut_level_set.function_space.tabulate_dof_coordinates()
    expected = dof_coordinates[:, 0] - 0.51
    np.testing.assert_allclose(
        cut_level_set.x.array[: expected.size], expected, atol=1.0e-13
    )


def test_mesh_cutter_runtime_quadrature_returns_runintgen_rules():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    rules = cutter.runtime_quadrature("phi<0", order=2)
    intersected_entities = cutter.locate_entities("phi=0")

    assert rules.kind == "per_entity"
    assert rules.tdim == 2
    assert rules.points.shape[0] == rules.weights.size
    assert rules.offsets.dtype == np.int32
    assert rules.parent_map.dtype == np.int32
    assert rules.offsets[0] == 0
    assert rules.offsets[-1] == rules.weights.size
    assert rules.parent_map.size == rules.offsets.size - 1
    assert set(rules.parent_map.tolist()).issubset(set(intersected_entities.tolist()))
    assert hasattr(rules, "_cutfemx_owner")


def test_mesh_cutter_runtime_quadrature_physical_points_are_lazy_cpp_cache():
    msh, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    rules = cutter.runtime_quadrature("phi<0", order=2)

    points = rules.points
    weights = rules.weights
    offsets = rules.offsets
    parent_map = rules.parent_map

    assert rules.__dict__["physical_points"] is None

    mapped = rules.with_physical_points()

    assert mapped is rules
    assert mapped.physical_points.shape == (msh.geometry.dim, rules.weights.size)
    assert np.all(np.isfinite(mapped.physical_points))
    assert np.shares_memory(mapped.physical_points, rules._cutfemx_owner.physical_points)
    assert np.shares_memory(mapped.points, points)
    assert np.shares_memory(mapped.weights, weights)
    assert np.shares_memory(mapped.offsets, offsets)
    assert np.shares_memory(mapped.parent_map, parent_map)


def test_mesh_cutter_runtime_quadrature_rejects_inclusive_selector():
    _, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    with pytest.raises(ValueError, match="inclusive selectors"):
        cutter.runtime_quadrature("phi<=0", order=2)


def test_cutfemx_form_compiles_runtime_quadrature_measure():
    msh, level_set = _line_level_set()

    cutter = cutfemx.mesh_cutter(level_set)
    rules = cutter.runtime_quadrature("phi<0", order=2)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dxq = ufl.Measure("dx", domain=msh, subdomain_data=rules)

    cut_form = cutfemx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dxq)

    assert cut_form.needs_runtime_data
    assert cut_form.function_spaces == [V, V]
    assert ("cell", -1) in cut_form.runtime_providers
    assert cut_form.runtime_providers[("cell", -1)] is rules
    assert cut_form.integral_summary == [("cell", -1, "runtime", True)]
