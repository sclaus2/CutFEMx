# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI

import cutfemx
import numpy as np
import pytest
from runintgen import dSq, dsq

import ufl
from dolfinx import fem, la, mesh


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


def _interior_facet_indices(msh):
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_to_cell = msh.topology.connectivity(fdim, tdim)
    num_facets = msh.topology.index_map(fdim).size_local
    return np.asarray(
        [facet for facet in range(num_facets) if len(facet_to_cell.links(facet)) == 2],
        dtype=np.int32,
    )


def test_cut_api_locate_entities_default_cells():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    assert cutter.level_set_names == ("phi",)
    intersected_entities = cutfemx.locate_entities(cutter, "phi=0")

    entities_exact = np.array([2, 4, 7, 10, 13, 15], dtype=np.int32)
    assert np.array_equal(intersected_entities, entities_exact)


def test_cut_api_locate_entities_accepts_inclusive_selectors():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    negative = cutfemx.locate_entities(cutter, "phi<0")
    positive = cutfemx.locate_entities(cutter, "phi>0")
    interface = cutfemx.locate_entities(cutter, "phi=0")

    assert np.array_equal(
        cutfemx.locate_entities(cutter, "phi<=0"),
        np.union1d(negative, interface),
    )
    assert np.array_equal(
        cutfemx.locate_entities(cutter, "phi>=0"),
        np.union1d(positive, interface),
    )


def test_cut_api_cut_accepts_cell_subset_as_host():
    msh, level_set = _line_level_set()

    entities_part = np.arange(11, dtype=np.int32)
    cutter = cutfemx.cut(level_set, entities_part, msh.topology.dim)
    intersected_entities = cutfemx.locate_entities(cutter, "phi=0")

    entities_exact = np.array([2, 4, 7, 10], dtype=np.int32)
    assert np.array_equal(intersected_entities, entities_exact)


def test_cut_api_cut_accepts_facet_subset_as_host_with_entity_dofs():
    msh, level_set = _line_level_set()

    msh.topology.create_entities(1)
    facets = np.arange(msh.topology.index_map(1).size_local, dtype=np.int32)
    cutter = cutfemx.cut(level_set, facets, 1)

    negative = cutfemx.locate_entities(cutter, "phi<0")
    intersected = cutfemx.locate_entities(cutter, "phi=0")
    positive = cutfemx.locate_entities(cutter, "phi>0")

    assert intersected.size > 0
    assert np.intersect1d(negative, intersected).size == 0
    assert np.intersect1d(negative, positive).size == 0
    assert np.intersect1d(intersected, positive).size == 0
    assert np.array_equal(
        np.sort(np.concatenate([negative, intersected, positive])), facets
    )


def test_cut_api_locate_entities_zero_dofs_are_interface():
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(2, 1),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(lambda x: x[0] - 0.5)

    cutter = cutfemx.cut(level_set)
    interface_cells = cutfemx.locate_entities(cutter, "phi=0")

    expected = np.arange(
        msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32
    )
    assert np.array_equal(interface_cells, expected)


def test_cut_api_cut_requires_entity_dim_with_subset():
    _, level_set = _line_level_set()

    with pytest.raises(ValueError, match="entity_dim must be supplied"):
        cutfemx.cut(level_set, entities=np.arange(11, dtype=np.int32))


def test_cut_api_create_cut_mesh_full():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    cut_mesh = cutfemx.create_cut_mesh(cutter, "phi<0", mode="full")

    assert cut_mesh.mesh is not None
    assert cut_mesh.parent_index.size == cut_mesh.is_cut_cell.size
    assert np.count_nonzero(cut_mesh.is_cut_cell) > 0
    assert np.count_nonzero(cut_mesh.is_cut_cell == 0) > 0


def test_cut_api_create_cut_mesh_rejects_interface_full_mode():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    with pytest.raises(ValueError, match="mode='full'"):
        cutfemx.create_cut_mesh(cutter, "phi=0", mode="full")


def test_cut_api_create_cut_mesh_accepts_facet_entities():
    msh, level_set = _line_level_set()

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    facet_cutter = cutfemx.cut(level_set, facets, 1)
    cut_facets = cutfemx.locate_entities(facet_cutter, "phi=0")

    cut_facet_cutter = cutfemx.cut(level_set, cut_facets, 1)
    cut_mesh = cutfemx.create_cut_mesh(cut_facet_cutter, "phi<0", mode="cut_only")

    assert cut_mesh.mesh is not None
    assert cut_mesh.mesh.topology.dim == 1
    assert cut_mesh.parent_index.size == cut_mesh.is_cut_cell.size
    assert set(cut_mesh.parent_index.tolist()).issubset(set(cut_facets.tolist()))
    assert np.all(cut_mesh.is_cut_cell == 1)


def test_cut_api_cut_accepts_facet_entities_as_host():
    msh, level_set = _line_level_set()

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    cutter = cutfemx.cut(level_set, facets, 1)

    cut_facets = cutfemx.locate_entities(cutter, "phi=0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    cut_mesh = cutfemx.create_cut_mesh(cutter, "phi<0", mode="cut_only")

    assert cutter.tdim == 1
    assert cut_facets.size > 0
    assert set(cut_facets.tolist()).issubset(set(facets.tolist()))
    assert set(rules.parent_map.tolist()).issubset(set(cut_facets.tolist()))
    assert cut_mesh.mesh is not None
    assert cut_mesh.mesh.topology.dim == 1
    assert set(cut_mesh.parent_index.tolist()).issubset(set(cut_facets.tolist()))


def test_cut_api_cut_accepts_cell_entities_as_host():
    msh, level_set = _line_level_set()

    cells = np.arange(11, dtype=np.int32)
    full_cutter = cutfemx.cut(level_set)
    expected_cut_cells = np.intersect1d(
        cutfemx.locate_entities(full_cutter, "phi=0"), cells
    )

    cutter = cutfemx.cut(level_set, cells, msh.topology.dim)
    cut_cells = cutfemx.locate_entities(cutter, "phi=0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    cut_mesh = cutfemx.create_cut_mesh(cutter, "phi<0", mode="full")

    assert cutter.tdim == msh.topology.dim
    assert np.array_equal(cut_cells, expected_cut_cells)
    assert set(rules.parent_map.tolist()).issubset(set(cut_cells.tolist()))
    assert cut_mesh.mesh is not None
    assert set(cut_mesh.parent_index.tolist()).issubset(set(cells.tolist()))


def test_cut_function_uses_cut_mesh_parent_map():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    cut_mesh = cutfemx.create_cut_mesh(cutter, "phi<0", mode="full")
    cut_level_set = cutfemx.fem.cut_function(level_set, cut_mesh)

    dof_coordinates = cut_level_set.function_space.tabulate_dof_coordinates()
    expected = dof_coordinates[:, 0] - 0.51
    np.testing.assert_allclose(
        cut_level_set.x.array[: expected.size], expected, atol=1.0e-13
    )


def test_cut_api_runtime_quadrature_returns_runintgen_rules():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    intersected_entities = cutfemx.locate_entities(cutter, "phi=0")

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


def test_cut_api_runtime_quadrature_accepts_exterior_facets():
    msh, level_set = _line_level_set()

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    cutter = cutfemx.cut(level_set, facets, 1)
    cut_facets = cutfemx.locate_entities(cutter, "phi=0")
    cut_facet_cutter = cutfemx.cut(level_set, cut_facets, 1)
    rules_from_all_facets = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    rules_from_cut_facets = cutfemx.runtime_quadrature(
        cut_facet_cutter, "phi<0", order=2
    )
    physical_points = rules_from_all_facets.with_physical_points().physical_points

    assert rules_from_all_facets.kind == "per_entity"
    assert rules_from_all_facets.tdim == 1
    assert rules_from_all_facets.points.shape[0] == rules_from_all_facets.weights.size
    assert physical_points.shape == (
        msh.geometry.dim,
        rules_from_all_facets.weights.size,
    )
    assert np.all(np.isfinite(physical_points))
    assert rules_from_all_facets.offsets[0] == 0
    assert rules_from_all_facets.offsets[-1] == rules_from_all_facets.weights.size
    assert (
        rules_from_all_facets.parent_map.size
        == rules_from_all_facets.offsets.size - 1
    )
    assert set(rules_from_all_facets.parent_map.tolist()).issubset(
        set(cut_facets.tolist())
    )
    assert set(rules_from_cut_facets.parent_map.tolist()).issubset(
        set(cut_facets.tolist())
    )
    np.testing.assert_allclose(
        rules_from_all_facets.weights.sum(), rules_from_cut_facets.weights.sum()
    )


def test_cut_api_runtime_quadrature_accepts_interior_facets():
    msh, level_set = _line_level_set()

    facets = _interior_facet_indices(msh)
    cutter = cutfemx.cut(level_set, facets, 1)
    cut_facets = cutfemx.locate_entities(cutter, "phi=0")
    cut_facet_cutter = cutfemx.cut(level_set, cut_facets, 1)
    rules_from_all_facets = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    rules_from_cut_facets = cutfemx.runtime_quadrature(
        cut_facet_cutter, "phi<0", order=2
    )
    physical_points = rules_from_all_facets.with_physical_points().physical_points

    assert cut_facets.size > 0
    assert set(cut_facets.tolist()).issubset(set(facets.tolist()))
    assert rules_from_all_facets.kind == "per_entity"
    assert rules_from_all_facets.tdim == 1
    assert rules_from_all_facets.points.shape[0] == rules_from_all_facets.weights.size
    assert physical_points.shape == (
        msh.geometry.dim,
        rules_from_all_facets.weights.size,
    )
    assert np.all(np.isfinite(physical_points))
    assert rules_from_all_facets.offsets[0] == 0
    assert rules_from_all_facets.offsets[-1] == rules_from_all_facets.weights.size
    assert (
        rules_from_all_facets.parent_map.size
        == rules_from_all_facets.offsets.size - 1
    )
    assert set(rules_from_all_facets.parent_map.tolist()).issubset(
        set(cut_facets.tolist())
    )
    assert set(rules_from_cut_facets.parent_map.tolist()).issubset(
        set(cut_facets.tolist())
    )
    np.testing.assert_allclose(
        rules_from_all_facets.weights.sum(), rules_from_cut_facets.weights.sum()
    )


def test_cutfemx_form_assembles_runtime_exterior_facet_scalar():
    msh, level_set = _line_level_set()

    msh.topology.create_entities(1)
    msh.topology.create_connectivity(1, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    cutter = cutfemx.cut(level_set, facets, 1)
    cut_facets = cutfemx.locate_entities(cutter, "phi=0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    one = fem.Constant(msh, np.float64(1.0))

    cut_form = cutfemx.fem.form(one * dsq(domain=msh, quadrature_provider=rules))
    value = cutfemx.fem.assemble_scalar(cut_form)
    value = msh.comm.allreduce(value, op=MPI.SUM)

    assert cut_form.needs_runtime_data
    assert ("exterior_facet", -1) in cut_form.runtime_providers
    assert set(cut_facets.tolist()).issubset(set(rules.parent_map.tolist()))
    assert np.isfinite(value)
    assert value > 0.0


def test_cutfemx_form_assembles_mixed_exterior_facet_scalar():
    msh, level_set = _line_level_set()

    fdim = msh.topology.dim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = mesh.exterior_facet_indices(msh.topology)
    cutter = cutfemx.cut(level_set, facets, fdim)
    standard_facets = cutfemx.locate_entities(cutter, "phi<0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    one = fem.Constant(msh, np.float64(1.0))

    tags = mesh.meshtags(
        msh,
        fdim,
        np.ascontiguousarray(standard_facets, dtype=np.int32),
        np.full(standard_facets.size, 1, dtype=np.int32),
    )
    ds_standard = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    standard_value = fem.assemble_scalar(fem.form(one * ds_standard(1)))

    runtime_form = cutfemx.fem.form(one * dsq(domain=msh, quadrature_provider=rules))
    runtime_value = cutfemx.fem.assemble_scalar(runtime_form)

    mixed_form = cutfemx.fem.form(
        one * dsq(0, domain=msh, subdomain_data=[standard_facets, rules])
    )
    mixed_value = cutfemx.fem.assemble_scalar(mixed_form)

    assert mixed_form.needs_runtime_data
    assert ("exterior_facet", 0) in mixed_form.runtime_providers
    np.testing.assert_allclose(mixed_value, standard_value + runtime_value)


def test_cutfemx_form_assembles_runtime_interior_facet_scalar():
    msh, level_set = _line_level_set()

    facets = _interior_facet_indices(msh)
    cutter = cutfemx.cut(level_set, facets, 1)
    cut_facets = cutfemx.locate_entities(cutter, "phi=0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    one = fem.Constant(msh, np.float64(1.0))

    cut_form = cutfemx.fem.form(one * dSq(domain=msh, quadrature_provider=rules))
    value = cutfemx.fem.assemble_scalar(cut_form)
    value = msh.comm.allreduce(value, op=MPI.SUM)

    assert cut_form.needs_runtime_data
    assert ("interior_facet", -1) in cut_form.runtime_providers
    assert set(cut_facets.tolist()).issubset(set(rules.parent_map.tolist()))
    assert np.isfinite(value)
    assert value > 0.0


def test_cutfemx_form_assembles_mixed_interior_facet_scalar():
    msh, level_set = _line_level_set()

    fdim = msh.topology.dim - 1
    facets = _interior_facet_indices(msh)
    cutter = cutfemx.cut(level_set, facets, fdim)
    standard_facets = cutfemx.locate_entities(cutter, "phi<0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    one = fem.Constant(msh, np.float64(1.0))

    tags = mesh.meshtags(
        msh,
        fdim,
        np.ascontiguousarray(standard_facets, dtype=np.int32),
        np.full(standard_facets.size, 1, dtype=np.int32),
    )
    dS_standard = ufl.Measure("dS", domain=msh, subdomain_data=tags)
    standard_value = fem.assemble_scalar(fem.form(one * dS_standard(1)))

    runtime_form = cutfemx.fem.form(one * dSq(domain=msh, quadrature_provider=rules))
    runtime_value = cutfemx.fem.assemble_scalar(runtime_form)

    mixed_form = cutfemx.fem.form(
        one * dSq(0, domain=msh, subdomain_data=[standard_facets, rules])
    )
    mixed_value = cutfemx.fem.assemble_scalar(mixed_form)

    assert mixed_form.needs_runtime_data
    assert ("interior_facet", 0) in mixed_form.runtime_providers
    np.testing.assert_allclose(mixed_value, standard_value + runtime_value)


def test_cutfemx_form_assembles_standard_raw_interior_facet_ids():
    msh, level_set = _line_level_set()

    facets = _interior_facet_indices(msh)
    cutter = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    standard_facets = cutfemx.locate_entities(cutter, "phi<0")
    one = fem.Constant(msh, np.float64(1.0))

    tags = mesh.meshtags(
        msh,
        msh.topology.dim - 1,
        np.ascontiguousarray(standard_facets, dtype=np.int32),
        np.full(standard_facets.size, 2, dtype=np.int32),
    )
    dS_tags = ufl.Measure("dS", domain=msh, subdomain_data=tags)
    dS_raw = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=2,
        subdomain_data=standard_facets,
    )

    expected = fem.assemble_scalar(fem.form(one * dS_tags(2)))
    form = cutfemx.fem.form(one * dS_raw)
    value = cutfemx.fem.assemble_scalar(form)

    assert not form.needs_runtime_data
    np.testing.assert_allclose(value, expected)


def test_cutfemx_form_assembles_runtime_interior_facet_jump_matrix():
    msh, level_set = _line_level_set()

    facets = _interior_facet_indices(msh)
    cutter = cutfemx.cut(level_set, facets, 1)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    V = fem.functionspace(msh, ("DG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    jump_u = u("+") - u("-")
    jump_v = v("+") - v("-")
    cut_form = cutfemx.fem.form(
        jump_u * jump_v * dSq(domain=msh, quadrature_provider=rules)
    )
    A = cutfemx.fem.assemble_matrix(cut_form)
    A.scatter_reverse()
    matrix_norm_sq = A.squared_norm()

    assert cut_form.needs_runtime_data
    assert ("interior_facet", -1) in cut_form.runtime_providers
    assert np.isfinite(matrix_norm_sq)
    assert matrix_norm_sq > 0.0


def test_cut_api_runtime_quadrature_physical_points_are_lazy_cpp_cache():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)

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


def test_cut_api_runtime_quadrature_accepts_inclusive_selector():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    inclusive_rules = cutfemx.runtime_quadrature(cutter, "phi<=0", order=2)
    negative_rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)

    assert np.array_equal(inclusive_rules.parent_map, negative_rules.parent_map)
    assert np.allclose(inclusive_rules.weights, negative_rules.weights)


def test_cut_api_multiple_level_sets_names_and_or_selector():
    _, level_set = _line_level_set()
    V = level_set.function_space
    second = fem.Function(V, name="cap")
    second.interpolate(lambda x: x[1] - 0.51)

    cutter = cutfemx.cut([level_set, second])

    assert cutter.level_set_names == ("phi", "cap")
    selected = cutfemx.locate_entities(cutter, "phi=0 or cap=0")
    first = cutfemx.locate_entities(cutfemx.cut(level_set), "phi=0")
    second_only = cutfemx.locate_entities(cutfemx.cut(second), "cap=0")
    assert set(selected.tolist()) == set(first.tolist()) | set(second_only.tolist())


def test_cut_api_multiple_level_sets_default_names_are_frozen():
    _, level_set = _line_level_set()
    V = level_set.function_space
    second = fem.Function(V)
    second.interpolate(lambda x: x[1] - 0.51)

    cutter = cutfemx.cut([level_set, second])
    assert cutter.level_set_names == ("phi", "phi1")

    second.name = "renamed_after_cut"
    cutter.update()
    assert cutter.level_set_names == ("phi", "phi1")
    selected = cutfemx.locate_entities(cutter, "phi=0 or phi1=0")
    assert selected.size > 0


def test_cut_api_rejects_duplicate_real_level_set_names():
    _, level_set = _line_level_set()
    V = level_set.function_space
    level_set.name = "fluid"
    second = fem.Function(V, name="fluid")

    with pytest.raises(ValueError, match="Duplicate level-set function name"):
        cutfemx.cut([level_set, second])


def test_cutfemx_form_compiles_runtime_quadrature_measure():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dxq = ufl.Measure("dx", domain=msh, subdomain_data=rules)

    cut_form = cutfemx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dxq)

    assert cut_form.needs_runtime_data
    assert cut_form._cpp_object.rank == 2
    assert cut_form.function_spaces == [V, V]
    assert ("cell", -1) in cut_form.runtime_providers
    assert cut_form.runtime_providers[("cell", -1)] is rules
    assert cut_form.integral_summary == [("cell", -1, "runtime", True)]


def test_cutfemx_form_assembles_runtime_scalar():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    dxq = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    one = fem.Constant(msh, np.float64(1.0))

    cut_form = cutfemx.fem.form(one * dxq)
    value = cutfemx.fem.assemble_scalar(cut_form)

    assert cut_form.needs_runtime_data
    assert cut_form._cpp_object.rank == 0
    assert np.isfinite(value)
    assert value > 0.0


def test_cutfemx_form_assembles_runtime_vector_and_matrix():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    inside_cells = cutfemx.locate_entities(cutter, "phi<0")
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    dx_inside = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[inside_cells, rules],
    )
    one = fem.Constant(msh, np.float64(1.0))
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    area_form = cutfemx.fem.form(one * dx_inside)
    L = cutfemx.fem.form(one * v * dx_inside)
    a = cutfemx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_inside)

    area = cutfemx.fem.assemble_scalar(area_form)
    b = cutfemx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    A = cutfemx.fem.assemble_matrix(a)
    A.scatter_reverse()
    matrix_norm_before_deactivation = A.squared_norm()
    domain = cutfemx.fem.active_domain(a)
    expected_active_cells = np.unique(
        np.concatenate([inside_cells, rules.parent_map])
    )
    assert np.array_equal(domain.active_cells, expected_active_cells)
    assert np.array_equal(domain.active_cells, np.unique(domain.active_cells))
    assert np.array_equal(domain.inactive_dofs, np.unique(domain.inactive_dofs))

    returned = cutfemx.fem.deactivate_outside(A, domain)
    xi = domain.indicator
    assert xi.name == "active_domain"

    if domain.inactive_dofs.size > 0:
        b.array[domain.inactive_dofs] = 3.0
    returned_with_rhs = cutfemx.fem.deactivate_outside(A, b, domain)

    owned_size = b.index_map.size_local * b.block_size
    owned_indicator_size = xi.x.index_map.size_local * xi.x.block_size
    load_sum = msh.comm.allreduce(np.sum(b.array[:owned_size]), op=MPI.SUM)
    area = msh.comm.allreduce(area, op=MPI.SUM)
    matrix_norm_sq = A.squared_norm()
    deactivated_dofs = msh.comm.allreduce(
        np.count_nonzero(xi.x.array[:owned_indicator_size] < 1.0e-8), op=MPI.SUM
    )
    matrix_nnz = msh.comm.allreduce(
        int(A.indptr[A.index_map(0).size_local]), op=MPI.SUM
    )

    assert np.isfinite(load_sum)
    assert np.isclose(load_sum, area, rtol=1.0e-12, atol=1.0e-12)
    assert np.isfinite(matrix_norm_sq)
    assert matrix_norm_sq > matrix_norm_before_deactivation
    assert returned is domain
    assert returned_with_rhs is domain
    assert domain.active_cells.size > 0
    assert domain.inactive_dofs.size > 0
    np.testing.assert_allclose(b.array[domain.inactive_dofs], 0.0)
    assert deactivated_dofs > 0
    assert matrix_nnz > 0


def test_cutfemx_quadrature_function_normal_assembles_runtime_cell_scalar():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    assert cutfemx.normal is cutfemx.level_set.normal
    n_gamma = cutfemx.normal(level_set)

    dx_rules = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    form = cutfemx.fem.form(ufl.inner(n_gamma, n_gamma) * dx_rules)
    value = cutfemx.fem.assemble_scalar(form)

    assert form.needs_runtime_data
    assert np.isfinite(value)
    assert value > 0.0


def test_cut_api_ghost_penalty_facets_are_owned_unique_interior_facets():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    facets = cutfemx.ghost_penalty_facets(cutter, "phi<0")

    fdim = msh.topology.dim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facet_to_cell = msh.topology.connectivity(fdim, msh.topology.dim)
    num_owned_facets = msh.topology.index_map(fdim).size_local

    assert facets.size > 0
    assert np.array_equal(facets, np.unique(facets))
    assert np.all(facets < num_owned_facets)
    assert all(len(facet_to_cell.links(int(facet))) == 2 for facet in facets)


def test_cutfemx_standard_only_form_active_domain_allows_no_inactive_dofs():
    msh, _ = _line_level_set()
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = cutfemx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    domain = cutfemx.fem.active_domain(a)
    A = cutfemx.fem.assemble_matrix(a)
    A.scatter_reverse()
    returned = cutfemx.fem.deactivate_outside(A, domain)

    all_cells = np.arange(
        msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32
    )
    assert isinstance(a, cutfemx.fem.CutForm)
    assert not a.needs_runtime_data
    assert np.array_equal(domain.active_cells, all_cells)
    assert domain.inactive_dofs.size == 0
    assert returned is domain


def test_cutfemx_deactivation_removes_old_selector_api():
    assert not hasattr(cutfemx.fem, "deactivate")
    assert not hasattr(cutfemx.fem, "locate_dofs")


def test_cutfemx_runtime_area_and_perimeter_for_circle():
    radius = 0.5
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(21, 21),
        cell_type=mesh.CellType.triangle,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V_phi)
    level_set.interpolate(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) - radius)

    cutter = cutfemx.cut(level_set)
    inside_cells = cutfemx.locate_entities(cutter, "phi<0")
    cut_rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=4)
    interface_rules = cutfemx.runtime_quadrature(cutter, "phi=0", order=4)
    one = fem.Constant(msh, np.float64(1.0))

    dx_inside = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[inside_cells, cut_rules],
    )
    dx_interface = ufl.Measure("dx", domain=msh, subdomain_data=interface_rules)

    area = cutfemx.fem.assemble_scalar(cutfemx.fem.form(one * dx_inside))
    perimeter = cutfemx.fem.assemble_scalar(cutfemx.fem.form(one * dx_interface))
    area = msh.comm.allreduce(area, op=MPI.SUM)
    perimeter = msh.comm.allreduce(perimeter, op=MPI.SUM)

    assert abs(area - np.pi * radius**2) < 1.0e-2
    assert abs(perimeter - 2.0 * np.pi * radius) < 1.0e-2
