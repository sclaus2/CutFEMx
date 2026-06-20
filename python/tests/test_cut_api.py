# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from collections import Counter

from mpi4py import MPI

import cutfemx
import numpy as np
import pytest

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


def _quadratic_circle_level_set():
    center = (0.47, 0.43)
    radius = 0.31
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = fem.functionspace(msh, ("Lagrange", 2))
    level_set = fem.Function(V)
    level_set.interpolate(
        lambda x: (x[0] - center[0]) ** 2
        + (x[1] - center[1]) ** 2
        - radius**2
    )
    return msh, level_set, center


def _quadratic_sphere_level_set():
    center = (0.47, 0.43, 0.41)
    radius = 0.31
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (4, 4, 4),
        cell_type=mesh.CellType.hexahedron,
    )
    V = fem.functionspace(msh, ("Lagrange", 2))
    level_set = fem.Function(V)
    level_set.interpolate(
        lambda x: (x[0] - center[0]) ** 2
        + (x[1] - center[1]) ** 2
        + (x[2] - center[2]) ** 2
        - radius**2
    )
    return msh, level_set, center


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


def _radial_normal(x, center):
    radius = ufl.sqrt(sum((x[i] - center[i]) ** 2 for i in range(len(center))))
    return ufl.as_vector(
        tuple((x[i] - center[i]) / radius for i in range(len(center)))
    )


def _projected_conormal(background_normal, surface_normal):
    value = background_normal - ufl.dot(background_normal, surface_normal) * surface_normal
    return value / ufl.sqrt(ufl.inner(value, value))


def test_cut_api_locate_entities_default_cells():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    assert cutter.level_set_names == ("phi",)
    intersected_entities = cutfemx.locate_entities(cutter, "phi=0")

    entities_exact = np.array([2, 4, 7, 10, 13, 15], dtype=np.int32)
    assert np.array_equal(intersected_entities, entities_exact)


def test_cut_api_accepts_linear_no_refinement_options():
    _, level_set = _line_level_set()

    options = {
        "cut_approximation": "linear",
        "cut_approximation_order": 1,
        "max_refinement_iterations": 0,
    }
    cutter = cutfemx.cut(level_set, **options)
    intersected_entities = cutfemx.locate_entities(cutter, "phi=0")

    fresh = cutfemx.locate_entities(cutfemx.cut(level_set, **options), "phi=0")
    assert np.array_equal(intersected_entities, fresh)

    level_set.interpolate(lambda x: x[1] - 0.51)
    cutfemx.update(cutter)
    updated = cutfemx.locate_entities(cutter, "phi=0")
    fresh_updated = cutfemx.locate_entities(cutfemx.cut(level_set, **options), "phi=0")
    assert np.array_equal(updated, fresh_updated)


def test_cut_api_update_reclassifies_when_level_set_values_change():
    _, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    initial = cutfemx.locate_entities(cutter, "phi=0")

    level_set.interpolate(lambda x: x[1] - 0.51)
    cutfemx.update(cutter)
    updated = cutfemx.locate_entities(cutter, "phi=0")
    fresh = cutfemx.locate_entities(cutfemx.cut(level_set), "phi=0")

    assert not np.array_equal(updated, initial)
    assert np.array_equal(updated, fresh)


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


def test_cut_api_cut_rejects_entity_dim_without_subset():
    _, level_set = _line_level_set()

    with pytest.raises(ValueError, match="entity_dim is only valid"):
        cutfemx.cut(level_set, entity_dim=0)


def test_cut_api_cut_rejects_invalid_level_set_inputs():
    _, level_set = _line_level_set()

    with pytest.raises(TypeError, match="expects a Function"):
        cutfemx.cut("phi")
    with pytest.raises(ValueError, match="requires at least one"):
        cutfemx.cut([])
    with pytest.raises(TypeError, match="sequence entries"):
        cutfemx.cut([level_set, object()])


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


def _convex_hull_edges_3d(points, vertices):
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    order = np.argsort(angles)
    return {
        tuple(sorted((int(vertices[order[i]]), int(vertices[order[(i + 1) % 4]]))))
        for i in range(4)
    }


def test_cut_api_mixed_hex_surface_mesh_splits_basix_quads_on_diagonal():
    if MPI.COMM_WORLD.size != 1:
        pytest.skip("cut visualization meshes duplicate partition-boundary vertices")

    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
        (4, 4, 4),
        cell_type=mesh.CellType.hexahedron,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    level_set = fem.Function(V)
    level_set.interpolate(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 0.65**2)

    cutter = cutfemx.cut(level_set)
    cut_mesh = cutfemx.create_cut_mesh(cutter, "phi=0", mode="cut_only")
    assert cut_mesh.mesh is not None
    assert cut_mesh.mesh.topology.cell_type == mesh.CellType.triangle

    surface_mesh = cut_mesh.mesh
    tdim = surface_mesh.topology.dim
    surface_mesh.topology.create_connectivity(tdim, 0)
    cell_to_vertex = surface_mesh.topology.connectivity(tdim, 0)
    assert cell_to_vertex is not None

    cell_vertices = []
    parent_indices = []
    num_cells = surface_mesh.topology.index_map(tdim).size_local
    for cell in range(num_cells):
        vertices = [int(v) for v in cell_to_vertex.links(cell)]
        assert len(vertices) == 3
        cell_vertices.append(vertices)
        parent_indices.append(int(cut_mesh.parent_index[cell]))

    checked_split_quads = 0
    for cell in range(num_cells - 1):
        if parent_indices[cell] != parent_indices[cell + 1]:
            continue

        parent_cells = [cell_vertices[cell], cell_vertices[cell + 1]]
        unique_vertices = sorted(
            {v for tri_vertices in parent_cells for v in tri_vertices}
        )
        if len(unique_vertices) != 4:
            continue

        edge_counts = Counter()
        for vertices in parent_cells:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge_counts[tuple(sorted((vertices[i], vertices[j])))] += 1

        boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
        hull_edges = _convex_hull_edges_3d(
            surface_mesh.geometry.x[unique_vertices], np.asarray(unique_vertices)
        )
        assert boundary_edges == hull_edges
        checked_split_quads += 1

    assert checked_split_quads > 0


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

    ds_runtime = ufl.Measure("ds", domain=msh, subdomain_data=rules)
    cut_form = cutfemx.fem.form(one * ds_runtime)
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

    ds_runtime = ufl.Measure("ds", domain=msh, subdomain_data=rules)
    runtime_form = cutfemx.fem.form(one * ds_runtime)
    runtime_value = cutfemx.fem.assemble_scalar(runtime_form)

    mixed_form = cutfemx.fem.form(
        one
        * ufl.Measure(
            "ds", domain=msh, subdomain_id=0, subdomain_data=[standard_facets, rules]
        )
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

    dS_runtime = ufl.Measure("dS", domain=msh, subdomain_data=rules)
    cut_form = cutfemx.fem.form(one * dS_runtime)
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

    dS_runtime = ufl.Measure("dS", domain=msh, subdomain_data=rules)
    runtime_form = cutfemx.fem.form(one * dS_runtime)
    runtime_value = cutfemx.fem.assemble_scalar(runtime_form)

    mixed_form = cutfemx.fem.form(
        one
        * ufl.Measure(
            "dS", domain=msh, subdomain_id=0, subdomain_data=[standard_facets, rules]
        )
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
        jump_u * jump_v * ufl.Measure("dS", domain=msh, subdomain_data=rules)
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


def test_cut_api_multiple_level_sets_accepts_cell_subset():
    msh, level_set = _line_level_set()
    V = level_set.function_space
    second = fem.Function(V, name="cap")
    second.interpolate(lambda x: x[1] - 0.51)
    cells = np.arange(11, dtype=np.int32)

    cutter = cutfemx.cut([level_set, second], cells, msh.topology.dim)
    selected = cutfemx.locate_entities(cutter, "phi=0 or cap=0")
    first = cutfemx.locate_entities(
        cutfemx.cut(level_set, cells, msh.topology.dim), "phi=0"
    )
    second_only = cutfemx.locate_entities(
        cutfemx.cut(second, cells, msh.topology.dim), "cap=0"
    )

    assert cutter.entity_dim == msh.topology.dim
    assert np.array_equal(cutter.entities, cells)
    assert set(selected.tolist()) == set(first.tolist()) | set(second_only.tolist())
    assert set(selected.tolist()).issubset(set(cells.tolist()))


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
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_data=rules)

    cut_form = cutfemx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_runtime)

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
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    one = fem.Constant(msh, np.float64(1.0))

    cut_form = cutfemx.fem.form(one * dx_runtime)
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


def test_cutfemx_block_deactivation_uses_per_row_active_domains():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    inside_cells = cutfemx.locate_entities(cutter, "phi<0")
    outside_cells = cutfemx.locate_entities(cutter, "phi>0")
    inside_rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    outside_rules = cutfemx.runtime_quadrature(cutter, "phi>0", order=2)
    dx_inside = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[inside_cells, inside_rules],
    )
    dx_outside = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=[outside_cells, outside_rules],
    )

    V1 = fem.functionspace(msh, ("Lagrange", 1))
    V2 = fem.functionspace(msh, ("Lagrange", 1))
    W = ufl.MixedFunctionSpace(V1, V2)
    u1, u2 = ufl.TrialFunctions(W)
    v1, v2 = ufl.TestFunctions(W)

    a = u1 * v1 * dx_inside + u2 * v2 * dx_outside
    a += 0.25 * u2 * v1 * dx_inside + 0.5 * u1 * v2 * dx_outside
    L = v1 * dx_inside + 2.0 * v2 * dx_outside

    a_blocks = ufl.extract_blocks(a)
    L_blocks = ufl.extract_blocks(L)
    a_forms = [
        [cutfemx.fem.form(a_blocks[i][j]) for j in range(2)] for i in range(2)
    ]
    L_forms = [cutfemx.fem.form(L_blocks[i]) for i in range(2)]

    A_blocks = [
        [cutfemx.fem.assemble_matrix(a_forms[i][j]) for j in range(2)]
        for i in range(2)
    ]
    for row in A_blocks:
        for A in row:
            A.scatter_reverse()
    b_blocks = [cutfemx.fem.assemble_vector(L_form) for L_form in L_forms]
    for b in b_blocks:
        b.scatter_reverse(la.InsertMode.add)

    domains = [
        cutfemx.fem.active_domain(a_forms[0][0]),
        cutfemx.fem.active_domain(a_forms[1][1]),
    ]
    zero_before = cutfemx.fem.zero_block_rows(A_blocks)
    assert set(domains[0].inactive_dofs.tolist()).issubset(
        set(zero_before[0].tolist())
    )
    assert set(domains[1].inactive_dofs.tolist()).issubset(
        set(zero_before[1].tolist())
    )

    for b, domain in zip(b_blocks, domains):
        b.array[domain.inactive_dofs] = 3.0
    returned = cutfemx.fem.deactivate_outside_blocks(A_blocks, domains, b_blocks)
    zero_after = cutfemx.fem.zero_block_rows(A_blocks)

    assert returned == domains
    assert all(rows.size == 0 for rows in zero_after)
    for i, domain in enumerate(domains):
        np.testing.assert_allclose(b_blocks[i].array[domain.inactive_dofs], 0.0)
        diagonal = A_blocks[i][i].to_scipy().diagonal()
        np.testing.assert_allclose(diagonal[domain.inactive_dofs], 1.0)


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


def test_cutfemx_level_set_value_assembles_runtime_cell_scalar():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi<0", order=2)
    assert cutfemx.level_set_value is cutfemx.level_set.level_set_value
    phi_q = cutfemx.level_set_value(level_set)

    dx_rules = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    form = cutfemx.fem.form(phi_q * phi_q * dx_rules)
    value = cutfemx.fem.assemble_scalar(form)

    assert form.needs_runtime_data
    assert np.isfinite(value)
    assert value > 0.0


def test_cutfemx_surface_normal_assembles_runtime_interface_scalar():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi=0", order=2)
    one = fem.Constant(msh, np.float64(1.0))
    n_h = cutfemx.surface_normal(cutter, "phi=0")

    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    measure_form = cutfemx.fem.form(one * dx_gamma)
    norm_form = cutfemx.fem.form(ufl.inner(n_h, n_h) * dx_gamma)
    orientation_form = cutfemx.fem.form(n_h[0] * dx_gamma)

    measure = cutfemx.fem.assemble_scalar(measure_form)
    norm_value = cutfemx.fem.assemble_scalar(norm_form)
    orientation = cutfemx.fem.assemble_scalar(orientation_form)

    assert norm_form.needs_runtime_data
    assert np.isfinite(norm_value)
    np.testing.assert_allclose(norm_value, measure, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(orientation, measure, rtol=1.0e-12, atol=1.0e-12)


def test_cutfemx_normal_matches_quadratic_circle_radial_normal():
    msh, level_set, center = _quadratic_circle_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi=0", order=5)
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    x = ufl.SpatialCoordinate(msh)
    n_q = cutfemx.normal(level_set)
    n_exact = _radial_normal(x, center)

    error_form = cutfemx.fem.form(ufl.inner(n_q - n_exact, n_q - n_exact) * dx_gamma)
    error = cutfemx.fem.assemble_scalar(error_form)

    assert error_form.needs_runtime_data
    np.testing.assert_allclose(error, 0.0, atol=1.0e-24)


def test_cutfemx_conormal_matches_projected_facet_normal_on_quadratic_circle():
    msh, level_set, center = _quadratic_circle_level_set()

    cell_cut = cutfemx.cut(level_set)
    cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
    facets = cutfemx.interior_facets_for_cells(msh, cut_cells)
    facet_cut = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    rules = cutfemx.runtime_quadrature(facet_cut, "phi=0", order=5)
    dS_gamma = ufl.Measure("dS", domain=msh, subdomain_data=rules)

    x = ufl.SpatialCoordinate(msh)
    n_exact = _radial_normal(x, center)
    n_facet = ufl.FacetNormal(msh)
    n_q = cutfemx.normal(level_set)
    mu = cutfemx.conormal(n_q)
    mu_exact_p = _projected_conormal(n_facet("+"), n_exact("+"))
    mu_exact_m = _projected_conormal(n_facet("-"), n_exact("-"))

    plus_form = cutfemx.fem.form(
        ufl.inner(mu("+") - mu_exact_p, mu("+") - mu_exact_p) * dS_gamma
    )
    minus_form = cutfemx.fem.form(
        ufl.inner(mu("-") - mu_exact_m, mu("-") - mu_exact_m) * dS_gamma
    )
    opposing_form = cutfemx.fem.form(
        ufl.inner(mu("+") + mu("-"), mu("+") + mu("-")) * dS_gamma
    )

    plus_error = cutfemx.fem.assemble_scalar(plus_form)
    minus_error = cutfemx.fem.assemble_scalar(minus_form)
    opposing = cutfemx.fem.assemble_scalar(opposing_form)

    assert plus_form.needs_runtime_data
    np.testing.assert_allclose(plus_error, 0.0, atol=1.0e-24)
    np.testing.assert_allclose(minus_error, 0.0, atol=1.0e-24)
    np.testing.assert_allclose(opposing, 0.0, atol=1.0e-24)


def test_cutfemx_conormal_matches_projected_facet_normal_on_quadratic_sphere():
    msh, level_set, center = _quadratic_sphere_level_set()

    cell_cut = cutfemx.cut(level_set)
    cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
    facets = cutfemx.interior_facets_for_cells(msh, cut_cells)
    facet_cut = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    rules = cutfemx.runtime_quadrature(
        facet_cut, "phi=0", order=4, backend="algoim"
    )
    dS_gamma = ufl.Measure("dS", domain=msh, subdomain_data=rules)

    x = ufl.SpatialCoordinate(msh)
    n_exact = _radial_normal(x, center)
    n_facet = ufl.FacetNormal(msh)
    n_q = cutfemx.normal(level_set)
    mu = cutfemx.conormal(n_q)
    mu_exact_p = _projected_conormal(n_facet("+"), n_exact("+"))
    mu_exact_m = _projected_conormal(n_facet("-"), n_exact("-"))

    plus_form = cutfemx.fem.form(
        ufl.inner(mu("+") - mu_exact_p, mu("+") - mu_exact_p) * dS_gamma
    )
    minus_form = cutfemx.fem.form(
        ufl.inner(mu("-") - mu_exact_m, mu("-") - mu_exact_m) * dS_gamma
    )
    opposing_form = cutfemx.fem.form(
        ufl.inner(mu("+") + mu("-"), mu("+") + mu("-")) * dS_gamma
    )

    plus_error = cutfemx.fem.assemble_scalar(plus_form)
    minus_error = cutfemx.fem.assemble_scalar(minus_form)
    opposing = cutfemx.fem.assemble_scalar(opposing_form)

    assert plus_form.needs_runtime_data
    np.testing.assert_allclose(plus_error, 0.0, atol=1.0e-24)
    np.testing.assert_allclose(minus_error, 0.0, atol=1.0e-24)
    np.testing.assert_allclose(opposing, 0.0, atol=1.0e-24)


def test_cutfemx_correction_distance_zero_on_linear_interface():
    msh, level_set = _line_level_set()

    cutter = cutfemx.cut(level_set)
    rules = cutfemx.runtime_quadrature(cutter, "phi=0", order=2)
    n_h = cutfemx.surface_normal(cutter, "phi=0")
    rho = cutfemx.correction_distance(
        level_set,
        cutter,
        "phi=0",
        direction=n_h,
    )

    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_data=rules)
    form = cutfemx.fem.form(rho * rho * dx_gamma)
    value = cutfemx.fem.assemble_scalar(form)

    assert form.needs_runtime_data
    np.testing.assert_allclose(value, 0.0, atol=1.0e-24)


def test_cutfemx_conormal_assembles_runtime_surface_skeleton_scalar():
    msh, level_set = _line_level_set()

    facets = _interior_facet_indices(msh)
    facet_cut = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    rules = cutfemx.runtime_quadrature(facet_cut, "phi=0", order=2)
    one = fem.Constant(msh, np.float64(1.0))
    n_q = cutfemx.normal(level_set)
    mu = cutfemx.conormal(n_q)

    dS_gamma = ufl.Measure("dS", domain=msh, subdomain_data=rules)
    measure_form = cutfemx.fem.form(one * dS_gamma)
    unit_form = cutfemx.fem.form(ufl.inner(mu("+"), mu("+")) * dS_gamma)
    orthogonal_form = cutfemx.fem.form((ufl.dot(mu("+"), n_q("+")) ** 2) * dS_gamma)
    opposing_form = cutfemx.fem.form(
        ufl.inner(mu("+") + mu("-"), mu("+") + mu("-")) * dS_gamma
    )

    measure = cutfemx.fem.assemble_scalar(measure_form)
    unit = cutfemx.fem.assemble_scalar(unit_form)
    orthogonal = cutfemx.fem.assemble_scalar(orthogonal_form)
    opposing = cutfemx.fem.assemble_scalar(opposing_form)

    assert unit_form.needs_runtime_data
    assert measure > 0.0
    np.testing.assert_allclose(unit, measure, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(orthogonal, 0.0, atol=1.0e-24)
    np.testing.assert_allclose(opposing, 0.0, atol=1.0e-24)


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


def test_cut_api_interior_facets_for_cells_stays_inside_cell_set():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_to_cell = msh.topology.connectivity(fdim, tdim)
    num_cells = msh.topology.index_map(tdim).size_local

    selected = np.arange(0, num_cells, 2, dtype=np.int32)
    selected_set = set(selected.tolist())
    facets = cutfemx.interior_facets_for_cells(msh, selected)

    for facet in facets:
        adjacent = [int(c) for c in facet_to_cell.links(int(facet))]
        assert len(adjacent) == 2
        assert all(c in selected_set for c in adjacent)

    all_cells = np.arange(num_cells, dtype=np.int32)
    all_facets = cutfemx.interior_facets_for_cells(msh, all_cells)
    assert np.array_equal(all_facets, _interior_facet_indices(msh))


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


def test_cutfemx_active_domain_supports_mixed_space():
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

    import basix.ufl

    velocity = basix.ufl.element(
        "Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)
    )
    pressure = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([velocity, pressure]))
    w = ufl.TrialFunction(W)
    z = ufl.TestFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(z)

    a = cutfemx.fem.form((ufl.inner(u, v) + p * q) * dx_inside)
    L = cutfemx.fem.form((v[0] + q) * dx_inside)
    A = cutfemx.fem.assemble_matrix(a)
    A.scatter_reverse()
    b = cutfemx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    domain = cutfemx.fem.active_domain(a)
    returned = cutfemx.fem.deactivate_outside(A, b, domain)

    expected_active_cells = np.unique(
        np.concatenate([inside_cells, rules.parent_map])
    )
    assert np.array_equal(domain.active_cells, expected_active_cells)
    assert domain.inactive_dofs.size > 0
    assert returned is domain
    np.testing.assert_allclose(b.array[domain.inactive_dofs], 0.0)


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


@pytest.mark.parametrize("backend", ["algoim", "algoim_general"])
def test_cutfemx_runtime_quadrature_algoim_rejects_simplex_hosts(backend):
    triangle_mesh = mesh.create_unit_square(
        MPI.COMM_WORLD, 2, 2, cell_type=mesh.CellType.triangle
    )
    tetrahedron_mesh = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [2, 2, 2],
        mesh.CellType.tetrahedron,
    )

    for msh in (triangle_mesh, tetrahedron_mesh):
        V_phi = fem.functionspace(msh, ("Lagrange", 1))
        level_set = fem.Function(V_phi)
        level_set.interpolate(lambda x: x[0] - 0.5)
        cutter = cutfemx.cut(level_set)

        with pytest.raises(
            ValueError, match="interval, quadrilateral, or hexahedron"
        ):
            cutfemx.runtime_quadrature(
                cutter, "phi<0", order=4, backend=backend
            )

        with pytest.raises(
            ValueError, match="interval, quadrilateral, or hexahedron"
        ):
            cutfemx.runtime_quadratures(
                cutter, ["phi<0", "phi=0"], order=4, backend=backend
            )


def test_cutfemx_runtime_quadratures_algoim_paired_selectors():
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, cell_type=mesh.CellType.quadrilateral
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 2))
    level_set = fem.Function(V_phi)
    level_set.interpolate(
        lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 0.2**2
    )

    cutter = cutfemx.cut(level_set)
    try:
        inside = cutfemx.runtime_quadrature(
            cutter, "phi<0", order=4, backend="algoim"
        )
        interface = cutfemx.runtime_quadrature(
            cutter, "phi=0", order=4, backend="algoim"
        )
        paired = cutfemx.runtime_quadratures(
            cutter, ["phi<0", "phi>0", "phi=0"], order=4, backend="algoim"
        )
    except RuntimeError as exc:
        if "without Algoim" in str(exc):
            pytest.skip("CutFEMx was built without Algoim support")
        raise

    assert set(paired) == {"phi<0", "phi>0", "phi=0"}
    assert np.sum(paired["phi<0"].weights) == pytest.approx(np.sum(inside.weights))
    assert np.sum(paired["phi=0"].weights) == pytest.approx(
        np.sum(interface.weights)
    )
    assert len(paired["phi>0"].weights) > 0


def test_cutfemx_runtime_quadrature_algoim_interval_interface_on_facets():
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, cell_type=mesh.CellType.quadrilateral
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 2))
    level_set = fem.Function(V_phi)
    level_set.interpolate(lambda x: (x[0] - 0.37) * (x[0] + 0.5))

    facets = _interior_facet_indices(msh)
    facet_cut = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    try:
        rules = cutfemx.runtime_quadrature(
            facet_cut, "phi=0", order=4, backend="algoim"
        )
    except RuntimeError as exc:
        if "without Algoim" in str(exc):
            pytest.skip("CutFEMx was built without Algoim support")
        raise

    assert rules.weights.size > 0
    np.testing.assert_allclose(rules.weights, 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        rules.physical_points[0],
        0.37,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_cutfemx_runtime_quadrature_algoim_embedded_quad_interface_on_3d_facets():
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (2, 2, 2),
        cell_type=mesh.CellType.hexahedron,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 2))
    level_set = fem.Function(V_phi)
    level_set.interpolate(lambda x: x[0] - 0.37)

    facets = _interior_facet_indices(msh)
    facet_cut = cutfemx.cut(level_set, facets, msh.topology.dim - 1)
    try:
        rules = cutfemx.runtime_quadrature(
            facet_cut, "phi=0", order=4, backend="algoim"
        )
    except RuntimeError as exc:
        if "without Algoim" in str(exc):
            pytest.skip("CutFEMx was built without Algoim support")
        raise

    assert rules.weights.size > 0
    assert np.all(rules.weights > 0.0)
    np.testing.assert_allclose(
        np.sum(rules.weights),
        2.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    rules.with_physical_points()
    np.testing.assert_allclose(
        rules.physical_points[0],
        0.37,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
