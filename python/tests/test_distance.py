# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI

import numpy as np

import pytest
import cutfemx
from cutfemx.distance import (
    NormalExtensionResult,
    SignMode,
    build_cell_triangle_map,
    compute_stl_bbox,
    compute_unsigned_distance,
    distribute_stl,
    extend_normal_velocity,
    from_stl,
    reinitialize,
    reinitialize_from_facets,
)

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, GhostMode, create_box, create_rectangle


def test_reinitialize_updates_parabolic_level_set():
    mesh = create_rectangle(
        MPI.COMM_SELF,
        [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
        [16, 16],
        CellType.triangle,
    )
    V = functionspace(mesh, ("Lagrange", 1))
    phi = Function(V)

    radius = 0.3
    phi.interpolate(lambda x: x[0] ** 2 + x[1] ** 2 - radius**2)
    before = phi.x.array.copy()

    dof_coords = V.tabulate_dof_coordinates()
    exact = np.sqrt(dof_coords[:, 0] ** 2 + dof_coords[:, 1] ** 2) - radius
    pre_error = np.max(np.abs(before - exact))

    reinitialize(phi, max_iter=500, tol=1e-10)

    after = phi.x.array
    post_error = np.max(np.abs(after - exact))

    assert np.max(np.abs(after - before)) > 1.0e-3
    assert post_error < 0.5 * pre_error


def test_quad_reinitialize_updates_parabolic_level_set():
    mesh = create_rectangle(
        MPI.COMM_SELF,
        [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
        [16, 16],
        CellType.quadrilateral,
    )
    V = functionspace(mesh, ("Lagrange", 1))
    phi = Function(V)

    radius = 0.3
    phi.interpolate(lambda x: x[0] ** 2 + x[1] ** 2 - radius**2)
    before = phi.x.array.copy()

    dof_coords = V.tabulate_dof_coordinates()
    exact = np.sqrt(dof_coords[:, 0] ** 2 + dof_coords[:, 1] ** 2) - radius
    pre_error = np.max(np.abs(before - exact))

    reinitialize(phi, max_iter=500, tol=1e-10)

    after = phi.x.array
    post_error = np.max(np.abs(after - exact))

    assert np.max(np.abs(after - before)) > 1.0e-3
    assert post_error < 0.5 * pre_error


def test_reinitialize_from_facets_uses_facets_as_zero_interface():
    mesh = create_rectangle(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [4, 4],
        CellType.triangle,
    )
    V = functionspace(mesh, ("Lagrange", 1))
    phi = Function(V)

    coords = V.tabulate_dof_coordinates()
    phi.x.array[:] = coords[:, 0] - 0.5
    phi.x.array[np.isclose(coords[:, 0], 0.5)] = 0.25
    phi.x.scatter_forward()

    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f_to_v = mesh.topology.connectivity(fdim, 0)
    points = mesh.geometry.x
    facets = []
    for facet in range(mesh.topology.index_map(fdim).size_local):
        vertices = f_to_v.links(facet)
        if np.allclose(points[vertices, 0], 0.5):
            facets.append(facet)

    assert facets

    reinitialize_from_facets(phi, np.asarray(facets, dtype=np.int32), max_iter=500)

    exact = coords[:, 0] - 0.5
    np.testing.assert_allclose(phi.x.array, exact, atol=1.0e-12)


def _write_cube_stl(path, lo=0.35, hi=0.65):
    p = {
        "000": (lo, lo, lo),
        "001": (lo, lo, hi),
        "010": (lo, hi, lo),
        "011": (lo, hi, hi),
        "100": (hi, lo, lo),
        "101": (hi, lo, hi),
        "110": (hi, hi, lo),
        "111": (hi, hi, hi),
    }
    triangles = [
        ("000", "010", "011"),
        ("000", "011", "001"),
        ("100", "101", "111"),
        ("100", "111", "110"),
        ("000", "001", "101"),
        ("000", "101", "100"),
        ("010", "110", "111"),
        ("010", "111", "011"),
        ("000", "100", "110"),
        ("000", "110", "010"),
        ("001", "011", "111"),
        ("001", "111", "101"),
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("solid cube\n")
        for tri in triangles:
            f.write("facet normal 0 0 0\nouter loop\n")
            for key in tri:
                x, y, z = p[key]
                f.write(f"vertex {x} {y} {z}\n")
            f.write("endloop\nendfacet\n")
        f.write("endsolid cube\n")


def test_component_anchor_applies_stl_sign(tmp_path):
    stl_path = tmp_path / "cube.stl"
    _write_cube_stl(stl_path)

    mesh = create_box(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [8, 8, 8],
        CellType.tetrahedron,
    )

    dist = from_stl(mesh, str(stl_path), sign_mode=SignMode.ComponentAnchor)
    values = dist.x.array

    assert np.min(values) < 0.0
    assert np.max(values) > 0.0


def test_hex_stl_distance_has_finite_signed_values(tmp_path):
    stl_path = tmp_path / "cube.stl"
    _write_cube_stl(stl_path)

    mesh = create_box(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [6, 6, 6],
        CellType.hexahedron,
    )

    dist = from_stl(mesh, str(stl_path), sign_mode=SignMode.LocalNormalBand)
    values = dist.x.array

    assert np.all(np.isfinite(values))
    assert np.min(values) < 0.0
    assert np.max(values) > 0.0


def test_unsigned_distance_pipeline_uses_distributed_stl_wrappers(tmp_path):
    stl_path = tmp_path / "cube.stl"
    _write_cube_stl(stl_path)

    bbox = np.asarray(compute_stl_bbox(str(stl_path)))
    np.testing.assert_allclose(
        bbox,
        np.array([[0.35, 0.35, 0.35], [0.65, 0.65, 0.65]]),
        rtol=0.0,
        atol=1.0e-7,
    )

    mesh = create_box(
        MPI.COMM_SELF,
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])],
        [4, 4, 4],
        CellType.tetrahedron,
    )
    V = functionspace(mesh, ("Lagrange", 1))
    dist = Function(V)

    soup = distribute_stl(mesh, str(stl_path), aabb_padding=0.05)
    cell_map = build_cell_triangle_map(mesh, soup, aabb_padding=0.05)
    compute_unsigned_distance(dist, soup, cell_map, max_iter=500, tol=1.0e-10)

    values = dist.x.array
    assert np.all(np.isfinite(values))
    assert np.min(values) >= -1.0e-12
    assert np.min(values) < 0.2
    assert np.max(values) > 0.0


def _build_circle_extension(comm=MPI.COMM_SELF, *, n=24, target_degree=None):
    radius = 0.45
    mesh = create_rectangle(
        comm,
        [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],
        [n, n],
        CellType.quadrilateral,
        ghost_mode=GhostMode.shared_facet,
    )
    V = functionspace(mesh, ("Lagrange", 1))
    phi = Function(V, name="phi")
    phi.interpolate(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) - radius)
    phi.x.scatter_forward()

    cut_data = cutfemx.cut(phi)
    interface_cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi=0", mode="cut_only")
    has_interface = interface_cut_mesh.mesh is not None
    if comm.allreduce(int(has_interface), op=MPI.MIN) == 0:
        return mesh, V, phi, None, radius

    V_gamma = functionspace(interface_cut_mesh.mesh, ("Lagrange", 2))
    interface_speed = Function(V_gamma, name="interface_speed")
    interface_speed.interpolate(
        lambda x: 1.0 + 0.25 * np.cos(3.0 * np.arctan2(x[1], x[0]))
    )
    interface_speed.x.scatter_forward()

    target_space = None
    if target_degree is not None:
        target_space = functionspace(mesh, ("Lagrange", target_degree))

    result = extend_normal_velocity(
        phi,
        interface_speed,
        interface_cut_mesh,
        k_ring=1,
        target_space=target_space,
        max_iter=500,
        tol=1.0e-10,
    )
    return mesh, result.speed.function_space, phi, result, radius


def test_extension_velocity_matches_circle_speed_near_interface():
    mesh, V, _, result, radius = _build_circle_extension()
    assert isinstance(result, NormalExtensionResult)

    coords = V.tabulate_dof_coordinates()
    exact_distance = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) - radius
    band = np.abs(exact_distance) < 2.5 / 24
    assert np.any(exact_distance[band] < 0.0)
    assert np.any(exact_distance[band] > 0.0)

    theta = np.arctan2(coords[:, 1], coords[:, 0])
    exact_speed = 1.0 + 0.25 * np.cos(3.0 * theta)
    np.testing.assert_allclose(result.speed.x.array[band], exact_speed[band], atol=0.1)

    velocity = result.velocity.x.array.reshape((-1, 2))
    velocity_norm = np.linalg.norm(velocity, axis=1)
    np.testing.assert_allclose(velocity_norm, result.speed.x.array, atol=1.0e-12)
    assert np.min(result.signed_distance.x.array) < 0.0
    assert np.max(result.signed_distance.x.array) > 0.0


def test_higher_order_extension_output_interpolates_sampled_speed():
    _, V, _, result, radius = _build_circle_extension(target_degree=2)

    assert result.speed.function_space.element.basix_element.degree == 2
    assert result.velocity.function_space.value_shape == (2,)

    coords = V.tabulate_dof_coordinates()
    exact_distance = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) - radius
    band = np.abs(exact_distance) < 1.5 / 24
    theta = np.arctan2(coords[:, 1], coords[:, 0])
    exact_speed = 1.0 + 0.25 * np.cos(3.0 * theta)

    np.testing.assert_allclose(result.speed.x.array[band], exact_speed[band], atol=0.08)
    assert np.min(result.speed.x.array) >= 0.70
    assert np.max(result.speed.x.array) <= 1.30


def test_normal_extension_payload_values_are_consistent_across_ghosts():
    comm = MPI.COMM_WORLD
    _, V, _, result, _ = _build_circle_extension(comm, n=16)
    if result is None:
        pytest.skip("normal-extension MPI check requires every rank to own cut cells")

    coords = V.tabulate_dof_coordinates()[:, :2]
    speed = np.asarray(result.speed.x.array, dtype=float)
    velocity = np.asarray(result.velocity.x.array, dtype=float).reshape((-1, 2))
    local_rows = [
        (
            tuple(np.round(coords[i], 12)),
            float(speed[i]),
            float(velocity[i, 0]),
            float(velocity[i, 1]),
        )
        for i in range(speed.size)
    ]

    gathered = comm.gather(local_rows, root=0)
    if comm.rank == 0:
        by_point = {}
        for rows in gathered:
            for row in rows:
                by_point.setdefault(row[0], []).append(row[1:])

        for values in by_point.values():
            if len(values) < 2:
                continue
            data = np.asarray(values, dtype=float)
            assert np.max(np.ptp(data, axis=0)) < 1.0e-11
