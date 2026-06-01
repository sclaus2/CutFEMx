from mpi4py import MPI

import numpy as np
from cutfemx.distance import (
    SignMode,
    build_cell_triangle_map,
    compute_stl_bbox,
    compute_unsigned_distance,
    distribute_stl,
    from_stl,
    reinitialize,
)

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_box, create_rectangle


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
