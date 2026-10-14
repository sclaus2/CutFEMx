"""Compressed runtime quadrature: fewer points, the same moments."""

import numpy as np
import pytest
from mpi4py import MPI

import cutfemx
from dolfinx import fem, mesh


def _sphere_cut(cell_type, n=5):
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=cell_type)
    V = fem.functionspace(msh, ("Lagrange", 2))
    phi = fem.Function(V)
    phi.interpolate(
        lambda x: (x[0] - 0.47) ** 2 + (x[1] - 0.52) ** 2 + (x[2] - 0.49) ** 2 - 0.31**2
    )
    return cutfemx.cut(phi)


def _exponents(degree, space):
    r = range(degree + 1)
    return [
        (a, b, c)
        for a in r
        for b in r
        for c in r
        if space == "tensor" or a + b + c <= degree
    ]


def _moments(rules, exponents):
    """Per rule, the sum of weight times each reference monomial."""
    x = np.asarray(rules.points).reshape(-1, 3)
    w = np.asarray(rules.weights)
    offsets = np.asarray(rules.offsets)
    columns = []
    for a, b, c in exponents:
        values = w * x[:, 0] ** a * x[:, 1] ** b * x[:, 2] ** c
        columns.append(np.add.reduceat(values, offsets[:-1]))
    return np.column_stack(columns)


@pytest.mark.parametrize(
    "cell_type, space",
    [(mesh.CellType.hexahedron, "tensor"), (mesh.CellType.tetrahedron, "total")],
)
@pytest.mark.parametrize("part", ["phi<0", "phi=0"])
@pytest.mark.parametrize("degree", [2, 4])
def test_compressed_rules_keep_moments(cell_type, space, part, degree):
    cut = _sphere_cut(cell_type)
    full = cutfemx.runtime_quadrature(cut, part, 6, backend="quadrays")
    small = cutfemx.runtime_quadrature(
        cut, part, 6, backend="quadrays", compress=degree
    )
    stats = small.compression
    n_rules = len(np.asarray(full.parent_map))
    assert stats["n_rules"] == n_rules
    assert stats["points_before"] == len(np.asarray(full.weights))
    assert stats["points_after"] == len(np.asarray(small.weights))
    assert stats["points_after"] < stats["points_before"]
    np.testing.assert_array_equal(small.parent_map, full.parent_map)
    assert np.all(np.asarray(small.weights) > 0)

    exponents = _exponents(degree, space)
    counts = np.diff(np.asarray(small.offsets))
    assert counts.max() <= len(exponents)
    scale = np.add.reduceat(np.abs(full.weights), np.asarray(full.offsets)[:-1])
    error = np.abs(_moments(small, exponents) - _moments(full, exponents))
    assert np.max(error / scale[:, None]) < 1e-9
    assert np.sum(small.weights) == pytest.approx(np.sum(full.weights), rel=1e-12)


def test_compressed_straight_interface_is_refused():
    cut = _sphere_cut(mesh.CellType.hexahedron)
    with pytest.raises((ValueError, RuntimeError)):
        cutfemx.runtime_quadrature(cut, "phi=0", 4, backend="straight", compress=2)
