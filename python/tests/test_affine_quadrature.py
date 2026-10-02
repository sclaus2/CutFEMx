# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Quadrature degrees of cut forms on affine quadrilateral and hexahedral meshes."""

from mpi4py import MPI

import cutfemx
import cutfemx.fem
import numpy as np
import pytest
from cutfemx.fem import _is_affine_tensor_product_mesh, _with_affine_quadrature_degrees

import ufl
from dolfinx import fem, mesh


def _box(cell_type, n=4):
    if cell_type in (mesh.CellType.quadrilateral, mesh.CellType.triangle):
        return mesh.create_rectangle(MPI.COMM_WORLD, [[-1.0, -1.0], [1.0, 1.0]], [n, n], cell_type)
    return mesh.create_box(
        MPI.COMM_WORLD, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], [n, n, n], cell_type
    )


def _perturb(msh):
    x = msh.geometry.x
    x[:, 0] += 0.05 * np.sin(3.0 * x[:, 1]) * np.cos(2.0 * x[:, 0])


def _cut_measures(msh, order=3):
    phi = fem.Function(fem.functionspace(msh, ("Lagrange", 1)))
    phi.interpolate(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) - 0.63)
    cut_data = cutfemx.cut(phi)
    inside = cutfemx.locate_entities(cut_data, "phi<0")
    rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    ghost = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
    dx = ufl.Measure("dx", domain=msh, subdomain_id=0, subdomain_data=[inside, rules])
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_id=0, subdomain_data=rules)
    dS = ufl.Measure("dS", domain=msh, subdomain_id=1, subdomain_data=ghost)
    return dx, dx_runtime, dS


def _ghost_penalty(u, v, n, dS):
    def d2n(w):
        H = ufl.grad(ufl.grad(w))
        return ufl.dot(ufl.dot(H("+") - H("-"), n("+")), n("+"))

    return (
        ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * dS
        + 0.1 * ufl.inner(d2n(u), d2n(v)) * dS
    )


@pytest.mark.parametrize(
    "cell_type, affine",
    [
        (mesh.CellType.quadrilateral, True),
        (mesh.CellType.hexahedron, True),
        (mesh.CellType.triangle, False),
        (mesh.CellType.tetrahedron, False),
    ],
)
def test_affine_tensor_product_mesh_detection(cell_type, affine):
    msh = _box(cell_type)
    assert _is_affine_tensor_product_mesh(msh) is affine

    # A sheared box is still affine, a curved one is not.
    x = msh.geometry.x
    x[:, 0] += 0.3 * x[:, 1]
    assert _is_affine_tensor_product_mesh(msh) is affine
    _perturb(msh)
    assert not _is_affine_tensor_product_mesh(msh)


def test_affine_quadrature_degrees_replace_geometric_inflation():
    msh = _box(mesh.CellType.hexahedron, 3)
    V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)
    dx, dx_runtime, dS = _cut_measures(msh)

    def degrees(form):
        return [
            itg.metadata().get("quadrature_degree")
            for itg in _with_affine_quadrature_degrees(form, msh).integrals()
        ]

    # UFL estimates 7, 12 and 14 for these Q2 integrals on hexahedra.
    assert degrees(ufl.inner(ufl.grad(u), ufl.grad(v)) * dx) == [4]
    assert degrees(_ghost_penalty(u, v, n, dS)) == [4, 4]
    # Mixed integrals keep a rule that classifies the Q2 tables exactly.
    assert degrees(ufl.inner(x, v) * dx) == [4]
    # Explicit degrees and runtime-only integrals are left alone.
    assert degrees(ufl.inner(u, v) * ufl.dx(domain=msh, metadata={"quadrature_degree": 9})) == [9]
    assert degrees(ufl.inner(u, v) * dx_runtime) == [None]

    _perturb(msh)
    assert degrees(_ghost_penalty(u, v, n, dS)) == [None, None]


@pytest.mark.parametrize(
    "cell_type, degree", [(mesh.CellType.quadrilateral, 2), (mesh.CellType.hexahedron, 1)]
)
def test_affine_quadrature_degrees_preserve_matrix(cell_type, degree):
    """Lower degrees are exact for polynomial integrands on affine cells."""
    msh = _box(cell_type, 4)
    gdim = msh.geometry.dim
    V = fem.functionspace(msh, ("Lagrange", degree, (gdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(msh)
    x = ufl.SpatialCoordinate(msh)
    dx, _, dS = _cut_measures(msh)
    a = (
        ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * dx
        + (1.0 + x[0] ** 2) * ufl.inner(u, v) * dx
        + _ghost_penalty(u, v, n, dS)
    )

    matrices = []
    for affine in (True, False):
        a_form = cutfemx.fem.form(a, affine_quadrature_degrees=affine)
        A = cutfemx.fem.assemble_matrix(a_form)
        A.scatter_reverse()
        matrices.append(A.to_dense())
        assert all(
            integral.metadata().get("quadrature_degree") is None for integral in a.integrals()
        )

    scale = np.abs(matrices[1]).max()
    assert scale > 0.0
    assert np.abs(matrices[0] - matrices[1]).max() <= 1e-11 * scale
