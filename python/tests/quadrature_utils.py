# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

import numpy as np
import basix
from runintgen import QuadratureRules


def runtime_quadrature_mesh(mesh, order: int) -> QuadratureRules:
    """Create full-cell runtime quadrature for tests."""
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    points_ref, weights_ref = basix.make_quadrature(
        mesh.basix_cell(),
        order,
        basix.QuadratureType.default,
        basix.PolysetType.standard,
    )
    points_ref = np.ascontiguousarray(points_ref, dtype=np.float64)
    weights_ref = np.ascontiguousarray(weights_ref, dtype=np.float64)

    cmap = mesh.geometry.cmaps[0]
    element = basix.create_element(
        basix.ElementFamily.P,
        mesh.basix_cell(),
        cmap.degree,
        basix.LagrangeVariant(cmap.variant),
    )
    phi = element.tabulate(1, points_ref)
    dphi = phi[1:, :, :, 0].transpose(1, 2, 0)

    x_dofmap = mesh.geometry.dofmaps[0]
    x_coords = mesh.geometry.x
    num_cells = mesh.topology.index_map(tdim).size_local
    num_points_per_cell = weights_ref.size

    points = np.tile(points_ref, (num_cells, 1))
    weights = np.empty(num_cells * num_points_per_cell, dtype=np.float64)
    offsets = np.arange(
        num_cells + 1, dtype=np.int32
    ) * np.int32(num_points_per_cell)
    parent_map = np.arange(num_cells, dtype=np.int32)

    for cell in range(num_cells):
        try:
            cell_dofs = x_dofmap.links(cell)
        except AttributeError:
            cell_dofs = x_dofmap[cell]
        cell_coords = x_coords[cell_dofs, :gdim]
        J = np.einsum("kd,pkt->pdt", cell_coords, dphi)
        if gdim == tdim:
            detJ = np.abs(np.linalg.det(J))
        else:
            JTJ = np.einsum("pdi,pdj->pij", J, J)
            detJ = np.sqrt(np.linalg.det(JTJ))
        q0 = cell * num_points_per_cell
        q1 = q0 + num_points_per_cell
        weights[q0:q1] = weights_ref * detJ

    return QuadratureRules(
        kind="per_entity",
        tdim=tdim,
        points=np.ascontiguousarray(points),
        weights=np.ascontiguousarray(weights),
        offsets=offsets,
        parent_map=parent_map,
    )
