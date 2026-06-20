# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import basix.ufl
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import (
    GhostMode,
    Mesh,
    RefinementOption,
    create_cell_partitioner,
    refine,
)

from cutfemx import cutfemx_cpp as _cpp

__all__ = [
    "NormalExtensionResult",
    "SignMode",
    "adapt_mesh_to_stl",
    "build_cell_triangle_map",
    "compute_distance_from_stl",
    "compute_signed_distance",
    "compute_stl_bbox",
    "compute_unsigned_distance",
    "distribute_stl",
    "distribute_stl_facets",
    "from_stl",
    "extend_normal_velocity",
    "reinitialize",
]


SignMode = _cpp.distance.SignMode


@dataclass(frozen=True)
class NormalExtensionResult:
    """Background fields produced by normal-direction velocity extension."""

    speed: Function
    velocity: Function
    signed_distance: Function


def _element_degree(V: Any) -> int:
    degree = V.ufl_element().degree
    return int(degree() if callable(degree) else degree)


def _vector_lagrange_space(mesh: Mesh, degree: int):
    element = basix.ufl.element(
        "Lagrange",
        mesh.basix_cell(),
        degree,
        shape=(mesh.geometry.dim,),
        dtype=mesh.geometry.x.dtype,
    )
    return functionspace(mesh, element)


def compute_stl_bbox(path: str):
    """Compute the STL axis-aligned bounding box without distributing facets."""
    return _cpp.distance.compute_stl_bbox(path)


def distribute_stl(mesh: Mesh, stl_path: str, aabb_padding: float = 0.05):
    """Read and distribute STL facets across the mesh communicator."""
    return _cpp.distance.distribute_stl(
        mesh.comm, mesh._cpp_object, stl_path, aabb_padding
    )


def build_cell_triangle_map(mesh: Mesh, soup, aabb_padding: float = 0.0):
    """Build the local cell-to-STL-triangle map used for near-field distances."""
    return _cpp.distance.build_cell_triangle_map(
        mesh._cpp_object, soup, aabb_padding
    )


def compute_unsigned_distance(
    dist_func: Function,
    soup,
    cell_map,
    max_iter: int = 1000,
    tol: float = 1.0e-10,
) -> None:
    """Fill ``dist_func`` with unsigned surface distance values."""
    _cpp.distance.compute_unsigned_distance(
        dist_func._cpp_object, soup, cell_map, max_iter, tol
    )


def compute_signed_distance(
    dist_func: Function,
    soup,
    cell_map,
    sign_mode: SignMode = SignMode.ComponentAnchor,
    max_iter: int = 1000,
    tol: float = 1.0e-10,
) -> None:
    """Fill ``dist_func`` with signed surface distance values."""
    _cpp.distance.compute_signed_distance(
        dist_func._cpp_object, soup, cell_map, sign_mode, max_iter, tol
    )


def from_stl(
    mesh: Mesh,
    stl_path: str,
    sign_mode: SignMode = SignMode.ComponentAnchor,
    max_iter: int = 1000,
    tol: float = 1.0e-10,
    aabb_padding: float = 0.05,
) -> Function:
    """Compute a P1 signed distance field from an STL surface."""
    rank = mesh.comm.Get_rank()
    t_start = time.time()

    V = functionspace(mesh, ("Lagrange", 1))
    dist_func = Function(V)
    t_space = time.time()

    soup = distribute_stl(mesh, stl_path, aabb_padding)
    t_dist = time.time()

    cell_map = build_cell_triangle_map(mesh, soup, aabb_padding)
    t_map = time.time()

    compute_signed_distance(dist_func, soup, cell_map, sign_mode, max_iter, tol)
    t_compute = time.time()

    if rank == 0:
        logger = logging.getLogger("cutfemx")
        logger.info("  [Timing] FunctionSpace: %.4fs", t_space - t_start)
        logger.info("  [Timing] Distribute:    %.4fs", t_dist - t_space)
        logger.info("  [Timing] BuildMap:      %.4fs", t_map - t_dist)
        logger.info("  [Timing] FIM+Sign:      %.4fs", t_compute - t_map)

    return dist_func


def reinitialize(
    phi: Function,
    max_iter: int = 200,
    tol: float = 1.0e-10,
) -> None:
    """Reinitialize a level-set function in place as a signed distance field."""
    _cpp.distance.reinitialize(phi._cpp_object, max_iter, tol)


def extend_normal_velocity(
    phi: Function,
    interface_speed: Function,
    interface_cut_mesh,
    *,
    k_ring: int = 1,
    target_space=None,
    normal_sign: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1.0e-10,
) -> NormalExtensionResult:
    """Extend scalar interface speed along normals into the background mesh.

    The C++ carrier solve is first-order and vertex based. If ``target_space``
    is supplied, the carrier fields are interpolated into that scalar Lagrange
    space and the matching vector Lagrange space for the velocity.
    """
    mesh = phi.function_space.mesh
    if interface_cut_mesh.mesh is None:
        raise ValueError("interface_cut_mesh must contain a non-empty mesh")
    if interface_speed.function_space.mesh is not interface_cut_mesh.mesh:
        raise ValueError("interface_speed must live on interface_cut_mesh.mesh")
    if k_ring < 0:
        raise ValueError("k_ring must be non-negative")
    if target_space is not None:
        if target_space.mesh is not mesh:
            raise ValueError("target_space must live on the same mesh as phi")
        if target_space.value_size != 1:
            raise ValueError("target_space must be scalar")

    carrier_space = functionspace(mesh, ("Lagrange", 1))
    carrier_vector_space = _vector_lagrange_space(mesh, 1)
    speed = Function(carrier_space, name="extension_speed")
    velocity = Function(carrier_vector_space, name="extension_velocity")
    signed_distance = Function(carrier_space, name="extension_signed_distance")

    _cpp.distance.extend_normal_velocity(
        phi._cpp_object,
        interface_speed._cpp_object,
        interface_cut_mesh._cpp_object,
        speed._cpp_object,
        velocity._cpp_object,
        signed_distance._cpp_object,
        k_ring,
        normal_sign,
        max_iter,
        tol,
    )

    if target_space is None:
        return NormalExtensionResult(speed, velocity, signed_distance)

    degree = _element_degree(target_space)
    target_vector_space = _vector_lagrange_space(mesh, degree)
    target_speed = Function(target_space, name=speed.name)
    target_velocity = Function(target_vector_space, name=velocity.name)
    target_signed_distance = Function(target_space, name=signed_distance.name)
    target_speed.interpolate(speed)
    target_velocity.interpolate(velocity)
    target_signed_distance.interpolate(signed_distance)
    return NormalExtensionResult(
        target_speed,
        target_velocity,
        target_signed_distance,
    )


def adapt_mesh_to_stl(
    mesh: Mesh,
    stl_path: str,
    *,
    nlevels: int = 1,
    k_ring: int = 1,
    aabb_padding: float = 0.0,
    ghost_mode: GhostMode = GhostMode.shared_facet,
    option: RefinementOption = RefinementOption.parent_cell,
) -> Mesh:
    """Refine a mesh around cells intersected by an STL surface."""
    partitioner = create_cell_partitioner(ghost_mode, 2)

    for _ in range(nlevels):
        edges = _cpp.distance.refinement_edges_from_stl(
            mesh.comm, mesh._cpp_object, stl_path, aabb_padding, k_ring
        )
        if edges.size == 0:
            break
        mesh, _, _ = refine(mesh, edges=edges, partitioner=partitioner, option=option)

    return mesh


compute_distance_from_stl = from_stl
distribute_stl_facets = distribute_stl
