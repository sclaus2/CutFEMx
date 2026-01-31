# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing

from cutfemx import cutfemx_cpp as _cpp

from cutcells import CutCells_float32, CutCells_float64

from dolfinx.fem import Function
from dolfinx.mesh import Mesh

__all__ = [
  "locate_entities",
  "locate_entities_part",
  "cut_entities",
  "ghost_penalty_facets",
  "facet_topology",
  "interior_facets",
  "compute_normal",
  "distribute_stl_facets",
  "build_cell_triangle_map",
  "compute_unsigned_distance",
  "compute_signed_distance",
  "SignMode",
  "compute_distance_from_stl",
  "compute_stl_bbox",
  "reinitialize"
]

SignMode = _cpp.level_set.SignMode

def compute_stl_bbox(path: str):
    """
    Compute axis-aligned bounding box of an STL file without loading everything.
    Returns:
        (min_corner, max_corner) where each is a numpy array of shape (3,)
    """
    return _cpp.level_set.compute_stl_bbox(path)

# ... (locate_entities etc remain) ...

def compute_signed_distance(
    dist_func: Function,
    soup,
    cell_map,
    sign_mode: SignMode = SignMode.ComponentAnchor,
    max_iter: int = 1000,
    tol: float = 1e-10
):
    """
    Compute signed distance field using FMM and specified sign method.
    """
    _cpp.level_set.compute_signed_distance(dist_func._cpp_object, soup, cell_map, sign_mode, max_iter, tol)

def compute_distance_from_stl(
    mesh: Mesh,
    stl_path: str,
    sign_mode: SignMode = SignMode.ComponentAnchor,
    max_iter: int = 1000,
    tol: float = 1e-10,
    aabb_padding: float = 0.05
):
    """
    High-level helper to compute distance from STL file.
    Performs distribution, map building, and FMM computation in one go.
    
    Args:
        mesh: Background mesh
        stl_path: Path to the STL file
        sign_mode: Method to determine sign (default: ComponentAnchor)
        max_iter: Maximum number of FMM iterations
        tol: Tolerance for convergence
        aabb_padding: Padding for AABB intersection tests
        
    Returns:
        dist_func: The computed signed distance function (Lagrange P1)
    """
    import time
    from mpi4py import MPI
    rank = mesh.comm.Get_rank()
    
    t_start = time.time()
    from dolfinx.fem import functionspace
    V = functionspace(mesh, ("Lagrange", 1))
    dist_func = Function(V)
    t_space = time.time()
    
    soup = distribute_stl_facets(mesh, stl_path, aabb_padding)
    t_dist = time.time()
    
    cell_map = build_cell_triangle_map(mesh, soup, aabb_padding)
    t_map = time.time()
    
    # Use signed distance computation
    compute_signed_distance(dist_func, soup, cell_map, sign_mode, max_iter, tol)
    t_compute = time.time()
    
    if rank == 0:
        print(f"  [Timing] FunctionSpace: {t_space - t_start:.4f}s")
        print(f"  [Timing] Distribute:    {t_dist - t_space:.4f}s")
        print(f"  [Timing] BuildMap:      {t_map - t_dist:.4f}s")
        print(f"  [Timing] FMM+Sign:      {t_compute - t_map:.4f}s")
    
    return dist_func

def locate_entities(
    level_set: Function,
    dim: int,
    ls_part: str,
    include_ghosts: bool = False,
) -> npt.NDArray[np.int32]:
  return _cpp.level_set.locate_entities(level_set._cpp_object,dim,ls_part,include_ghosts)

def locate_entities_part(
    level_set: Function,
    entities: npt.NDArray[np.int32],
    dim: int,
    ls_part: str
) -> npt.NDArray[np.int32]:
  return _cpp.level_set.locate_entities_part(level_set._cpp_object,entities,dim,ls_part)

def cut_entities(
    level_set: Function,
    dof_coordinates: typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
    entities: npt.NDArray[np.int32],
    tdim: int,
    cut_type: str
) -> typing.Union[CutCells_float32, CutCells_float64]:
      return _cpp.level_set.cut_entities(level_set._cpp_object,dof_coordinates,entities,tdim,cut_type)

def ghost_penalty_facets(
    level_set: Function,
    cut_type: str
) -> npt.NDArray[np.int32]:
      return _cpp.level_set.ghost_penalty_facets(level_set._cpp_object,cut_type)

def facet_topology(
    mesh: Mesh,
    facet_ids: npt.NDArray[np.int32]
) -> npt.NDArray[np.int32]:
      return _cpp.level_set.facet_topology(mesh._cpp_object,facet_ids)

def compute_normal(normal: Function,level_set: Function,entities: npt.NDArray[np.int32]):
    _cpp.level_set.compute_normal(normal._cpp_object,level_set._cpp_object,entities)
    return normal

def interior_facets(
    level_set: Function,
    cells: npt.NDArray[np.int32],
    cut_type: str
) -> npt.NDArray[np.int32]:
      return _cpp.level_set.interior_facets(level_set._cpp_object,cells,cut_type)


def distribute_stl_facets(
    mesh: Mesh,
    stl_path: str,
    aabb_padding: float = 0.05
):
    """
    Read and distribute STL file facets across MPI ranks based on mesh partitioning.
    
    Args:
        mesh: The background mesh
        stl_path: Path to the STL file
        aabb_padding: Padding for AABB intersection tests
        
    Returns:
        TriSoup object containing the distributed surface mesh
    """
    return _cpp.level_set.distribute_stl_facets(mesh.comm, mesh._cpp_object, stl_path, aabb_padding)

def build_cell_triangle_map(
    mesh: Mesh,
    soup,
    aabb_padding: float = 0.0
):
    """
    Build a map from local mesh cells to intersecting surface triangles.
    
    Args:
        mesh: The background mesh
        soup: TriSoup object from distribute_stl_facets
        aabb_padding: Padding for AABB intersection tests
        
    Returns:
        CellTriangleMap object
    """
    return _cpp.level_set.build_cell_triangle_map(mesh._cpp_object, soup, aabb_padding)

def compute_unsigned_distance(
    dist_func: Function,
    soup,
    cell_map,
    max_iter: int = 1000,
    tol: float = 1e-10
):
    """
    Compute unsigned distance field from STL surface using Fast Marching Method.
    
    Args:
        dist_func: Function to store the computed distance
        soup: TriSoup object
        cell_map: CellTriangleMap object
        max_iter: Maximum number of FMM iterations
        tol: Tolerance for convergence
    """
    _cpp.level_set.compute_unsigned_distance(dist_func._cpp_object, soup, cell_map, max_iter, tol)



def reinitialize(
    phi: Function,
    max_iter: int = 200,
    tol: float = 1e-10
):
    """
    Reinitialize a level set function to a signed distance function using FMM.
    
    Args:
        phi: The level set function (modified in-place)
        max_iter: Maximum FMM iterations
        tol: Convergence tolerance
    """
    _cpp.level_set.reinitialize(phi._cpp_object, max_iter, tol)
