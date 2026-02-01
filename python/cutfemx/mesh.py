# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing
import ufl
import basix

from mpi4py import MPI as _MPI

from cutfemx import cutfemx_cpp as _cpp
from dolfinx import cpp as _dcpp

from dolfinx.mesh import Mesh, refine, create_cell_partitioner, GhostMode, RefinementOption
from cutcells import CutCells_float32, CutCells_float64

__all__ = [
  "create_cut_cells_mesh",
  "create_cut_mesh",
  "adapt_mesh_to_stl"
]

class CutMesh:
    """A mesh."""

    # _bg_mesh: typing.Union[_dcpp.mesh.Mesh_float32, _dcpp.mesh.Mesh_float64]
    # _cut_mesh: typing.Union[_cpp.mesh.CutMesh_float32, _cpp.mesh.CutMesh_float64]

    def __init__(self, cut_msh):
      self._cpp_object = cut_msh
      cell_type = cut_msh.cut_mesh.topology.cell_type
      gdim = cut_msh.cut_mesh.geometry.dim
      dtype=cut_msh.cut_mesh.geometry.x.dtype
      domain = ufl.Mesh(basix.ufl.element("Lagrange", cell_type.name, 1, shape=(gdim,), dtype=dtype))
      #dolfinx python mesh object of cutmesh used for writing mesh to file
      self._mesh = Mesh(cut_msh.cut_mesh,domain)

def create_cut_cells_mesh(
        comm: _MPI.Comm,
        cut_cells: typing.Union[CutCells_float32, CutCells_float64]
        )->typing.Union[_cpp.mesh.CutMesh_float32,_cpp.mesh.CutMesh_float64]:
        cut_msh = _cpp.mesh.create_cut_cells_mesh(comm,cut_cells)
        return CutMesh(cut_msh)

def create_cut_mesh(
        comm: _MPI.Comm,
        cut_cells: typing.Union[CutCells_float32, CutCells_float64],
        mesh: Mesh,
        entities: npt.NDArray[np.int32]
        )->typing.Union[_cpp.mesh.CutMesh_float32,_cpp.mesh.CutMesh_float64]:
        return CutMesh(_cpp.mesh.create_cut_mesh(comm,cut_cells,mesh._cpp_object,entities))

def adapt_mesh_to_stl(mesh: Mesh,
                      stl_path: str,
                      *,
                      nlevels: int = 1,
                      k_ring: int = 1,
                      aabb_padding: float = 0.0,
                      ghost_mode: GhostMode = GhostMode.shared_facet,
                      option: RefinementOption = RefinementOption.parent_cell) -> Mesh:
    """
    Adaptively refine the mesh around an STL surface using geometry-driven indicators.

    Args:
        mesh: The background mesh to refine.
        stl_path: Path to the STL file.
        nlevels: Number of refinement iterations.
        k_ring: Number of rings of neighbors to mark around cut cells.
        aabb_padding: Padding for initial STL search.
        ghost_mode: Ghost mode for the refined mesh partitioner.
        option: Refinement option (e.g., preserve parent cell).
    
    Returns:
        Refined dolfinx mesh.
    """
    comm = mesh.comm
    partitioner = create_cell_partitioner(ghost_mode)

    for i in range(nlevels):
        # Obtain refinement edges from C++ helper
        edges = _cpp.mesh.refinement_edges_from_stl(comm, mesh._cpp_object, stl_path, aabb_padding, k_ring)
        
        if edges.size == 0:
            break
            
        # Refine
        mesh, parent_cell, parent_facet = refine(mesh, edges=edges, partitioner=partitioner, option=option)
        
    return mesh