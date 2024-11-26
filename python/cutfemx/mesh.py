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

from dolfinx.mesh import Mesh
from cutcells import CutCells_float32, CutCells_float64

__all__ = [
  "create_cut_cells_mesh",
  "create_cut_mesh"
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
        num_local_cells: int,
        cut_cells: typing.Union[CutCells_float32, CutCells_float64]
        )->typing.Union[_cpp.mesh.CutMesh_float32,_cpp.mesh.CutMesh_float64]:
        cut_msh = _cpp.mesh.create_cut_cells_mesh(comm,num_local_cells,cut_cells)
        return CutMesh(cut_msh)

def create_cut_mesh(
        comm: _MPI.Comm,
        cut_cells: typing.Union[CutCells_float32, CutCells_float64],
        mesh: Mesh,
        entities: npt.NDArray[np.int32]
        )->typing.Union[_cpp.mesh.CutMesh_float32,_cpp.mesh.CutMesh_float64]:
        return CutMesh(_cpp.mesh.create_cut_mesh(comm,cut_cells,mesh._cpp_object,entities))