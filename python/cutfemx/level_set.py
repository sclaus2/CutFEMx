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

__all__ = [
  "locate_entities",
  "locate_entities_part",
  "cut_entities",
  "ghost_penalty_facets",
  "compute_normal"
]

def locate_entities(
    level_set: Function,
    dim: int,
    ls_part: str
) -> npt.NDArray[np.int32]:
  return _cpp.level_set.locate_entities(level_set._cpp_object,dim,ls_part)

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
) -> typing.Union[CutCells_float32, CutCells_float64]:
      return _cpp.level_set.ghost_penalty_facets(level_set._cpp_object,cut_type)

def compute_normal(normal: Function,level_set: Function,entities: npt.NDArray[np.int32]):
    _cpp.level_set.compute_normal(normal._cpp_object,level_set._cpp_object,entities)
    return normal

