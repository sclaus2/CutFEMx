# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt

from cutfemx import cutfemx_cpp as _cpp

from dolfinx.fem import Function

__all__ = [
  "locate_entities",
  "locate_entities_part"
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
