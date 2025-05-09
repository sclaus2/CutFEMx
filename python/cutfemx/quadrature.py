# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing

from cutcells import CutCells_float32, CutCells_float64
from cutfemx import cutfemx_cpp as _cpp

from dolfinx.mesh import Mesh
from dolfinx.fem import Function

__all__ = [
  "runtime_quadrature",
  "physical_points"
]

def runtime_quadrature(
        level_set: typing.Union[Function],
        ls_part: str,
        order: int)->typing.Union[_cpp.quadrature.QuadratureRules_float32, _cpp.quadrature.QuadratureRules_float64]:

      return _cpp.quadrature.runtime_quadrature(level_set._cpp_object,ls_part,order)

def physical_points(
        runtime_rules: typing.Union[_cpp.quadrature.QuadratureRules_float32, _cpp.quadrature.QuadratureRules_float64],
        mesh: Mesh)->typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
      points_phys = _cpp.quadrature.physical_points(runtime_rules, mesh._cpp_object)
      return points_phys

