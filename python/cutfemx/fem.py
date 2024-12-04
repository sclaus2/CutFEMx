# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing

from cutfemx import cutfemx_cpp as _cpp

from cutfemx.mesh import CutMesh

from dolfinx.mesh import Mesh
from dolfinx.fem import Function

__all__ = [
  "cut_function",
]

def cut_function(
        u: Function,
        sub_mesh: CutMesh)->Function:

      return _cpp.fem.create_cut_function(u._cpp_object,sub_mesh._cpp_object)


