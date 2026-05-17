# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Main module for CutFEMx"""
import sys

from cutfemx import cutfemx_cpp as _cpp
from cutfemx.mesh_cutter import MeshCutter, mesh_cutter

if hasattr(_cpp, "level_set"):
    from cutfemx import level_set
else:
    level_set = None

if hasattr(_cpp, "quadrature"):
    from cutfemx import quadrature
else:
    quadrature = None

if hasattr(_cpp, "mesh"):
    from cutfemx import mesh
else:
    mesh = None

if hasattr(_cpp, "fem") and hasattr(_cpp.fem, "IntegralType"):
    from cutfemx import fem
elif hasattr(_cpp, "fem") and hasattr(_cpp.fem, "create_cut_function"):
    from cutfemx import fem_migration as fem

    sys.modules[f"{__name__}.fem"] = fem
    cut_function = fem.cut_function
else:
    fem = None


def get_include(user=False):
    import os

    d = os.path.dirname(__file__)
    return os.path.join(d, "include")
