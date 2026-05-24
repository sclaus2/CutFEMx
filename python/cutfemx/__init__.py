# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Main module for CutFEMx"""

from cutfemx import cutfemx_cpp as _cpp
from cutfemx import distance
from cutfemx import fem
from cutfemx.cut import (
    CutData,
    CutMesh,
    create_cut_mesh,
    cut,
    locate_entities,
    runtime_quadrature,
    update,
)

cut_function = fem.cut_function
level_set = None
mesh = None
quadrature = None


def get_include(user=False):
    import os

    d = os.path.dirname(__file__)
    return os.path.join(d, "include")
