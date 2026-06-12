# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Main module for CutFEMx"""

__version__ = "0.2.0"

from cutfemx import cutfemx_cpp as cutfemx_cpp
from cutfemx import distance as distance
from cutfemx import extensions as extensions
from cutfemx import fem as fem
from cutfemx import level_set as level_set
from cutfemx._runintgen_adapter import (
    QuadratureFunction as QuadratureFunction,
)
from cutfemx.cut import (
    CutData as CutData,
)
from cutfemx.cut import (
    CutMesh as CutMesh,
)
from cutfemx.cut import (
    create_cut_mesh as create_cut_mesh,
)
from cutfemx.cut import (
    cut as cut,
)
from cutfemx.cut import (
    ghost_penalty_facets as ghost_penalty_facets,
)
from cutfemx.cut import (
    interior_facets_for_cells as interior_facets_for_cells,
)
from cutfemx.cut import (
    locate_entities as locate_entities,
)
from cutfemx.cut import (
    runtime_quadrature as runtime_quadrature,
)
from cutfemx.cut import (
    runtime_quadratures as runtime_quadratures,
)
from cutfemx.cut import (
    update as update,
)
from cutfemx.level_set import (
    conormal as conormal,
)
from cutfemx.level_set import (
    correction_distance as correction_distance,
)
from cutfemx.level_set import (
    level_set_value as level_set_value,
)
from cutfemx.level_set import (
    NormalEvaluator as NormalEvaluator,
)
from cutfemx.level_set import (
    normal as normal,
)
from cutfemx.level_set import (
    surface_normal as surface_normal,
)

cut_function = fem.cut_function
mesh = None
quadrature = None


def get_include(user=False):
    import os

    d = os.path.dirname(__file__)
    return os.path.join(d, "include")
