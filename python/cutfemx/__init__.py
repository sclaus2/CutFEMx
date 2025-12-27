# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Main module for CutFEMx"""
from cutfemx import level_set, quadrature, mesh, fem

def get_include(user=False):
    import os

    d = os.path.dirname(__file__)
    return os.path.join(d, "include")

