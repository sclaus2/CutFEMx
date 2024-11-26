# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Main module for CutFEMx"""
from cutfemx import level_set, quadrature, mesh

def get_include(user=False):
    import os

    d = os.path.dirname(__file__)
    if os.path.exists(os.path.join(d, "wrappers")):
        # Package is installed
        return os.path.join(d, "wrappers")
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "src")

