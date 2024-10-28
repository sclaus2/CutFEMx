# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT


def get_include_path():
    """Return path to nanobind wrapper header files"""
    import pathlib

    return pathlib.Path(__file__).parent
