# Copyright (c) 2022 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Main module for CutFEMx"""

# Import version information
from cutfemx._version import (
    __version__,
    __author__,
    __email__,
    __license__,
    __copyright__,
    __url__,
    __description__,
    version_info,
)

from cutfemx import level_set, quadrature, mesh, fem, extensions


def get_include(user=False):
    import os

    d = os.path.dirname(__file__)
    if os.path.exists(os.path.join(d, "wrappers")):
        # Package is installed
        return os.path.join(d, "wrappers")
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "src")


def get_version():
    """Return the version string."""
    return __version__


def get_version_info():
    """Return detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "copyright": __copyright__,
        "url": __url__,
        "description": __description__,
        "version_info": version_info,
    }
