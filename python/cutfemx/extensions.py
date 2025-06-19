# Copyright (c) 2022-2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""This module contains extensions functionality for CutFEMx."""

from cutfemx.cutfemx_cpp import extensions

__all__ = [
    "build_root_to_cut_mapping",
    "get_badly_cut_parent_indices",
]


def get_badly_cut_parent_indices(cut_cells, mesh_size_h, threshold_factor=0.1):
    """
    Identify badly cut cells based on volume threshold and return their parent indices.

    Parameters
    ----------
    cut_cells : CutCells
        CutCells object containing all cut cells (must be triangulated)
    mesh_size_h : float
        Characteristic mesh size
    threshold_factor : float, optional
        Factor to multiply with mesh size (default 0.1 for Ch threshold)

    Returns
    -------
    list[int]
        Vector of parent cell indices for badly cut cells

    Notes
    -----
    The CutCells must be triangulated for accurate volume computation.
    Cells with volume < threshold_factor * mesh_size_h are considered badly cut.
    """
    return extensions.get_badly_cut_parent_indices(cut_cells, mesh_size_h, threshold_factor)


def build_root_to_cut_mapping(badly_cut_parents, interior_cells, mesh, cut_cells):
    """
    Build mapping from badly cut parent cells to interior cells.

    This function implements the root-to-cut cell mapping using a breadth-first search
    to find nearby interior cells for each badly cut cell following the Burman-Hansbo approach.

    Parameters
    ----------
    badly_cut_parents : list[int]
        Vector of parent indices for badly cut cells
    interior_cells : list[int]
        Vector of interior cell indices
    mesh : dolfinx.mesh.Mesh
        DOLFINx mesh object
    cut_cells : CutCells
        CutCells object containing cut cell information

    Returns
    -------
    list[int]
        Flat vector containing mapping [badly_cut_0, interior_0, badly_cut_1, interior_1, ...]
        Each pair represents a badly cut cell and its associated interior cell.
    """
    # Determine the floating point type from mesh and get the underlying C++ object
    if hasattr(mesh, "geometry") and hasattr(mesh.geometry, "x"):
        mesh_cpp = mesh._cpp_object  # Get the underlying C++ mesh object
        if mesh.geometry.x.dtype.name == "float32":
            return extensions.build_root_to_cut_mapping_float32(
                badly_cut_parents, interior_cells, mesh_cpp, cut_cells
            )
        else:
            return extensions.build_root_to_cut_mapping_float64(
                badly_cut_parents, interior_cells, mesh_cpp, cut_cells
            )
    else:
        # Default to float64
        mesh_cpp = mesh._cpp_object  # Get the underlying C++ mesh object
        return extensions.build_root_to_cut_mapping_float64(
            badly_cut_parents, interior_cells, mesh_cpp, cut_cells
        )
