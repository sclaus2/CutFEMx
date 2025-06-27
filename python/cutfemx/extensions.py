# Copyright (c) 2022-2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""This module contains extensions functionality for CutFEMx."""

from cutfemx.cutfemx_cpp import extensions

__all__ = [
    "build_dof_to_cells_mapping",
    "build_dof_to_root_mapping",
    "build_root_mapping",
    "create_root_to_cell_map",
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


def build_root_mapping(badly_cut_parents, interior_cells, mesh, cut_cells):
    """
    Build mapping from cut cell parents to interior cells using propagation algorithm.

    This function implements a propagation algorithm for mapping ALL cut cells to interior cells:

    1. Start with all cut_cells.parent_map cells (extracted automatically)
    2. Find cut cells that directly share facets with interior cells - these get direct mapping
    3. Iteratively propagate: find unmapped cut cells connected to mapped cut cells via facets
    4. Continue until all cut cells are mapped or max iterations reached

    This approach ensures that all cut cells in connected regions are grouped together with
    the same interior cell for extension operations.

    Parameters
    ----------
    badly_cut_parents : list[int]
        Vector of parent indices for badly cut cells (parameter kept for compatibility,
        but algorithm now processes all cut cells from cut_cells.parent_map)
    interior_cells : list[int]
        Vector of interior cell indices
    mesh : dolfinx.mesh.Mesh
        DOLFINx mesh object
    cut_cells : CutCells
        CutCells object containing cut cell information (used to extract parent_map)

    Returns
    -------
    dict[int, int]
        Dictionary mapping cut cell parent indices to interior cell indices.
        Each cut cell is mapped to exactly one interior cell that serves
        as the extension source. All cut cells from cut_cells.parent_map are processed.
    """
    # Extract the parent map from cut_cells
    cut_cell_parents = cut_cells.parent_map

    # Determine the floating point type from mesh and get the underlying C++ object
    if hasattr(mesh, "geometry") and hasattr(mesh.geometry, "x"):
        mesh_cpp = mesh._cpp_object  # Get the underlying C++ mesh object
        if mesh.geometry.x.dtype.name == "float32":
            return extensions.build_root_mapping_float32(
                badly_cut_parents, interior_cells, mesh_cpp, cut_cell_parents
            )
        else:
            return extensions.build_root_mapping_float64(
                badly_cut_parents, interior_cells, mesh_cpp, cut_cell_parents
            )
    else:
        # Default to float64
        mesh_cpp = mesh._cpp_object  # Get the underlying C++ mesh object
        return extensions.build_root_mapping_float64(
            badly_cut_parents, interior_cells, mesh_cpp, cut_cell_parents
        )


def create_root_to_cell_map(cell_to_root_mapping):
    """
    Create inverse mapping from interior (root) cells to vectors of cut cells.

    This function takes a mapping from cut cells to interior cells and creates
    the inverse mapping where each interior cell maps to a list of all cut cells
    that are mapped to it.

    Parameters
    ----------
    cell_to_root_mapping : dict[int, int]
        Mapping from cut cell indices to interior cell indices

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping interior cell indices to lists of cut cell indices
        that map to them
    """
    return extensions.create_root_to_cell_map(cell_to_root_mapping)


def build_dof_to_cells_mapping(cut_cells, root, function_space):
    """
    Build mapping from DOF indices to vectors of cell indices containing that DOF.

    This function iterates over all cut cell parents and their corresponding root cells
    to collect which cells contain each DOF that are specific to cut cells (not shared
    with the interior/root cells).

    Parameters
    ----------
    cut_cells : CutCells
        CutCells object containing all cut cells (used to extract parent_map)
    root : dict[int, int]
        Mapping from cut cell parents to root cells (from build_root_mapping)
    function_space : dolfinx.fem.FunctionSpace
        Function space for which to build the dof mapping

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping DOF indices to lists of cut cell indices that contain each DOF.
        Only includes DOFs that are on the cut boundary (not shared with root cells).

    Notes
    -----
    The function processes cut cell parents and filters out DOFs that are shared with the
    corresponding root cells. This gives a mapping of boundary/cut-specific DOFs to
    the cut cells that contain them.
    """
    # Extract the parent map from cut_cells
    cut_cell_parents = cut_cells.parent_map

    # Extract the underlying C++ FunctionSpace object
    function_space_cpp = function_space._cpp_object
    return extensions.build_dof_to_cells_mapping(cut_cell_parents, root, function_space_cpp)


def build_dof_to_root_mapping(dof_to_cells_mapping, cut_cell_to_root_mapping, function_space, mesh):
    """
    Build mapping from DOF indices to closest root cell indices.

    For each DOF, this function finds all possible root cells (via the cut cells that
    contain the DOF), computes the distance from the DOF coordinate to each root cell
    midpoint, and selects the closest one.

    Parameters
    ----------
    dof_to_cells_mapping : dict[int, list[int]]
        Mapping from DOF indices to lists of cut cell indices (from build_dof_to_cells_mapping)
    cut_cell_to_root_mapping : dict[int, int]
        Mapping from cut cell parents to root cells (from build_root_mapping)
    function_space : dolfinx.fem.FunctionSpace
        Function space for DOF coordinate information
    mesh : dolfinx.mesh.Mesh
        Mesh object for computing cell midpoints

    Returns
    -------
    dict[int, int]
        Dictionary mapping DOF indices to closest root cell indices.
        Each DOF is mapped to exactly one root cell - the one whose midpoint
        is closest to the DOF coordinate.

    Notes
    -----
    This function uses geometric distance to select the most appropriate root cell
    for each DOF, which is useful for extension operations where the spatial
    relationship between DOFs and root cells matters.
    """
    # Extract the underlying C++ objects
    function_space_cpp = function_space._cpp_object
    mesh_cpp = mesh._cpp_object

    # Get DOF coordinates
    dof_coords = function_space.tabulate_dof_coordinates()

    # Determine the floating point type from mesh
    if hasattr(mesh, "geometry") and hasattr(mesh.geometry, "x"):
        if mesh.geometry.x.dtype.name == "float32":
            return extensions.build_dof_to_root_mapping_float32(
                dof_to_cells_mapping,
                cut_cell_to_root_mapping,
                dof_coords.flatten(),
                function_space_cpp,
                mesh_cpp,
            )
        else:
            return extensions.build_dof_to_root_mapping_float64(
                dof_to_cells_mapping,
                cut_cell_to_root_mapping,
                dof_coords.flatten(),
                function_space_cpp,
                mesh_cpp,
            )
    else:
        # Default to float64
        return extensions.build_dof_to_root_mapping_float64(
            dof_to_cells_mapping,
            cut_cell_to_root_mapping,
            dof_coords.flatten(),
            function_space_cpp,
            mesh_cpp,
        )
