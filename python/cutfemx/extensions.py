# Copyright (c) 2022-2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""This module contains extensions functionality for CutFEMx."""

from cutfemx.cutfemx_cpp import extensions

__all__ = [
    "apply_constraints",
    "build_dof_to_cells_mapping",
    "build_dof_to_root_mapping",
    "build_root_mapping",
    "create_root_to_cell_map",
    "evaluate_basis_at_dof_coordinates",
    "extend_function",
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

    # Determine the floating point type from function space
    if hasattr(function_space, "mesh") and hasattr(function_space.mesh, "geometry"):
        if function_space.mesh.geometry.x.dtype.name == "float32":
            return extensions.build_dof_to_cells_mapping_float32(
                cut_cell_parents, root, function_space_cpp
            )
        else:
            return extensions.build_dof_to_cells_mapping_float64(
                cut_cell_parents, root, function_space_cpp
            )
    else:
        # Default to float64
        return extensions.build_dof_to_cells_mapping_float64(
            cut_cell_parents, root, function_space_cpp
        )


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


def evaluate_basis_at_dof_coordinates(dof_to_root_mapping, function_space, mesh):
    """
    Evaluate basis functions to create DOF constraint relationships.

    For each cut DOF, this function maps its physical coordinate to the reference space
    of its associated root cell, evaluates all basis functions at that point, and
    returns the constraint relationships as (root_dof, weight) pairs.

    Parameters
    ----------
    dof_to_root_mapping : dict
        Dictionary mapping cut DOF indices to root cell indices
    function_space : dolfinx.fem.FunctionSpace
        The function space whose basis functions to evaluate
    mesh : dolfinx.mesh.Mesh
        The mesh containing the root cells

    Returns
    -------
    dict[int, list[tuple[int, float]]]
        Dictionary mapping cut DOF indices to lists of (root_dof_index, weight) pairs.
        Each entry represents: cut_dof -> [(root_dof_1, weight_1), (root_dof_2, weight_2), ...]

    Notes
    -----
    This function creates discrete extension constraints where each cut DOF is expressed
    as a linear combination of root DOFs weighted by basis function values.
    Only non-zero weights (> 1e-12) are included in the constraint pairs.
    """
    function_space_cpp = function_space._cpp_object
    mesh_cpp = mesh._cpp_object

    # Determine the floating point type from mesh
    if hasattr(mesh, "geometry") and hasattr(mesh.geometry, "x"):
        if mesh.geometry.x.dtype.name == "float32":
            return extensions.evaluate_basis_at_dof_coordinates_float32(
                dof_to_root_mapping, function_space_cpp, mesh_cpp
            )
        else:
            return extensions.evaluate_basis_at_dof_coordinates_float64(
                dof_to_root_mapping, function_space_cpp, mesh_cpp
            )
    else:
        # Default to float64
        return extensions.evaluate_basis_at_dof_coordinates_float64(
            dof_to_root_mapping, function_space_cpp, mesh_cpp
        )


def apply_constraints(matrix, constraints, function_space):
    """
    Apply constraints to a matrix: A ↦ C^T A C.

    Transforms the matrix to eliminate cut DOFs while preserving the underlying physics.
    The constraint matrix C expresses cut DOFs as linear combinations of interior DOFs.

    Parameters
    ----------
    matrix : dolfinx.la.MatrixCSR
        The matrix to be transformed (modified in-place)
    constraints : dict[int, list[tuple[int, float]]]
        Dictionary mapping cut DOF indices to lists of (root_dof_index, weight) pairs
    function_space : dolfinx.fem.FunctionSpace
        The function space for DOF indexing

    Notes
    -----
    This function implements the mathematical transformation A ↦ C^T A C where:
    - A is the original matrix
    - C is the constraint matrix expressing cut DOFs as linear combinations of interior DOFs
    - The result eliminates cut DOFs from the linear system

    The matrix is modified in-place to avoid memory overhead.
    """
    function_space_cpp = function_space._cpp_object

    # Determine the floating point type from matrix
    if hasattr(matrix, "array") and hasattr(matrix.array(), "dtype"):
        if matrix.array().dtype.name == "float32":
            return extensions.apply_constraints_matrixcsr_float32(
                matrix, constraints, function_space_cpp
            )
        else:
            return extensions.apply_constraints_matrixcsr_float64(
                matrix, constraints, function_space_cpp
            )
    else:
        # Default to float64
        return extensions.apply_constraints_matrixcsr_float64(
            matrix, constraints, function_space_cpp
        )


def extend_function(u_interior, constraints, function_space):
    """
    Extend solution vector from interior DOFs to full domain: u_full = C u_interior.

    Reconstructs the full field by applying the constraint relationships.
    Cut DOF values are computed as linear combinations of interior DOF values.

    Parameters
    ----------
    u_interior : numpy.ndarray
        Solution vector defined on interior DOFs only
    constraints : dict[int, list[tuple[int, float]]]
        Dictionary mapping cut DOF indices to lists of (root_dof_index, weight) pairs
    function_space : dolfinx.fem.FunctionSpace
        The function space for DOF indexing

    Returns
    -------
    numpy.ndarray
        Extended solution vector including both interior and cut DOFs

    Notes
    -----
    This function implements: u_cut = Σ weight_i * u_interior[root_dof_i] for each cut DOF.
    The resulting vector contains values for all DOFs (interior + cut).
    """
    import numpy as np

    function_space_cpp = function_space._cpp_object

    # Determine the floating point type from input array
    if u_interior.dtype.name == "float32":
        result = extensions.extend_function_float32(u_interior, constraints, function_space_cpp)
    else:
        result = extensions.extend_function_float64(u_interior, constraints, function_space_cpp)

    return np.array(result, dtype=u_interior.dtype)
