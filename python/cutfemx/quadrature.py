# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing

from cutcells import CutCells_float32, CutCells_float64
from cutfemx import cutfemx_cpp as _cpp

from dolfinx.mesh import Mesh
from dolfinx.fem import Function

__all__ = [
  "runtime_quadrature",
  "physical_points"
]


def runtime_quadrature(
        level_set: typing.Union[Function],
        ls_part: str,
        order: int)->typing.Union[_cpp.quadrature.QuadratureRules_float32, _cpp.quadrature.QuadratureRules_float64]:

      return _cpp.quadrature.runtime_quadrature(level_set._cpp_object,ls_part,order)

def physical_points(
        runtime_rules: typing.Union[_cpp.quadrature.QuadratureRules_float32, _cpp.quadrature.QuadratureRules_float64],
        mesh: Mesh)->typing.Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
      points_phys = _cpp.quadrature.physical_points(runtime_rules, mesh._cpp_object)
      return points_phys

import basix
from dolfinx.fem import CoordinateElement

# Helper function for generating full quadrature (Testing purposes)
def runtime_quadrature_mesh(mesh: Mesh, order: int):
    """
    Generate quadrature rules for the full domain (all cells) for testing purposes.
    Mimics runtime_quadrature structure but covers specific cells (here all).
    """
    dtype = mesh.geometry.x.dtype
    
    # 1. Get Reference Quadrature
    tdim = mesh.topology.dim
    # ct = mesh.topology.cell_type
    
    # Convert dolfinx cell type to basix
    bct = mesh.basix_cell()
    
    points_ref, weights_ref = basix.make_quadrature(bct, order, basix.QuadratureType.default, basix.PolysetType.standard)
    points_ref = np.array(points_ref, dtype=mesh.geometry.x.dtype)
    weights_ref = np.array(weights_ref, dtype=mesh.geometry.x.dtype)
    
    num_points = len(weights_ref)
    num_cells = mesh.topology.index_map(tdim).size_local # Local cells
    
    # 2. Compute Jacobians to map weights
    gdim = mesh.geometry.dim
    cmap = mesh.geometry.cmap
    
    # Create basix element to tabulate derivatives
    degree = cmap.degree
    variant = basix.LagrangeVariant(cmap.variant)
    el = basix.create_element(basix.ElementFamily.P, bct, degree, variant)
    
    # Evaluate coordinate basis derivatives at quadrature points
    # shape: (derivative, num_points, num_dofs, value_size)
    phi = el.tabulate(1, points_ref)
    dphi = phi[1:] # (tdim, num_points, num_dofs, 1) usually
    
    x_dofs = mesh.geometry.dofmap
    x_coords = mesh.geometry.x
    
    # Pre-compute basis derivative at quadrature points in Ref Element
    # phi_0: Basis values (pts, dofs, 1)
    phi_0 = phi[0] 
    
    rules_list = []
    
    for c in range(num_cells):
        try:
             cell_dofs = x_dofs.links(c)
        except AttributeError:
             cell_dofs = x_dofs[c]
             
        cell_coords = x_coords[cell_dofs] # (num_dofs, 3)
        
        # Calculate J at all points: (num_points, gdim, tdim)
        dphi_eval = dphi[:, :, :, 0].transpose(1, 2, 0) # (pts, dofs, tdim)
        
        # J[p, g, t] = sum_k coords[k, g] * dphi[p, k, t]
        J = np.einsum('kd, pkt -> pdt', cell_coords[:, :gdim], dphi_eval) 
        
        # Determinant
        if gdim == tdim:
            detJ = np.abs(np.linalg.det(J))
        elif gdim == 3 and tdim == 2:
            JTJ = np.einsum('pdi, pdj -> pij', J, J)
            detJ = np.sqrt(np.linalg.det(JTJ))
        else:
            raise NotImplementedError(f"Unsupported dimensions gdim={gdim}, tdim={tdim}")
            
        w_phys = weights_ref * detJ
        
        # Points Phys -> Ref
        # x_phys_vals = ... (Incorrect, we need Reference Points for Kernel)
        # Use points_ref directly
        
        # Ensure contiguous and correct dtype
        x_ref_data = np.ascontiguousarray(points_ref, dtype=dtype)
        w_phys_data = np.ascontiguousarray(w_phys, dtype=dtype)
        
        # Create Single Rule
        if dtype == np.float32:
             rule = _cpp.quadrature.QuadratureRule_float32(x_ref_data, w_phys_data)
        else:
             rule = _cpp.quadrature.QuadratureRule_float64(x_ref_data, w_phys_data)
        
        rules_list.append(rule)
        
    parent_map = np.arange(num_cells, dtype=np.int64)
    
    # Create Container
    if mesh.geometry.x.dtype == np.float32:
        qr = _cpp.quadrature.QuadratureRules_float32(rules_list, parent_map)
    else:
        qr = _cpp.quadrature.QuadratureRules_float64(rules_list, parent_map)
        
    return qr

