# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Any

import numpy as np
import numpy.typing as npt

from cutfemx import cutfemx_cpp as _cpp
from cutfemx.cut import CutData


@dataclass
class CellAggregation:
    """Python handle for CutFEMx cell aggregation data."""

    _cpp_object: Any

    @property
    def active_cells(self):
        return self._cpp_object.active_cells

    @property
    def cut_cells(self):
        return self._cpp_object.cut_cells

    @property
    def interior_cells(self):
        return self._cpp_object.interior_cells

    @property
    def well_posed_cells(self):
        return self._cpp_object.well_posed_cells

    @property
    def ill_posed_cells(self):
        return self._cpp_object.ill_posed_cells

    @property
    def root_cell(self):
        return self._cpp_object.root_cell

    @property
    def aggregate_id(self):
        return self._cpp_object.aggregate_id

    @property
    def propagation_depth(self):
        return self._cpp_object.propagation_depth

    @property
    def rootless_cells(self):
        return self._cpp_object.rootless_cells

    @property
    def cut_volume_fraction(self):
        return self._cpp_object.cut_volume_fraction


@dataclass
class ExtensionQuadrature:
    """Full-cell bad/root quadrature for aggregation extension terms."""

    _cpp_object: Any

    @property
    def bad_cells(self):
        return self._cpp_object.bad_cells

    @property
    def root_cells(self):
        return self._cpp_object.root_cells

    @property
    def offsets(self):
        return self._cpp_object.offsets

    @property
    def bad_reference_points(self):
        return self._cpp_object.bad_reference_points

    @property
    def root_reference_points(self):
        return self._cpp_object.root_reference_points

    @property
    def weights(self):
        return self._cpp_object.weights

    @property
    def tdim(self):
        return self._cpp_object.tdim


@dataclass
class ExtensionPenaltyTerm:
    """One PDE-dependent aggregation extension penalty contribution."""

    V: Any
    beta: Any
    quadrature_degree: int
    product: str = "L2"
    cut_data: Any | None = None
    aggregation: Any | None = None

    def __post_init__(self):
        if self.product != "L2":
            raise NotImplementedError("Only L2 extension penalty terms are implemented in v1")

    def with_domain(
        self, cut_data: CutData, aggregation: CellAggregation
    ) -> "ExtensionPenaltyTerm":
        """Return a copy bound to the cut data and aggregation used for assembly."""
        return ExtensionPenaltyTerm(
            self.V,
            self.beta,
            self.quadrature_degree,
            self.product,
            cut_data,
            aggregation,
        )


def create_cell_aggregation(
    cut_data: CutData,
    selector: str,
    volume_fraction_threshold: float,
    *,
    root_policy: str = "interior_or_well_cut",
    max_iterations: int = -1,
    allow_rootless: bool = False,
) -> CellAggregation:
    """Create serial v1 cell aggregation data from CutFEMx cut data."""
    if not isinstance(cut_data, CutData):
        raise TypeError("create_cell_aggregation expects a cutfemx.CutData object")
    if not hasattr(_cpp, "extensions"):
        raise RuntimeError("CutFEMx was built without extensions support")

    dtype = cut_data.mesh.geometry.x.dtype
    if dtype.name == "float32":
        cpp_object = _cpp.extensions.create_cell_aggregation_float32(
            cut_data._cpp_object,
            selector,
            volume_fraction_threshold,
            root_policy,
            max_iterations,
            allow_rootless,
        )
    else:
        cpp_object = _cpp.extensions.create_cell_aggregation_float64(
            cut_data._cpp_object,
            selector,
            volume_fraction_threshold,
            root_policy,
            max_iterations,
            allow_rootless,
        )
    return CellAggregation(cpp_object)


def _geometry_type_name(V) -> str:
    dtype = V.mesh.geometry.x.dtype
    return "float32" if dtype.name == "float32" else "float64"


def _cellwise_values(beta: Any, mesh, dtype: np.dtype) -> npt.NDArray:
    if isinstance(beta, np.ndarray):
        values = np.asarray(beta, dtype=dtype)
        if values.ndim != 1:
            raise ValueError("Cellwise extension penalty beta array must be one-dimensional")
        return np.ascontiguousarray(values)

    if not (hasattr(beta, "function_space") and hasattr(beta, "x")):
        raise TypeError(
            "Extension penalty beta must be a scalar, numpy cell array, or DG0 Function"
        )

    Vb = beta.function_space
    if Vb.mesh is not mesh:
        raise ValueError("Extension penalty beta Function must live on the same mesh")

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    values = np.empty(num_cells, dtype=dtype)
    for cell in range(num_cells):
        dofs = Vb.dofmap.cell_dofs(cell)
        if len(dofs) != 1:
            raise ValueError(
                "Extension penalty beta Function must have exactly one dof per cell "
                "(DG0 is supported in v1)"
            )
        values[cell] = beta.x.array[dofs[0]]
    return np.ascontiguousarray(values)


def create_extension_penalty_matrix(V, cut_data: CutData, aggregation: CellAggregation):
    """Create a MatrixCSR with sparsity for bad/root extension penalty pairs."""
    if isinstance(V, ExtensionPenaltyTerm):
        V = V.V
    if not isinstance(cut_data, CutData):
        raise TypeError("create_extension_penalty_matrix expects a cutfemx.CutData object")
    if not isinstance(aggregation, CellAggregation):
        raise TypeError("create_extension_penalty_matrix expects a CellAggregation object")
    type_name = _geometry_type_name(V)
    return getattr(
        _cpp.extensions, f"create_extension_penalty_matrix_{type_name}"
    )(V._cpp_object, cut_data._cpp_object, aggregation._cpp_object)


def extension_quadrature(
    V,
    cut_data: CutData,
    aggregation: CellAggregation,
    quadrature_degree: int,
) -> ExtensionQuadrature:
    """Build standard full-cell quadrature on bad cells and pull it to roots."""
    if isinstance(V, ExtensionPenaltyTerm):
        quadrature_degree = V.quadrature_degree
        V = V.V
    if not isinstance(cut_data, CutData):
        raise TypeError("extension_quadrature expects a cutfemx.CutData object")
    if not isinstance(aggregation, CellAggregation):
        raise TypeError("extension_quadrature expects a CellAggregation object")
    type_name = _geometry_type_name(V)
    cpp_object = getattr(_cpp.extensions, f"extension_quadrature_{type_name}")(
        V._cpp_object,
        cut_data._cpp_object,
        aggregation._cpp_object,
        quadrature_degree,
    )
    return ExtensionQuadrature(cpp_object)


def assemble_extension_penalty(
    A,
    V,
    cut_data: CutData,
    aggregation: CellAggregation,
    beta: Any | None = None,
    quadrature_degree: int | None = None,
):
    """Assemble an L2 full-bad-cell extension penalty into ``A``.

    ``beta`` may be a scalar, a one-dimensional cellwise numpy array, or a
    scalar DG0 ``dolfinx.fem.Function``. Function coefficients are evaluated on
    the bad cell. The quadrature domain is the full background bad cell, while
    the same physical quadrature points are also evaluated in root coordinates.
    """
    if not isinstance(cut_data, CutData):
        raise TypeError("assemble_extension_penalty expects a cutfemx.CutData object")
    if not isinstance(aggregation, CellAggregation):
        raise TypeError("assemble_extension_penalty expects a CellAggregation object")
    if isinstance(V, ExtensionPenaltyTerm):
        if beta is not None or quadrature_degree is not None:
            raise ValueError("Do not pass beta/quadrature_degree when using ExtensionPenaltyTerm")
        term = V
        V = term.V
        beta = term.beta
        quadrature_degree = term.quadrature_degree
    if beta is None or quadrature_degree is None:
        raise TypeError("assemble_extension_penalty requires beta and quadrature_degree")

    type_name = _geometry_type_name(V)
    if isinstance(beta, numbers.Real):
        getattr(_cpp.extensions, f"assemble_extension_penalty_scalar_{type_name}")(
            A._cpp_object,
            V._cpp_object,
            cut_data._cpp_object,
            aggregation._cpp_object,
            beta,
            quadrature_degree,
        )
    else:
        values = _cellwise_values(beta, V.mesh, V.mesh.geometry.x.dtype)
        getattr(_cpp.extensions, f"assemble_extension_penalty_cellwise_{type_name}")(
            A._cpp_object,
            V._cpp_object,
            cut_data._cpp_object,
            aggregation._cpp_object,
            values,
            quadrature_degree,
        )
    return A


def extension_penalty_matrix(
    V,
    cut_data: CutData,
    aggregation: CellAggregation,
    beta: Any | None = None,
    quadrature_degree: int | None = None,
):
    """Create and assemble one L2 full-bad-cell extension penalty MatrixCSR."""
    if not isinstance(cut_data, CutData):
        raise TypeError("extension_penalty_matrix expects a cutfemx.CutData object")
    if not isinstance(aggregation, CellAggregation):
        raise TypeError("extension_penalty_matrix expects a CellAggregation object")
    if isinstance(V, ExtensionPenaltyTerm):
        if beta is not None or quadrature_degree is not None:
            raise ValueError("Do not pass beta/quadrature_degree when using ExtensionPenaltyTerm")
        term = V
        V = term.V
        beta = term.beta
        quadrature_degree = term.quadrature_degree
    if beta is None or quadrature_degree is None:
        raise TypeError("extension_penalty_matrix requires beta and quadrature_degree")

    type_name = _geometry_type_name(V)
    if isinstance(beta, numbers.Real):
        return getattr(_cpp.extensions, f"extension_penalty_matrix_scalar_{type_name}")(
            V._cpp_object,
            cut_data._cpp_object,
            aggregation._cpp_object,
            beta,
            quadrature_degree,
        )

    values = _cellwise_values(beta, V.mesh, V.mesh.geometry.x.dtype)
    return getattr(_cpp.extensions, f"extension_penalty_matrix_cellwise_{type_name}")(
        V._cpp_object,
        cut_data._cpp_object,
        aggregation._cpp_object,
        values,
        quadrature_degree,
    )
