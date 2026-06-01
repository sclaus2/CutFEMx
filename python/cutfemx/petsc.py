# Copyright (c) 2024-2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from collections.abc import Sequence
import functools
import numbers
from typing import Any

from petsc4py import PETSc

from dolfinx.fem import petsc as _dolfinx_petsc
from dolfinx.fem.bcs import DirichletBC

from cutfemx import cutfemx_cpp as _cpp
from cutfemx import extensions as _extensions
from cutfemx.fem import ActiveDomain, CutForm

__all__ = [
    "assemble_matrix",
    "assemble_vector",
    "create_matrix",
    "create_vector",
    "deactivate_outside",
    "deactivate_outside_blocks",
    "zero_block_rows",
    "zero_rows",
]


def _runtime_petsc_module() -> Any:
    if not hasattr(_cpp, "fem") or not hasattr(_cpp.fem, "petsc"):
        raise RuntimeError("CutFEMx was built without runtime PETSc support.")
    return _cpp.fem.petsc


def _unsupported_packed_data(constants: Any | None, coeffs: Any | None) -> None:
    if constants is not None or coeffs is not None:
        raise NotImplementedError(
            "CutFEMx migration PETSc assembly currently packs runtime form data in C++."
        )


def _normalise_extension_terms(extension_terms: Sequence[Any] | None) -> list[Any]:
    if extension_terms is None:
        return []

    terms = []
    for item in extension_terms:
        if isinstance(item, _extensions.ExtensionPenaltyTerm):
            term = item
        elif isinstance(item, tuple) and len(item) == 3:
            term, cut_data, aggregation = item
            if not isinstance(term, _extensions.ExtensionPenaltyTerm):
                raise TypeError(
                    "Extension term tuples must be "
                    "(ExtensionPenaltyTerm, CutData, CellAggregation)"
                )
            term = term.with_domain(cut_data, aggregation)
        else:
            raise TypeError(
                "extension_terms entries must be ExtensionPenaltyTerm objects "
                "bound to cut_data/aggregation, or "
                "(ExtensionPenaltyTerm, CutData, CellAggregation) tuples"
            )

        if term.cut_data is None or term.aggregation is None:
            raise ValueError(
                "PETSc extension terms must be bound to cut_data and aggregation"
            )
        terms.append(term)
    return terms


def _assemble_extension_terms(A: PETSc.Mat, terms: Sequence[Any]) -> None:
    petsc_mod = _runtime_petsc_module()
    for term in terms:
        type_name = _extensions._geometry_type_name(term.V)
        if isinstance(term.beta, numbers.Real):
            getattr(petsc_mod, f"assemble_extension_penalty_scalar_{type_name}")(
                A,
                term.V._cpp_object,
                term.cut_data._cpp_object,
                term.aggregation._cpp_object,
                PETSc.ScalarType(term.beta),
                term.quadrature_degree,
            )
        else:
            values = _extensions._cellwise_values(
                term.beta, term.V.mesh, term.V.mesh.geometry.x.dtype
            )
            getattr(petsc_mod, f"assemble_extension_penalty_cellwise_{type_name}")(
                A,
                term.V._cpp_object,
                term.cut_data._cpp_object,
                term.aggregation._cpp_object,
                values,
                term.quadrature_degree,
            )


def create_vector(V: Any, kind: str | None = None) -> PETSc.Vec:
    """Create a PETSc vector for a CutFEMx/DOLFINx space or space list."""
    return _dolfinx_petsc.create_vector(V, kind)


def create_matrix(
    a: Any,
    kind: str | None = None,
    extension_terms: Sequence[Any] | None = None,
) -> PETSc.Mat:
    """Create a PETSc matrix for a CutFEMx or DOLFINx bilinear form."""
    if isinstance(a, CutForm):
        terms = _normalise_extension_terms(extension_terms)
        if not terms:
            return _runtime_petsc_module().create_matrix_float64(a._cpp_object, kind)
        return _runtime_petsc_module().create_matrix_with_extension_sparsity_float64(
            a._cpp_object,
            [term.V._cpp_object for term in terms],
            [term.aggregation._cpp_object for term in terms],
            kind,
        )
    if extension_terms is not None:
        raise NotImplementedError(
            "Extension terms are currently supported for CutFEMx runtime forms only."
        )
    return _dolfinx_petsc.create_matrix(a, kind)


@functools.singledispatch
def assemble_vector(
    L: Any,
    constants: Any | None = None,
    coeffs: Any | None = None,
    kind: str | None = None,
) -> PETSc.Vec:
    """Assemble a linear CutFEMx or DOLFINx form into a PETSc vector."""
    if isinstance(L, CutForm):
        _unsupported_packed_data(constants, coeffs)
        if kind not in (None, PETSc.Vec.Type.MPI):
            raise NotImplementedError(
                "CutFEMx migration PETSc assembly currently creates MPI vectors."
            )
        b = create_vector(L.function_spaces[0], kind)
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, L)
        return b
    return _dolfinx_petsc.assemble_vector(L, constants, coeffs, kind)


@assemble_vector.register
def _(
    b: PETSc.Vec,
    L: Any,
    constants: Any | None = None,
    coeffs: Any | None = None,
) -> PETSc.Vec:
    """Assemble a linear CutFEMx or DOLFINx form into an existing PETSc vector."""
    if isinstance(L, CutForm):
        _unsupported_packed_data(constants, coeffs)
        _runtime_petsc_module().assemble_vector_float64(b, L._cpp_object)
        return b
    return _dolfinx_petsc.assemble_vector(b, L, constants, coeffs)


@functools.singledispatch
def assemble_matrix(
    a: Any,
    bcs: Sequence[DirichletBC] | None = None,
    diagonal: float = 1.0,
    constants: Any | None = None,
    coeffs: Any | None = None,
    kind: str | None = None,
    extension_terms: Sequence[Any] | None = None,
) -> PETSc.Mat:
    """Assemble a bilinear CutFEMx or DOLFINx form into a PETSc matrix."""
    if isinstance(a, CutForm):
        terms = _normalise_extension_terms(extension_terms)
        A = create_matrix(a, kind, terms)
        assemble_matrix(A, a, bcs, diagonal, constants, coeffs, terms)
        return A
    if extension_terms is not None:
        raise NotImplementedError(
            "Extension terms are currently supported for CutFEMx runtime forms only."
        )
    return _dolfinx_petsc.assemble_matrix(
        a, bcs, diagonal, constants, coeffs, kind
    )


@assemble_matrix.register
def _(
    A: PETSc.Mat,
    a: Any,
    bcs: Sequence[DirichletBC] | None = None,
    diagonal: float = 1.0,
    constants: Any | None = None,
    coeffs: Any | None = None,
    extension_terms: Sequence[Any] | None = None,
) -> PETSc.Mat:
    """Assemble a bilinear CutFEMx or DOLFINx form into an existing PETSc matrix."""
    if isinstance(a, CutForm):
        _unsupported_packed_data(constants, coeffs)
        terms = _normalise_extension_terms(extension_terms)
        cpp_bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
        petsc_mod = _runtime_petsc_module()
        petsc_mod.assemble_matrix_float64(A, a._cpp_object, cpp_bcs)
        _assemble_extension_terms(A, terms)
        if a.function_spaces[0] is a.function_spaces[1]:
            petsc_mod.insert_diagonal_float64(
                A,
                a._cpp_object.function_spaces[0],
                cpp_bcs,
                PETSc.ScalarType(diagonal),
            )
        return A
    if extension_terms is not None:
        raise NotImplementedError(
            "Extension terms are currently supported for CutFEMx runtime forms only."
        )
    return _dolfinx_petsc.assemble_matrix(A, a, bcs, diagonal, constants, coeffs)


def deactivate_outside(
    A: PETSc.Mat,
    b_or_active_domain: Any,
    active_domain_or_none: ActiveDomain | None = None,
    diagonal: float = 1.0,
    rhs_value: float = 0.0,
) -> ActiveDomain:
    """Deactivate PETSc matrix rows outside a form-derived active domain."""
    petsc_mod = _runtime_petsc_module()
    if isinstance(b_or_active_domain, ActiveDomain):
        if active_domain_or_none is not None:
            raise TypeError("deactivate_outside(A, active_domain) takes no RHS vector")
        domain = b_or_active_domain
        petsc_mod.deactivate_outside_float64(
            A,
            domain._cpp_object,
            PETSc.ScalarType(diagonal),
        )
        return domain

    if active_domain_or_none is None:
        raise TypeError("deactivate_outside(A, b, active_domain) requires active_domain")
    b = b_or_active_domain
    domain = active_domain_or_none
    petsc_mod.deactivate_outside_matrix_vector_float64(
        A, b, domain._cpp_object, PETSc.ScalarType(diagonal), PETSc.ScalarType(rhs_value)
    )
    return domain


def _petsc_matrix_block_rows(A_blocks: Any) -> list[list[PETSc.Mat | None]]:
    if isinstance(A_blocks, PETSc.Mat):
        try:
            rows, cols = A_blocks.getNestSize()
        except Exception as exc:
            raise TypeError(
                "deactivate_outside_blocks expects a nested PETSc Mat or "
                "a nested sequence of PETSc Mat blocks"
            ) from exc
        return [
            [A_blocks.getNestSubMatrix(i, j) for j in range(cols)]
            for i in range(rows)
        ]
    return [list(row) for row in A_blocks]


def deactivate_outside_blocks(
    A_blocks: Any,
    active_domains: Sequence[ActiveDomain],
    b_blocks: Sequence[PETSc.Vec] | None = None,
    diagonal: float = 1.0,
    rhs_value: float = 0.0,
) -> list[ActiveDomain]:
    """Deactivate PETSc block rows from per-row active-domain support."""
    petsc_mod = _runtime_petsc_module()
    domains = list(active_domains)
    cpp_domains = [domain._cpp_object for domain in domains]
    mat_blocks = _petsc_matrix_block_rows(A_blocks)

    if b_blocks is None:
        petsc_mod.deactivate_outside_blocks_float64(
            mat_blocks,
            cpp_domains,
            PETSc.ScalarType(diagonal),
        )
        return domains

    petsc_mod.deactivate_outside_blocks_matrix_vector_float64(
        mat_blocks,
        list(b_blocks),
        cpp_domains,
        PETSc.ScalarType(diagonal),
        PETSc.ScalarType(rhs_value),
    )
    return domains


def zero_rows(A: PETSc.Mat, tol: float = 0.0) -> list[int]:
    """Return owned local PETSc rows whose assembled entries are zero."""
    return list(_runtime_petsc_module().zero_rows_float64(A, float(tol)))


def zero_block_rows(A_blocks: Any, tol: float = 0.0) -> list[list[int]]:
    """Return zero rows for each row of a PETSc block system."""
    return [
        list(rows)
        for rows in _runtime_petsc_module().zero_block_rows_float64(
            _petsc_matrix_block_rows(A_blocks), float(tol)
        )
    ]
