# Copyright (c) 2024-2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from collections.abc import Sequence
import functools
from typing import Any

from petsc4py import PETSc

from dolfinx.fem import petsc as _dolfinx_petsc
from dolfinx.fem.bcs import DirichletBC

from cutfemx import cutfemx_cpp as _cpp
from cutfemx.fem import ActiveDomain, CutForm

__all__ = [
    "assemble_matrix",
    "assemble_vector",
    "create_matrix",
    "create_vector",
    "deactivate_outside",
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


def create_vector(V: Any, kind: str | None = None) -> PETSc.Vec:
    """Create a PETSc vector for a CutFEMx/DOLFINx space or space list."""
    return _dolfinx_petsc.create_vector(V, kind)


def create_matrix(a: Any, kind: str | None = None) -> PETSc.Mat:
    """Create a PETSc matrix for a CutFEMx or DOLFINx bilinear form."""
    if isinstance(a, CutForm):
        return _runtime_petsc_module().create_matrix_float64(a._cpp_object, kind)
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
) -> PETSc.Mat:
    """Assemble a bilinear CutFEMx or DOLFINx form into a PETSc matrix."""
    if isinstance(a, CutForm):
        A = create_matrix(a, kind)
        assemble_matrix(A, a, bcs, diagonal, constants, coeffs)
        return A
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
) -> PETSc.Mat:
    """Assemble a bilinear CutFEMx or DOLFINx form into an existing PETSc matrix."""
    if isinstance(a, CutForm):
        _unsupported_packed_data(constants, coeffs)
        cpp_bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
        petsc_mod = _runtime_petsc_module()
        petsc_mod.assemble_matrix_float64(A, a._cpp_object, cpp_bcs)
        if a.function_spaces[0] is a.function_spaces[1]:
            petsc_mod.insert_diagonal_float64(
                A,
                a._cpp_object.function_spaces[0],
                cpp_bcs,
                PETSc.ScalarType(diagonal),
            )
        return A
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
