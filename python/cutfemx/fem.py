# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import basix.ufl
import ufl
from cutfemx import cutfemx_cpp as _cpp
from cutfemx.cut import CutMesh
from dolfinx import default_scalar_type, fem, la
from dolfinx.fem import Function

__all__ = [
    "ActiveDomain",
    "CutForm",
    "active_domain",
    "apply_lifting",
    "assemble_matrix",
    "assemble_scalar",
    "assemble_vector",
    "create_matrix",
    "create_sparsity_pattern",
    "cut_form",
    "cut_function",
    "deactivate_outside",
    "deactivate_outside_blocks",
    "form",
    "zero_block_rows",
    "zero_rows",
]


@dataclass
class CutForm:
    """Compiled runintgen form held by CutFEMx during the migration."""

    _cpp_object: Any
    ufl_form: ufl.Form
    compiled: Any
    mesh: Any
    function_spaces: list[Any]
    runtime_providers: dict[tuple[str, int], Any]
    custom_data_owners: list[Any]

    @property
    def integral_summary(self) -> list[tuple[str, int, str, bool]]:
        """Return ``(type, id, kernel_mode, needs_custom_data)`` per integral."""
        return [
            (
                info.integral_type,
                int(info.subdomain_id),
                info.kernel.mode,
                bool(info.needs_custom_data),
            )
            for info in self.compiled.jit_info.integral_infos
        ]

    @property
    def needs_runtime_data(self) -> bool:
        """Whether any generated integral needs runintgen custom data."""
        return any(item[3] for item in self.integral_summary)

    @property
    def dtype(self) -> np.dtype:
        """Scalar dtype of the migration runtime form."""
        return np.dtype(np.float64)

    @property
    def rank(self) -> int:
        """Rank of the migration runtime form."""
        return self._cpp_object.rank


class ActiveDomain:
    """Form-derived active-domain support used for deactivation."""

    def __init__(self, cpp_object: Any, function_space: Any):
        self._cpp_object = cpp_object
        self._function_space = function_space
        self._indicator: Function | None = None

    @property
    def indicator(self) -> Function:
        if self._indicator is None:
            indicator = _wrap_cpp_function(
                self._cpp_object.indicator, self._function_space, "active_domain"
            )
            indicator._cutfemx_active_domain_owner = self
            self._indicator = indicator
        return self._indicator

    @property
    def active_cells(self) -> np.ndarray:
        return self._cpp_object.active_cells

    @property
    def inactive_dofs(self) -> np.ndarray:
        return self._cpp_object.inactive_dofs


def _normalised_subdomain_id(value: Any) -> int:
    return -1 if value in ("otherwise", "everywhere") else int(value)


def _runtime_providers(jit_info: Any) -> dict[tuple[str, int], Any]:
    providers: dict[tuple[str, int], Any] = {}
    for group in jit_info.analysis.groups:
        if group.quadrature_providers:
            for sid, provider in group.quadrature_providers.items():
                providers[(group.integral_type, _normalised_subdomain_id(sid))] = (
                    provider
                )
        else:
            for sid in group.subdomain_ids:
                providers[(group.integral_type, _normalised_subdomain_id(sid))] = (
                    group.quadrature_provider
                )
    return providers


def _create_cpp_form(
    *,
    compiled: Any,
    mesh: Any,
    function_spaces: list[Any],
    entity_maps: Sequence[Any] | None = None,
    custom_data: Any | None = None,
) -> tuple[Any, list[Any]]:
    """Create the C++ CutFEMx form and return it with runtime-data owners."""
    if not hasattr(_cpp, "fem") or not hasattr(_cpp.fem, "create_form_float64"):
        raise RuntimeError("CutFEMx was built without runtime Form support")

    from cutfemx._runintgen_adapter import (
        _active_coefficients,
        _as_cpp_object,
        _build_subdomains,
        _require_dolfinx,
        _resolve_custom_data,
        _runtime_domain_and_custom_data,
        _standard_domain,
        _ufl_to_dolfinx_integral_type,
    )

    if entity_maps is not None:
        raise NotImplementedError(
            "CutFEMx migration runtime forms do not yet support entity maps."
        )

    integral_type_names = (
        "cell",
        "exterior_facet",
        "interior_facet",
        "vertex",
        "ridge",
    )
    _require_dolfinx()
    subdomains = _build_subdomains(compiled.ufl_form, compiled.ufcx_form)

    original_coefficients = compiled.ufl_form.coefficients()
    coefficients = []
    for i in range(int(compiled.ufcx_form.num_coefficients)):
        original_index = int(compiled.ufcx_form.original_coefficient_positions[i])
        coefficients.append(_as_cpp_object(original_coefficients[original_index]))

    constants = [_as_cpp_object(c) for c in compiled.ufl_form.constants()]

    offsets = [
        int(compiled.ufcx_form.form_integral_offsets[i])
        for i in range(len(integral_type_names) + 1)
    ]
    info_by_position = {
        info.integral_position: info for info in compiled.jit_info.integral_infos
    }

    integrals: dict[
        tuple[int, int, int],
        tuple[int, np.ndarray, list[int], int | None],
    ] = {}
    owners: list[Any] = []
    owner_cache: dict[tuple[int, str, int], Any] = {}
    ffi_obj = compiled.module.ffi

    for type_index, integral_type_name in enumerate(integral_type_names):
        for j in range(offsets[type_index], offsets[type_index + 1]):
            integral_index = j - offsets[type_index]
            info = info_by_position[j]
            integral = compiled.ufcx_form.form_integrals[j]
            kernel_fn = integral.tabulate_tensor_float64
            if kernel_fn == ffi_obj.NULL:
                raise RuntimeError("Generated float64 UFCx kernel is NULL.")
            kernel_ptr = int(ffi_obj.cast("uintptr_t", kernel_fn))
            subdomain_id = int(compiled.ufcx_form.form_integral_ids[j])
            active_coeffs = [
                int(i)
                for i in _active_coefficients(
                    ffi_obj,
                    compiled.ufcx_form,
                    integral,
                )
            ]

            if info.needs_custom_data:
                entities, data_ptr, owner = _runtime_domain_and_custom_data(
                    compiled=compiled,
                    mesh=mesh,
                    integral_type=info.integral_type,
                    subdomain_id=subdomain_id,
                    kernel_idx=0,
                    custom_data=custom_data,
                    cache=owner_cache,
                )
                if owner is not None:
                    owners.append(owner)
            else:
                dolfinx_integral_type = _ufl_to_dolfinx_integral_type(
                    integral_type_name
                )
                entities = _standard_domain(
                    mesh=mesh,
                    integral_type=dolfinx_integral_type,
                    subdomain_id=subdomain_id,
                    subdomains=subdomains,
                )
                data_ptr = _resolve_custom_data(
                    custom_data,
                    dolfinx_integral_type,
                    subdomain_id,
                    0,
                )

            integrals[(type_index, integral_index, 0)] = (
                kernel_ptr,
                np.ascontiguousarray(entities, dtype=np.int32).ravel(),
                active_coeffs,
                None if data_ptr is None else int(data_ptr),
            )

    needs_facet_permutations = any(
        bool(compiled.ufcx_form.form_integrals[j].needs_facet_permutations)
        for j in range(offsets[0], offsets[-1])
    )

    cpp_form = _cpp.fem.create_form_float64(
        [_as_cpp_object(space) for space in function_spaces],
        integrals,
        _as_cpp_object(mesh),
        coefficients,
        constants,
        needs_facet_permutations,
    )
    return cpp_form, owners


def _compile_cut_form(
    ufl_form: ufl.Form,
    *,
    dtype: Any | None = None,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
    jit_comm: Any | None = None,
    entity_maps: Sequence[Any] | None = None,
    custom_data: Any | None = None,
) -> CutForm:
    from cutfemx._runintgen_adapter import (
        _reject_standard_quadrature_functions,
        compile_form,
    )

    _domains = list(ufl_form.subdomain_data().keys())
    if len(_domains) != 1:
        raise RuntimeError("CutFEMx migration form expects exactly one UFL domain.")
    domain = _domains[0]
    msh = domain.ufl_cargo()
    if msh is None:
        raise RuntimeError("Expecting to find a DOLFINx Mesh in the form.")

    dtype = default_scalar_type if dtype is None else dtype
    if np.dtype(dtype) != np.dtype(np.float64):
        raise NotImplementedError("CutFEMx migration forms currently support float64.")

    options = dict(form_compiler_options or {})
    options["scalar_type"] = dtype
    _reject_standard_quadrature_functions(ufl_form)
    compiled = compile_form(
        msh.comm if jit_comm is None else jit_comm,
        ufl_form,
        form_compiler_options=options,
        jit_options=jit_options,
    )
    spaces = [arg.ufl_function_space() for arg in ufl_form.arguments()]
    if options.get("part", "full") == "diagonal" and spaces:
        spaces = [spaces[0]]
    cpp_form, owners = _create_cpp_form(
        compiled=compiled,
        mesh=msh,
        function_spaces=spaces,
        entity_maps=entity_maps,
        custom_data=custom_data,
    )
    return CutForm(
        _cpp_object=cpp_form,
        ufl_form=ufl_form,
        compiled=compiled,
        mesh=msh,
        function_spaces=spaces,
        runtime_providers=_runtime_providers(compiled.jit_info),
        custom_data_owners=owners,
    )


def form(
    form_object: Any,
    dtype: Any | None = None,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
    jit_comm: Any | None = None,
    entity_maps: Sequence[Any] | None = None,
    custom_data: Any | None = None,
) -> Any:
    """Compile a UFL form for the CutFEMx migration pipeline."""

    if isinstance(form_object, ufl.ZeroBaseForm):
        return fem.form(
            form_object,
            dtype=default_scalar_type if dtype is None else dtype,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            jit_comm=jit_comm,
            entity_maps=entity_maps,
        )

    if isinstance(form_object, ufl.Form):
        if entity_maps is not None:
            raise NotImplementedError(
                "CutFEMx migration forms do not yet support entity maps."
            )
        return _compile_cut_form(
            form_object,
            dtype=dtype,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            jit_comm=jit_comm,
            entity_maps=entity_maps,
            custom_data=custom_data,
        )

    if isinstance(form_object, Sequence) and not isinstance(form_object, (str, bytes)):
        return [
            form(
                item,
                dtype=dtype,
                form_compiler_options=form_compiler_options,
                jit_options=jit_options,
                jit_comm=jit_comm,
                entity_maps=entity_maps,
                custom_data=custom_data,
            )
            for item in form_object
        ]

    return form_object


cut_form = form


def _same_family_element_for_mesh(u: Function, cut_mesh: CutMesh):
    element = u.function_space.ufl_element()
    if element.is_mixed or element.is_quadrature or element.is_real:
        raise ValueError("cut_function currently supports point-evaluation elements")

    if cut_mesh.mesh is None:
        raise ValueError("Cannot create a cut function on an empty cut mesh")

    return basix.ufl.element(
        element.element_family,
        cut_mesh.mesh.topology.cell_name(),
        element.degree,
        lagrange_variant=element.lagrange_variant,
        dpc_variant=element.dpc_variant,
        discontinuous=element.discontinuous,
        shape=element.reference_value_shape,
        dtype=element.dtype,
    )


def cut_function(u: Function, cut_mesh: CutMesh) -> Function:
    """Create a function on a cut mesh using the C++ parent-map path."""
    if not hasattr(_cpp, "fem") or not hasattr(_cpp.fem, "create_cut_function"):
        raise RuntimeError("CutFEMx was built without the cut_function component")

    cpp_func = _cpp.fem.create_cut_function(u._cpp_object, cut_mesh._cpp_object)
    element = _same_family_element_for_mesh(u, cut_mesh)
    V_cut = fem.FunctionSpace(cut_mesh.mesh, element, cppV=cpp_func.function_space)

    out = Function(V_cut, dtype=u.x.array.dtype, name=u.name)
    out._cpp_object = cpp_func
    out._x = la.Vector(out._cpp_object.x)
    return out


def assemble_scalar(M: Any) -> Any:
    """Assemble a scalar form through CutFEMx when it is a runtime form."""
    if isinstance(M, CutForm):
        return _cpp.fem.assemble_scalar_float64(M._cpp_object)
    return fem.assemble_scalar(M)


def _unsupported_packed_data(constants: Any | None, coeffs: Any | None) -> None:
    if constants is not None or coeffs is not None:
        raise NotImplementedError(
            "CutFEMx migration assembly currently packs runtime form data in C++."
        )


def _cpp_bcs_by_block(bcs: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [[bc._cpp_object for bc in block] for block in bcs]


def _lifting_x0_arrays(x0: Sequence[Any] | None) -> list[np.ndarray] | None:
    if x0 is None:
        return None

    arrays = []
    for values in x0:
        if values is None:
            arrays.append(np.empty(0, dtype=np.float64))
        elif hasattr(values, "array"):
            arrays.append(np.asarray(values.array))
        elif hasattr(values, "x") and hasattr(values.x, "array"):
            arrays.append(np.asarray(values.x.array))
        else:
            arrays.append(np.asarray(values))
    return arrays


def apply_lifting(
    b: np.ndarray,
    a: Sequence[CutForm | None],
    bcs: Sequence[Sequence[Any]],
    x0: Sequence[Any] | None = None,
    alpha: float = 1.0,
) -> None:
    """Apply Dirichlet lifting for CutFEMx runtime bilinear forms."""
    if not all(form is None or isinstance(form, CutForm) for form in a):
        return fem.apply_lifting(b, a, bcs, x0=x0, alpha=alpha)

    cpp_forms = [None if form is None else form._cpp_object for form in a]
    _cpp.fem.apply_lifting_float64(
        b,
        cpp_forms,
        _cpp_bcs_by_block(bcs),
        _lifting_x0_arrays(x0),
        np.float64(alpha),
    )


def _wrap_cpp_function(cpp_func: Any, V: Any, name: str) -> Function:
    out = Function(V, dtype=cpp_func.x.array.dtype, name=name)
    out._cpp_object = cpp_func
    out._x = la.Vector(out._cpp_object.x)
    return out


def _same_function_space(V0: Any, V1: Any) -> bool:
    if V0 is V1:
        return True
    try:
        return bool(V0 == V1)
    except Exception:
        return False


def active_domain(a: CutForm) -> ActiveDomain:
    """Build the active-domain support object for a bilinear CutFEMx form."""
    if not isinstance(a, CutForm):
        raise TypeError("active_domain expects a CutFEMx CutForm")
    if not hasattr(_cpp.fem, "active_domain_float64"):
        raise RuntimeError("CutFEMx was built without active-domain support.")
    return ActiveDomain(
        _cpp.fem.active_domain_float64(a._cpp_object), a.function_spaces[0]
    )


def deactivate_outside(
    A: la.MatrixCSR,
    b_or_active_domain: la.Vector | ActiveDomain,
    active_domain_or_none: ActiveDomain | None = None,
    *,
    diagonal: float = 1.0,
    rhs_value: float = 0.0,
) -> ActiveDomain:
    """Deactivate matrix rows outside a form-derived active domain."""
    if isinstance(b_or_active_domain, ActiveDomain):
        if active_domain_or_none is not None:
            raise TypeError("deactivate_outside(A, active_domain) takes no RHS vector")
        domain = b_or_active_domain
        _cpp.fem.deactivate_outside_float64(
            A._cpp_object, domain._cpp_object, np.float64(diagonal)
        )
        return domain

    if active_domain_or_none is None:
        raise TypeError("deactivate_outside(A, b, active_domain) requires active_domain")
    b = b_or_active_domain
    domain = active_domain_or_none
    _cpp.fem.deactivate_outside_matrix_vector_float64(
        A._cpp_object,
        b._cpp_object,
        domain._cpp_object,
        np.float64(diagonal),
        np.float64(rhs_value),
    )
    return domain


def _cpp_matrix_block_rows(A_blocks: Sequence[Sequence[Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in A_blocks:
        rows.append([None if A is None else A._cpp_object for A in row])
    return rows


def deactivate_outside_blocks(
    A_blocks: Sequence[Sequence[la.MatrixCSR | None]],
    active_domains: Sequence[ActiveDomain],
    b_blocks: Sequence[la.Vector] | None = None,
    *,
    diagonal: float = 1.0,
    rhs_value: float = 0.0,
) -> list[ActiveDomain]:
    """Deactivate block rows from per-row active-domain support.

    The inactive rows are taken directly from ``active_domains[i]``. Only the
    matching diagonal block ``A_blocks[i][i]`` and optional RHS block
    ``b_blocks[i]`` are modified.
    """
    domains = list(active_domains)
    cpp_domains = [domain._cpp_object for domain in domains]
    cpp_A_blocks = _cpp_matrix_block_rows(A_blocks)

    if b_blocks is None:
        _cpp.fem.deactivate_outside_blocks_float64(
            cpp_A_blocks, cpp_domains, np.float64(diagonal)
        )
        return domains

    _cpp.fem.deactivate_outside_blocks_matrix_vector_float64(
        cpp_A_blocks,
        [b._cpp_object for b in b_blocks],
        cpp_domains,
        np.float64(diagonal),
        np.float64(rhs_value),
    )
    return domains


def zero_rows(A: la.MatrixCSR, *, tol: float = 0.0) -> np.ndarray:
    """Return owned scalar rows whose assembled MatrixCSR entries are zero."""
    return np.asarray(
        _cpp.fem.zero_rows_float64(A._cpp_object, float(tol)), dtype=np.int32
    )


def zero_block_rows(
    A_blocks: Sequence[Sequence[la.MatrixCSR | None]], *, tol: float = 0.0
) -> list[np.ndarray]:
    """Return zero rows for each row of a MatrixCSR block system."""
    return [
        np.asarray(rows, dtype=np.int32)
        for rows in _cpp.fem.zero_block_rows_float64(
            _cpp_matrix_block_rows(A_blocks), float(tol)
        )
    ]


def create_sparsity_pattern(a: Any) -> Any:
    """Create a sparsity pattern for a runtime CutFEMx form."""
    if isinstance(a, CutForm):
        return _cpp.fem.create_sparsity_pattern_float64(a._cpp_object)
    return fem.create_sparsity_pattern(a)


def create_matrix(a: Any, block_mode: la.BlockMode | None = None) -> la.MatrixCSR:
    """Create a MatrixCSR compatible with a runtime CutFEMx form."""
    if not isinstance(a, CutForm):
        return fem.create_matrix(a, block_mode)

    if block_mode is None:
        return la.MatrixCSR(_cpp.fem.create_matrix_float64(a._cpp_object))

    sp = create_sparsity_pattern(a)
    sp.finalize()
    return la.matrix_csr(sp, block_mode=block_mode, dtype=a.dtype)


@functools.singledispatch
def assemble_vector(
    L: Any,
    constants: Any | None = None,
    coeffs: Any | None = None,
) -> la.Vector:
    """Assemble a linear runtime form into a new vector."""
    if isinstance(L, CutForm):
        _unsupported_packed_data(constants, coeffs)
        b = fem.create_vector(L.function_spaces[0], L.dtype)
        b.array[:] = 0
        _cpp.fem.assemble_vector_float64(b.array, L._cpp_object)
        return b
    return fem.assemble_vector(L, constants, coeffs)


@assemble_vector.register(np.ndarray)
def _assemble_vector_array(
    b: np.ndarray,
    L: Any,
    constants: Any | None = None,
    coeffs: Any | None = None,
) -> np.ndarray:
    """Assemble a linear runtime form into an existing array."""
    if isinstance(L, CutForm):
        _unsupported_packed_data(constants, coeffs)
        _cpp.fem.assemble_vector_float64(b, L._cpp_object)
        return b
    return fem.assemble_vector(b, L, constants, coeffs)


@functools.singledispatch
def assemble_matrix(
    a: Any,
    bcs: Sequence[Any] | None = None,
    diag: float = 1.0,
    constants: Any | None = None,
    coeffs: Any | None = None,
    block_mode: la.BlockMode | None = None,
) -> la.MatrixCSR:
    """Assemble a bilinear runtime form into a new MatrixCSR."""
    if isinstance(a, CutForm):
        A = create_matrix(a, block_mode)
        _assemble_matrix_csr(A, a, bcs, diag, constants, coeffs)
        return A
    return fem.assemble_matrix(a, bcs, diag, constants, coeffs, block_mode)


@assemble_matrix.register
def _assemble_matrix_csr(
    A: la.MatrixCSR,
    a: Any,
    bcs: Sequence[Any] | None = None,
    diag: float = 1.0,
    constants: Any | None = None,
    coeffs: Any | None = None,
) -> la.MatrixCSR:
    """Assemble a bilinear runtime form into an existing MatrixCSR."""
    if not isinstance(a, CutForm):
        return fem.assemble_matrix(A, a, bcs, diag, constants, coeffs)

    _unsupported_packed_data(constants, coeffs)
    cpp_bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    _cpp.fem.assemble_matrix_float64(A._cpp_object, a._cpp_object, cpp_bcs)

    if _same_function_space(a.function_spaces[0], a.function_spaces[1]):
        _cpp.fem.insert_diagonal_float64(
            A._cpp_object,
            a._cpp_object.function_spaces[0],
            cpp_bcs,
            np.float64(diag),
        )
    return A
