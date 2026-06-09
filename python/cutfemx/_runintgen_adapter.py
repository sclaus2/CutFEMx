"""CutFEMx-owned DOLFINx adapter for runintgen forms.

runintgen stays backend-neutral. This module is the DOLFINx binding layer used
by CutFEMx to compile runintgen kernels, build DOLFINx integration domains, and
construct runintgen custom data from CutFEMx quadrature providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import MethodType
from typing import TYPE_CHECKING, Any

import cffi
import numpy as np
import numpy.typing as npt

import ufl
from cutfemx import cutfemx_cpp as _cpp

if TYPE_CHECKING:
    from runintgen.jit import JITFormInfo
    from runintgen.runtime_data import QuadratureRules, RuntimeQuadraturePayload


fem: Any | None = None
IntegralType: Any | None = None
_DOLFINX_IMPORT_ERROR: BaseException | None = None


@dataclass
class CompiledRunintForm:
    """Compiled runintgen form without associated DOLFINx runtime objects."""

    ufl_form: ufl.Form
    ufcx_form: Any
    module: Any
    code: tuple[str | None, str | None]
    dtype: npt.DTypeLike
    jit_info: JITFormInfo


def _require_dolfinx() -> tuple[Any, Any]:
    """Load DOLFINx lazily and return ``(fem, IntegralType)``."""
    global IntegralType, fem, _DOLFINX_IMPORT_ERROR

    if fem is not None and IntegralType is not None:
        return fem, IntegralType

    try:
        from dolfinx import fem as dolfinx_fem
        from dolfinx.fem import IntegralType as dolfinx_integral_type
    except Exception as exc:  # pragma: no cover - depends on environment
        fem = None
        IntegralType = None
        _DOLFINX_IMPORT_ERROR = exc
        raise ImportError("DOLFINx is required for this function") from exc

    fem = dolfinx_fem
    IntegralType = dolfinx_integral_type
    _DOLFINX_IMPORT_ERROR = None
    return fem, IntegralType


def _is_function_space(value: Any) -> bool:
    return (
        hasattr(value, "ufl_domain")
        and hasattr(value, "ufl_element")
        and hasattr(value, "mesh")
    )


def _as_dolfinx_mesh(value: Any) -> Any:
    if isinstance(value, ufl.Mesh):
        mesh = value.ufl_cargo()
        if mesh is None:
            raise ValueError(
                "cutfemx.QuadratureFunction requires a concrete DOLFINx mesh, "
                "not a cargo-free UFL mesh."
            )
        return mesh
    return value


def _quadrature_function_set_values(
    self: Any,
    quadrature: Any,
    values: npt.ArrayLike,
) -> None:
    rule_id = getattr(quadrature, "rule_id", None)
    if rule_id is None:
        raise TypeError("quadrature must carry a stable rule_id.")
    self._runintgen_values[str(rule_id)] = values
    self._runintgen_cache.pop(str(rule_id), None)


def _quadrature_function_set_evaluator(self: Any, evaluator: Any) -> None:
    self._runintgen_source = evaluator
    self._runintgen_cache.clear()


def _quadrature_function_invalidate(self: Any) -> None:
    self._runintgen_cache.clear()
    source = getattr(self, "_runintgen_source", None)
    invalidate = getattr(source, "invalidate", None)
    if invalidate is not None:
        invalidate()


def _quadrature_function_update(self: Any, *, version: int | None = None) -> None:
    source = getattr(self, "_runintgen_source", None)
    update = getattr(source, "update", None)
    if update is not None:
        update(version=version)
    self._runintgen_cache.clear()


def _quadrature_function_is_cellwise_constant(self: Any) -> bool:
    return False


def QuadratureFunction(
    space_or_mesh: Any,
    source: Any | None = None,
    *,
    name: str | None = None,
    space: Any | None = None,
    shape: tuple[int, ...] = (),
    dtype: npt.DTypeLike | None = None,
) -> Any:
    """Create a DOLFINx Function with runintgen quadrature semantics."""
    dolfinx_fem, _ = _require_dolfinx()
    from runintgen.quadrature_function import QuadratureFunctionSpec

    if space is not None:
        function_space = space
    elif _is_function_space(space_or_mesh):
        function_space = space_or_mesh
    else:
        mesh = _as_dolfinx_mesh(space_or_mesh)
        element = ("DG", 0, shape) if shape else ("DG", 0)
        function_space = dolfinx_fem.functionspace(mesh, element)

    function = dolfinx_fem.Function(function_space, name=name, dtype=dtype)

    value_shape = tuple(shape or function.ufl_shape)
    if shape and value_shape != tuple(function.ufl_shape):
        raise ValueError(
            "The supplied shape does not match the DOLFINx FunctionSpace "
            f"shape {tuple(function.ufl_shape)}."
        )
    value_size = int(prod(value_shape)) if value_shape else 1
    function._runintgen_quadrature_function = QuadratureFunctionSpec(
        name=name,
        value_shape=value_shape,
        value_size=value_size,
    )
    function._runintgen_source = source
    function._runintgen_values = {}
    function._runintgen_cache = {}
    function.set_values = MethodType(_quadrature_function_set_values, function)
    function.set_evaluator = MethodType(_quadrature_function_set_evaluator, function)
    function.invalidate = MethodType(_quadrature_function_invalidate, function)
    function.update = MethodType(_quadrature_function_update, function)
    function.is_cellwise_constant = MethodType(
        _quadrature_function_is_cellwise_constant,
        function,
    )
    return function


def jit(
    comm: Any,
    ufl_object: Any,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
) -> tuple[Any, Any, tuple[str | None, str | None]]:
    """Compile one UFL form with runintgen using DOLFINx JIT options."""
    if not isinstance(ufl_object, ufl.Form):
        raise TypeError(type(ufl_object))

    import runintgen.jit as runintgen_jit

    import ffcx
    from dolfinx import jit as dolfinx_jit

    def _local_jit(
        form_object: ufl.Form,
        *,
        form_compiler_options: dict[str, Any] | None = None,
        jit_options: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, tuple[str | None, str | None]]:
        p_ffcx = ffcx.get_options(form_compiler_options)
        p_jit = dolfinx_jit.get_options(jit_options)
        compiled_forms, module, code = runintgen_jit.compile_forms(
            [form_object],
            options=p_ffcx,
            **p_jit,
        )
        return compiled_forms[0], module, code

    mpi_jit = dolfinx_jit.mpi_jit_decorator(_local_jit)
    return mpi_jit(
        comm,
        ufl_object,
        form_compiler_options=form_compiler_options,
        jit_options=jit_options,
    )


def compile_form(
    comm: Any,
    form: ufl.Form,
    form_compiler_options: dict[str, Any] | None = None,
    jit_options: dict[str, Any] | None = None,
) -> CompiledRunintForm:
    """Compile a UFL form without binding DOLFINx functions/domains."""
    import ffcx

    if form_compiler_options is None:
        form_compiler_options = {}
    p_ffcx = ffcx.get_options(form_compiler_options)
    ufcx_form, module, code = jit(
        comm,
        form,
        form_compiler_options=p_ffcx,
        jit_options=jit_options,
    )
    sidecar = module._runintgen_jit.forms[0]
    return CompiledRunintForm(
        ufl_form=form,
        ufcx_form=ufcx_form,
        module=module,
        code=code,
        dtype=np.dtype(p_ffcx["scalar_type"]),
        jit_info=sidecar,
    )


def _normalised_subdomain_id(value: Any) -> int:
    return -1 if value in ("otherwise", "everywhere") else int(value)


def _runtime_providers(jit_info: JITFormInfo) -> dict[tuple[str, int], Any]:
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


def _reject_standard_quadrature_functions(form_object: ufl.Form) -> None:
    """Reject standard-integral quadrature functions until supported."""
    from runintgen.measures import is_runtime_integral
    from runintgen.quadrature_function import integral_quadrature_functions

    for integral in form_object.integrals():
        if is_runtime_integral(integral):
            continue
        functions = integral_quadrature_functions(integral)
        if not functions:
            continue
        labels = []
        for function in functions:
            spec = getattr(function, "_runintgen_quadrature_function")
            labels.append(spec.name or "<unnamed>")
        raise NotImplementedError(
            "QuadratureFunction in standard DOLFINx integrals is not "
            "implemented yet. Use a runtime measure with QuadratureRules, or "
            "provide the quantity as an ordinary DOLFINx Function if standard "
            "coefficient interpolation is intended. Affected quadrature "
            f"functions: {', '.join(labels)}."
        )


def _as_cpp_object(obj: Any) -> Any:
    return getattr(obj, "_cpp_object", obj)


def _ufl_to_dolfinx_integral_type(ufl_type: str) -> Any:
    _, integral_type = _require_dolfinx()
    mapping = {
        "cell": integral_type.cell,
        "exterior_facet": integral_type.exterior_facet,
        "interior_facet": integral_type.interior_facet,
        "vertex": integral_type.vertex,
    }
    return mapping[ufl_type]


def _integral_type_index(integral_type: Any) -> int:
    try:
        return int(integral_type)
    except (TypeError, ValueError):
        name = getattr(integral_type, "name", None)
        order = ("cell", "exterior_facet", "interior_facet", "vertex", "ridge")
        if name in order:
            return order.index(name)
        raise


def _default_cell_entities(mesh: Any) -> npt.NDArray[np.int32]:
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    return np.arange(num_cells, dtype=np.int32)


def _standard_domain(
    *,
    mesh: Any,
    integral_type: Any,
    subdomain_id: int,
    subdomains: dict[Any, list[tuple[int, npt.NDArray[np.int32]]]] | None,
) -> npt.NDArray[np.int32]:
    if subdomain_id == -1 and integral_type == _ufl_to_dolfinx_integral_type("cell"):
        return _default_cell_entities(mesh)

    if subdomains is not None:
        for sid, entities in subdomains.get(integral_type, []):
            if int(sid) == int(subdomain_id):
                return np.ascontiguousarray(entities, dtype=np.int32).ravel()

    raise RuntimeError(
        "Could not resolve integration domain for standard integral "
        f"{integral_type} id {subdomain_id}."
    )


def _raw_entity_array(data: Any) -> npt.NDArray[np.int32] | None:
    """Return ``data`` as raw local entity ids if it has that shape."""
    if data is None:
        return None
    if hasattr(data, "find") and hasattr(data, "dim"):
        return None
    try:
        values = np.asarray(data, dtype=np.int32)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1:
        return None
    return np.ascontiguousarray(values, dtype=np.int32)


def _standard_domains_from_raw_entities(
    *,
    mesh: Any,
    integral_type_name: str,
    ids: list[int],
    data: Any,
) -> list[tuple[int, npt.NDArray[np.int32]]] | None:
    """Build standard domains from raw local cell or facet ids."""
    raw = _raw_entity_array(data)
    if raw is None:
        return None

    if integral_type_name == "cell":
        entities = raw
    elif integral_type_name in {"exterior_facet", "interior_facet"}:
        entities = _facet_integration_rows(mesh, raw, integral_type_name)
    else:
        return None

    return [(sid, np.ascontiguousarray(entities, dtype=np.int32)) for sid in ids]


def _build_subdomains(
    form_object: ufl.Form,
    ufcx_form: Any,
) -> dict[Any, list[tuple[int, npt.NDArray[np.int32]]]]:
    """Build standard DOLFINx integration domains where possible."""
    from runintgen.measures import has_runtime_quadrature

    from dolfinx.fem.forms import get_integration_domains

    _require_dolfinx()
    sd = form_object.subdomain_data()
    if not sd:
        return {}
    (domain,) = list(sd.keys())
    mesh = domain.ufl_cargo()
    domain_data = sd.get(domain)
    subdomains: dict[Any, list[tuple[int, npt.NDArray[np.int32]]]] = {}
    offsets = [
        int(ufcx_form.form_integral_offsets[i]) for i in range(len(IntegralType) + 1)
    ]
    for integral_type_name, subdomain_data in domain_data.items():
        integral_type = _ufl_to_dolfinx_integral_type(integral_type_name)
        type_index = _integral_type_index(integral_type)
        integral_ids = [
            int(ufcx_form.form_integral_ids[j])
            for j in range(offsets[type_index], offsets[type_index + 1])
        ]
        ids = sorted({sid for sid in integral_ids if sid != -1})
        if not ids:
            continue
        raw_domains: list[tuple[int, npt.NDArray[np.int32]]] = []
        if mesh is not None:
            for integral in form_object.integrals():
                if integral.integral_type() != integral_type_name:
                    continue
                if has_runtime_quadrature(integral.subdomain_data()):
                    continue
                try:
                    sid = _normalised_subdomain_id(integral.subdomain_id())
                except (TypeError, ValueError):
                    continue
                if sid == -1:
                    continue
                item_domains = _standard_domains_from_raw_entities(
                    mesh=mesh,
                    integral_type_name=integral_type_name,
                    ids=[sid],
                    data=integral.subdomain_data(),
                )
                if item_domains is not None:
                    raw_domains.extend(item_domains)
        if raw_domains:
            subdomains.setdefault(integral_type, []).extend(raw_domains)
        try:
            data = subdomain_data[0]
            if all(d is data for d in subdomain_data if d is not None):
                subdomains.setdefault(integral_type, []).extend(
                    get_integration_domains(
                        integral_type,
                        data,
                        ids,
                    )
                )
        except Exception:
            continue
    return subdomains


def _active_coefficients(
    ffi_obj: cffi.FFI,
    ufcx_form: Any,
    integral: Any,
) -> npt.NDArray[np.int8]:
    if integral.enabled_coefficients == ffi_obj.NULL:
        return np.array([], dtype=np.int8)
    values = [
        i
        for i in range(int(ufcx_form.num_coefficients))
        if bool(integral.enabled_coefficients[i])
    ]
    return np.asarray(values, dtype=np.int8)


def _resolve_custom_data(
    custom_data: int | dict[tuple[Any, ...], int | Any | None] | Any | None,
    integral_type: Any,
    subdomain_id: int,
    kernel_idx: int,
) -> int | None:
    if custom_data is None:
        return None
    if not isinstance(custom_data, dict):
        return int(custom_data)

    integral_type_variants = [integral_type]
    if isinstance(integral_type, str):
        try:
            integral_type_variants.append(_ufl_to_dolfinx_integral_type(integral_type))
        except Exception:
            pass
    else:
        name = getattr(integral_type, "name", None)
        if name is not None:
            integral_type_variants.append(name)

    keys = []
    for type_variant in integral_type_variants:
        keys.extend(
            [
                (type_variant, subdomain_id, kernel_idx),
                (type_variant, subdomain_id),
            ]
        )
    keys.extend([(subdomain_id, kernel_idx), (subdomain_id,)])
    for key in keys:
        if key in custom_data:
            value = custom_data[key]
            return None if value is None else int(value)
    return None


def _mesh_basix_cell(mesh: Any) -> Any:
    if hasattr(mesh, "basix_cell"):
        return mesh.basix_cell()

    topology = getattr(mesh, "topology", None)
    cell_type = getattr(topology, "cell_type", None)
    name = getattr(cell_type, "name", None)
    if name is None:
        raise TypeError("Could not determine the mesh Basix cell type.")

    import basix

    try:
        return getattr(basix.CellType, str(name))
    except AttributeError as exc:
        raise TypeError(
            f"Could not map DOLFINx cell type {name!r} to a Basix cell type."
        ) from exc


def _facet_integration_rows(
    mesh: Any,
    facets: npt.ArrayLike,
    integral_type: str,
) -> npt.NDArray[np.int32]:
    raw = np.ascontiguousarray(facets, dtype=np.int32).ravel()
    if raw.size == 0:
        width = 2 if integral_type == "exterior_facet" else 4
        return np.empty((0, width), dtype=np.int32)

    dtype = np.dtype(mesh.geometry.x.dtype)
    if dtype == np.dtype(np.float64):
        flat = _cpp.facet_integration_rows_float64(
            _as_cpp_object(mesh), raw, integral_type
        )
    elif dtype == np.dtype(np.float32):
        flat = _cpp.facet_integration_rows_float32(
            _as_cpp_object(mesh), raw, integral_type
        )
    else:
        raise ValueError(f"Unsupported mesh geometry dtype {dtype}.")

    width = 2 if integral_type == "exterior_facet" else 4
    return np.asarray(flat, dtype=np.int32).reshape((-1, width))


def _facet_payload_with_rows(
    *,
    mesh: Any,
    provider: Any,
    integral_type: str,
) -> tuple[npt.NDArray[np.int32], RuntimeQuadraturePayload]:
    """Map raw facet ids to rows and preserve mixed standard/runtime entries."""
    from runintgen.runtime_data import (
        RuntimeEntityMap,
        RuntimeQuadraturePayload,
        as_runtime_quadrature_payload,
        facet_runtime_quadrature_payload,
        interior_facet_runtime_quadrature_payload,
    )

    source_payload = as_runtime_quadrature_payload(provider)
    source_rules = source_payload.rules
    if source_rules.parent_map is None:
        raise RuntimeError(
            f"Runtime {integral_type} quadrature requires QuadratureRules.parent_map "
            "with local facet ids."
        )

    raw_entities = source_payload.entities.entity_indices
    if raw_entities.ndim != 1:
        raise ValueError(
            "CutFEMx mixed facet subdomain_data expects raw local facet ids. "
            "Do not pass pre-expanded DOLFINx facet rows."
        )

    loop_rows = _facet_integration_rows(mesh, raw_entities, integral_type)
    cut_mask = source_payload.entities.is_cut != 0
    if int(np.count_nonzero(cut_mask)) != int(source_rules.num_rules):
        raise ValueError("Runtime facet entity map disagrees with quadrature rules.")

    runtime_rows = np.empty(
        (source_rules.num_rules, loop_rows.shape[1]), dtype=np.int32
    )
    for row, rule_index in zip(
        loop_rows[cut_mask],
        source_payload.entities.rule_indices[cut_mask],
        strict=True,
    ):
        runtime_rows[int(rule_index)] = row

    runtime_payload = RuntimeQuadraturePayload(
        rules=source_rules,
        entities=RuntimeEntityMap.runtime_only(source_rules),
        quadrature_functions=source_payload.quadrature_functions,
    )
    if integral_type == "exterior_facet":
        mapped_runtime = facet_runtime_quadrature_payload(
            parent_cell_type=_mesh_basix_cell(mesh),
            quadrature=runtime_payload,
            entity_indices=runtime_rows,
        )
    elif integral_type == "interior_facet":
        mapped_runtime = interior_facet_runtime_quadrature_payload(
            parent_cell_type=_mesh_basix_cell(mesh),
            quadrature=runtime_payload,
            entity_indices=runtime_rows,
        )
    else:  # pragma: no cover - caller controls this.
        raise ValueError(f"Unsupported facet integral type {integral_type!r}.")

    entities = RuntimeEntityMap(
        entity_indices=np.ascontiguousarray(loop_rows, dtype=np.int32),
        is_cut=source_payload.entities.is_cut,
        rule_indices=source_payload.entities.rule_indices,
    )
    mapped_payload = RuntimeQuadraturePayload(
        rules=mapped_runtime.rules,
        entities=entities,
        quadrature_functions=source_payload.quadrature_functions,
    )
    return mapped_payload.entity_indices, mapped_payload


def _needs_physical_points_for_q_functions(
    module: Any,
    rules: QuadratureRules,
) -> bool:
    from runintgen.quadrature_function import (
        quadrature_function_source,
        quadrature_function_values,
    )

    for info in getattr(module, "quadrature_functions", []) or []:
        explicit = quadrature_function_values(info.terminal)
        if str(rules.rule_id) in explicit:
            continue
        source = quadrature_function_source(info.terminal)
        if source is not None:
            if hasattr(source, "evaluate"):
                continue
            return True
        if hasattr(info.terminal, "eval") and hasattr(info.terminal, "function_space"):
            return True
    return False


def _quadrature_for_custom_data(
    mesh: Any,
    module: Any,
    provider: Any,
) -> Any:
    from runintgen.runtime_data import (
        RuntimeQuadraturePayload,
        as_runtime_quadrature_payload,
    )

    payload = as_runtime_quadrature_payload(provider)
    rules = payload.rules
    if (
        rules.physical_points is None
        and _needs_physical_points_for_q_functions(module, rules)
    ):
        # CutFEMx runtime quadrature providers expose physical points lazily.
        physical_points = getattr(rules, "physical_points", None)
        if physical_points is None and hasattr(rules, "with_physical_points"):
            physical_points = rules.with_physical_points().physical_points
        if physical_points is None:
            physical_points = _compute_physical_points(mesh, rules)
        from runintgen.runtime_data import QuadratureRules

        rules = QuadratureRules(
            tdim=rules.tdim,
            points=rules.points,
            secondary_points=rules.secondary_points,
            weights=rules.weights,
            offsets=rules.offsets,
            parent_map=rules.parent_map,
            rule_id=rules.rule_id,
            kind=rules.kind,
            gdim=mesh.geometry.dim,
            physical_points=physical_points,
        )
        return RuntimeQuadraturePayload(
            rules=rules,
            entities=payload.entities,
            quadrature_functions=payload.quadrature_functions,
        )
    return provider


def _runtime_domain_and_custom_data(
    *,
    compiled: CompiledRunintForm,
    mesh: Any,
    integral_type: str,
    subdomain_id: int,
    kernel_idx: int,
    custom_data: int | dict[tuple[Any, ...], int | Any | None] | Any | None,
    cache: dict[tuple[int, str, int], Any],
) -> tuple[npt.NDArray[np.int32], int | None, Any | None]:
    """Return runtime entity domain and custom-data pointer for one integral."""
    from runintgen.basix_runtime import CustomData
    from runintgen.runtime_data import as_runtime_quadrature_payload

    providers = _runtime_providers(compiled.jit_info)
    provider = providers.get((integral_type, subdomain_id))
    if provider is None:
        raise RuntimeError(
            "Could not find runtime quadrature provider for "
            f"{integral_type} subdomain {subdomain_id}."
        )

    if integral_type == "cell":
        payload = as_runtime_quadrature_payload(provider)
        if payload.parent_map is None:
            raise RuntimeError(
                "DOLFINx runtime quadrature requires QuadratureRules.parent_map "
                "so per-entity rules can be mapped to mesh entities."
            )
        entities = np.ascontiguousarray(payload.entity_indices, dtype=np.int32)
        quadrature_provider = provider
    elif integral_type in {"exterior_facet", "interior_facet"}:
        entities, quadrature_provider = _facet_payload_with_rows(
            mesh=mesh,
            provider=provider,
            integral_type=integral_type,
        )
    else:
        raise NotImplementedError(
            "CutFEMx runtime forms do not support runtime "
            f"{integral_type!r} integrals yet."
        )

    data_ptr = _resolve_custom_data(
        custom_data,
        integral_type,
        subdomain_id,
        kernel_idx,
    )
    owner = None
    if data_ptr is None:
        cache_key = (id(provider), integral_type, subdomain_id)
        owner = cache.get(cache_key)
        if owner is None:
            quadrature = _quadrature_for_custom_data(
                mesh,
                compiled.jit_info.module,
                quadrature_provider,
            )
            owner = CustomData(
                compiled.jit_info.module,
                quadrature=quadrature,
                quadrature_function_evaluator=(
                    _evaluate_background_quadrature_function
                ),
            )
            cache[cache_key] = owner
        data_ptr = int(owner.ptr)

    return entities, data_ptr, owner


def _compute_physical_points(
    mesh: Any,
    rules: QuadratureRules,
) -> npt.NDArray[np.float64]:
    """Map parent-cell reference quadrature points to physical points."""
    geometry = mesh.geometry
    dofmap = geometry.dofmaps[0]
    x = geometry.x
    gdim = int(getattr(geometry, "dim", x.shape[1]))
    if rules.parent_map is None:
        raise ValueError("QuadratureRules.parent_map is required.")
    if rules.gdim is not None and int(rules.gdim) != gdim:
        raise ValueError(
            f"QuadratureRules.gdim={rules.gdim} does not match mesh geometry "
            f"dimension {gdim}."
        )
    parent_map = np.asarray(rules.parent_map, dtype=np.int32)
    if np.any(parent_map < 0) or np.any(parent_map >= dofmap.shape[0]):
        raise ValueError("QuadratureRules.parent_map contains invalid local cells.")

    ref_points = _points_as_2d(rules.points, rules.tdim)
    physical_points = np.empty((gdim, rules.total_points), dtype=np.float64)
    cmap = geometry.cmaps[0]
    if rules.kind == "per_entity":
        if rules.offsets is None:
            raise ValueError("per-entity QuadratureRules require offsets.")
        for rule_index, cell in enumerate(parent_map):
            q0 = int(rules.offsets[rule_index])
            q1 = int(rules.offsets[rule_index + 1])
            if q0 == q1:
                continue
            cell_geometry = np.asarray(x[dofmap[int(cell)], :gdim], dtype=np.float64)
            physical_points[:, q0:q1] = cmap.push_forward(
                ref_points[q0:q1], cell_geometry
            ).T
    elif rules.kind == "shared":
        nq = int(ref_points.shape[0])
        for entity_index, cell in enumerate(parent_map):
            q0 = entity_index * nq
            q1 = q0 + nq
            cell_geometry = np.asarray(x[dofmap[int(cell)], :gdim], dtype=np.float64)
            physical_points[:, q0:q1] = cmap.push_forward(
                ref_points, cell_geometry
            ).T
    else:  # pragma: no cover - QuadratureRules validates this.
        raise ValueError(f"Unsupported QuadratureRules kind {rules.kind!r}.")
    return physical_points


def _points_as_2d(
    points: npt.NDArray[np.float64],
    tdim: int,
) -> npt.NDArray[np.float64]:
    if tdim == 0:
        if points.ndim == 2:
            return points
        return points.reshape((-1, 0))
    if points.ndim == 2:
        return points
    return points.reshape((-1, tdim))


def _cells_for_quadrature_points(rules: QuadratureRules) -> npt.NDArray[np.int32]:
    if rules.parent_map is None:
        raise ValueError("QuadratureRules.parent_map is required.")
    if rules.kind == "per_entity":
        if rules.offsets is None:
            raise ValueError("per-entity QuadratureRules require offsets.")
        counts = np.diff(rules.offsets).astype(np.int64, copy=False)
        return np.repeat(rules.parent_map, counts).astype(np.int32, copy=False)
    if rules.kind == "shared":
        return np.repeat(rules.parent_map, rules.weights.size).astype(
            np.int32,
            copy=False,
        )
    raise ValueError(f"Unsupported QuadratureRules kind {rules.kind!r}.")


def _points_for_dolfinx_eval(rules: QuadratureRules) -> npt.NDArray[np.float64]:
    if rules.physical_points is None:
        raise ValueError("QuadratureRules.physical_points is required.")
    points = np.asarray(rules.physical_points, dtype=np.float64).T
    if points.shape[1] == 3:
        return np.ascontiguousarray(points, dtype=np.float64)
    if points.shape[1] > 3:
        raise ValueError("DOLFINx Function.eval expects at most 3 coordinates.")
    padded = np.zeros((points.shape[0], 3), dtype=np.float64)
    padded[:, : points.shape[1]] = points
    return padded


def _evaluate_background_quadrature_function(
    info: Any,
    rules: QuadratureRules,
) -> npt.NDArray[np.float64]:
    """Evaluate a DOLFINx-backed quadrature function source."""
    from runintgen.quadrature_function import quadrature_function_source
    from runintgen.runtime_data import QuadratureEvaluationContext

    function = info.terminal
    source = quadrature_function_source(function)
    if source is not None and hasattr(source, "evaluate"):
        context = QuadratureEvaluationContext(
            rules=rules,
            info=info,
            mesh=getattr(function.function_space, "mesh", None),
        )
        return source.evaluate(context)

    if not hasattr(function, "eval") or not hasattr(function, "function_space"):
        raise ValueError(
            f"QuadratureFunction {info.label!r} has no explicit values, no "
            "callable source, and no DOLFINx background Function evaluator."
        )

    if rules.physical_points is None:
        physical_points = _compute_physical_points(function.function_space.mesh, rules)
        from runintgen.runtime_data import QuadratureRules

        rules = QuadratureRules(
            tdim=rules.tdim,
            points=rules.points,
            secondary_points=rules.secondary_points,
            weights=rules.weights,
            offsets=rules.offsets,
            parent_map=rules.parent_map,
            rule_id=rules.rule_id,
            kind=rules.kind,
            gdim=function.function_space.mesh.geometry.dim,
            physical_points=physical_points,
        )

    points = _points_for_dolfinx_eval(rules)
    cells = _cells_for_quadrature_points(rules)
    values = np.asarray(function.eval(points, cells), dtype=np.float64)
    if rules.total_points == 1:
        values = values.reshape(1, info.value_size)
    return values


__all__ = [
    "CompiledRunintForm",
    "QuadratureFunction",
    "_active_coefficients",
    "_as_cpp_object",
    "_build_subdomains",
    "_reject_standard_quadrature_functions",
    "_require_dolfinx",
    "_resolve_custom_data",
    "_runtime_domain_and_custom_data",
    "_standard_domain",
    "_ufl_to_dolfinx_integral_type",
    "compile_form",
    "jit",
]
