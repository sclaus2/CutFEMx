# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import typing

import numpy as np
import numpy.typing as npt

from cutfemx import cutfemx_cpp as _cpp
from cutfemx._runintgen_adapter import QuadratureFunction
from dolfinx.fem import Function

__all__ = [
    "NormalEvaluator",
    "conormal",
    "correction_distance",
    "level_set_value",
    "normal",
    "surface_normal",
]


def _require_c_contiguous_array(
    value: typing.Any, dtype: np.dtype, name: str
) -> npt.NDArray[typing.Any]:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(value).__name__}.")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {value.dtype}.")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous.")
    return value


def _points_as_2d_view(
    points: npt.NDArray[typing.Any], tdim: int
) -> npt.NDArray[typing.Any]:
    if tdim <= 0:
        raise ValueError(
            "Level-set quadrature evaluators require points with tdim > 0."
        )
    if points.ndim == 2:
        if points.shape[1] != tdim:
            raise ValueError(
                f"rules.points second dimension must equal tdim={tdim}, "
                f"got {points.shape[1]}."
            )
        return points
    if points.ndim != 1:
        raise ValueError("rules.points must be a one- or two-dimensional array.")
    if points.size % tdim != 0:
        raise ValueError("rules.points size is not divisible by rules.tdim.")
    return points.reshape((-1, tdim))


def _evaluate_vector_quadrature_function(
    function: typing.Any,
    context: typing.Any,
    *,
    label: str,
) -> npt.NDArray[np.float64]:
    spec = getattr(function, "_runintgen_quadrature_function", None)
    if spec is None:
        raise TypeError(f"{label} must be a CutFEMx QuadratureFunction.")
    if not spec.value_shape:
        raise ValueError(f"{label} must be vector-valued.")
    value_size = int(spec.value_size)
    source = getattr(function, "_runintgen_source", None)
    if source is None or not hasattr(source, "evaluate"):
        raise ValueError(
            f"{label} must have a context-aware evaluator in the first pass."
        )
    values = np.asarray(source.evaluate(context), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, value_size))
    if values.shape != (context.rules.total_points, value_size):
        raise ValueError(
            f"{label} evaluator returned shape {values.shape}, expected "
            f"({context.rules.total_points}, {value_size})."
        )
    return np.ascontiguousarray(values, dtype=np.float64)


class _LevelSetValueEvaluator:
    """Batch evaluator for level-set values at runtime quadrature points."""

    def __init__(self, level_set: Function) -> None:
        self.level_set = level_set
        self.version = 0

    def cache_key(self, context: typing.Any) -> tuple[int, int]:
        return (id(self.level_set), self.version)

    def invalidate(self) -> None:
        self.version += 1

    def update(self, *, version: int | None = None) -> None:
        self.version = self.version + 1 if version is None else int(version)

    def evaluate(self, context: typing.Any) -> npt.NDArray[np.float64]:
        rules = context.rules
        if rules.parent_map is None:
            raise ValueError("level_set_value requires QuadratureRules.parent_map.")
        if rules.kind != "per_entity":
            raise ValueError("level_set_value currently requires per-entity rules.")
        if rules.offsets is None:
            raise ValueError("level_set_value requires per-entity offsets.")

        dtype = np.dtype(self.level_set.x.array.dtype)
        points = _points_as_2d_view(
            _require_c_contiguous_array(rules.points, dtype, "rules.points"),
            int(rules.tdim),
        )
        if not points.flags.c_contiguous:
            raise ValueError("rules.points must remain C-contiguous after reshaping.")
        offsets = _require_c_contiguous_array(
            rules.offsets, np.dtype(np.int32), "rules.offsets"
        )
        parent_map = _require_c_contiguous_array(
            rules.parent_map, np.dtype(np.int32), "rules.parent_map"
        )
        if dtype == np.dtype(np.float64):
            values = _cpp.level_set.evaluate_values_float64(
                self.level_set._cpp_object,
                points,
                offsets,
                parent_map,
            )
        elif dtype == np.dtype(np.float32):
            values = _cpp.level_set.evaluate_values_float32(
                self.level_set._cpp_object,
                points,
                offsets,
                parent_map,
            )
        else:
            raise ValueError(f"Unsupported level-set scalar dtype {dtype}.")

        return _require_c_contiguous_array(
            values, np.dtype(np.float64), "level-set values"
        )


class NormalEvaluator:
    """Batch evaluator for level-set normals at runtime quadrature points."""

    def __init__(self, level_set: Function, *, sign: float = 1.0) -> None:
        self.level_set = level_set
        self.sign = float(sign)
        self.version = 0

    def cache_key(self, context: typing.Any) -> tuple[int, int, float]:
        return (id(self.level_set), self.version, self.sign)

    def invalidate(self) -> None:
        self.version += 1

    def update(self, *, version: int | None = None) -> None:
        self.version = self.version + 1 if version is None else int(version)

    def evaluate(self, context: typing.Any) -> npt.NDArray[np.float64]:
        rules = context.rules
        if rules.parent_map is None:
            raise ValueError("NormalEvaluator requires QuadratureRules.parent_map.")
        if rules.kind != "per_entity":
            raise ValueError("NormalEvaluator currently requires per-entity rules.")
        if rules.offsets is None:
            raise ValueError("NormalEvaluator requires per-entity offsets.")

        dtype = np.dtype(self.level_set.x.array.dtype)
        points = _points_as_2d_view(
            _require_c_contiguous_array(rules.points, dtype, "rules.points"),
            int(rules.tdim),
        )
        if not points.flags.c_contiguous:
            raise ValueError("rules.points must remain C-contiguous after reshaping.")
        offsets = _require_c_contiguous_array(
            rules.offsets, np.dtype(np.int32), "rules.offsets"
        )
        parent_map = _require_c_contiguous_array(
            rules.parent_map, np.dtype(np.int32), "rules.parent_map"
        )
        if dtype == np.dtype(np.float64):
            values = _cpp.level_set.evaluate_normals_float64(
                self.level_set._cpp_object,
                points,
                offsets,
                parent_map,
                self.sign,
            )
        elif dtype == np.dtype(np.float32):
            values = _cpp.level_set.evaluate_normals_float32(
                self.level_set._cpp_object,
                points,
                offsets,
                parent_map,
                self.sign,
            )
        else:
            raise ValueError(f"Unsupported level-set scalar dtype {dtype}.")

        gdim = self.level_set.function_space.mesh.geometry.dim
        values = _require_c_contiguous_array(
            values, np.dtype(np.float64), "normal values"
        )
        return values.reshape((points.shape[0], gdim))


def _single_equality_level_set_index(cut_data: typing.Any, selector: str) -> int:
    text = str(selector).replace(" ", "")
    if not text.endswith("=0") or any(op in text[:-2] for op in ("<", ">")):
        raise ValueError(
            "surface_normal currently requires a single equality selector "
            "such as 'phi=0'."
        )
    name = text[:-2]
    names = tuple(cut_data.level_set_names)
    if name in names:
        return names.index(name)
    raise ValueError(
        f"Unknown level-set selector name {name!r}. Available names are "
        f"{', '.join(names)}."
    )


def _quadrature_owner_from_context(context: typing.Any) -> typing.Any:
    rules = context.rules
    owner = getattr(rules, "_cutfemx_owner", None)
    if owner is not None:
        return owner

    metadata = context.metadata or {}
    base_rules = metadata.get("base_rules")
    owner = getattr(base_rules, "_cutfemx_owner", None)
    if owner is not None:
        return owner

    raise ValueError(
        "surface_normal requires CutFEMx runtime quadrature provenance. "
        "Use cutfemx.runtime_quadrature(cut_data, 'phi=0', ...) with the "
        "straight backend."
    )


class _SurfaceNormalEvaluator:
    """Batch evaluator for geometric cut-surface normals."""

    def __init__(self, cut_data: typing.Any, selector: str) -> None:
        self.cut_data = cut_data
        self.selector = str(selector)
        self.level_set_index = _single_equality_level_set_index(cut_data, selector)
        self.version = 0

    def cache_key(self, context: typing.Any) -> tuple[int, str, int]:
        owner = _quadrature_owner_from_context(context)
        return (id(owner), self.selector, self.version)

    def invalidate(self) -> None:
        self.version += 1

    def update(self, *, version: int | None = None) -> None:
        self.version = self.version + 1 if version is None else int(version)

    def evaluate(self, context: typing.Any) -> npt.NDArray[np.float64]:
        rules = context.rules
        if rules.parent_map is None:
            raise ValueError("surface_normal requires QuadratureRules.parent_map.")
        if rules.kind != "per_entity":
            raise ValueError("surface_normal currently requires per-entity rules.")
        if rules.offsets is None:
            raise ValueError("surface_normal requires per-entity offsets.")

        owner = _quadrature_owner_from_context(context)
        dtype = np.dtype(rules.points.dtype)
        points = _points_as_2d_view(
            _require_c_contiguous_array(rules.points, dtype, "rules.points"),
            int(rules.tdim),
        )
        if not points.flags.c_contiguous:
            raise ValueError("rules.points must remain C-contiguous after reshaping.")
        offsets = _require_c_contiguous_array(
            rules.offsets, np.dtype(np.int32), "rules.offsets"
        )
        parent_map = _require_c_contiguous_array(
            rules.parent_map, np.dtype(np.int32), "rules.parent_map"
        )

        if dtype == np.dtype(np.float64):
            values = _cpp.level_set.evaluate_surface_normals_float64(
                self.cut_data._cpp_object,
                owner,
                self.level_set_index,
                points,
                offsets,
                parent_map,
            )
        elif dtype == np.dtype(np.float32):
            values = _cpp.level_set.evaluate_surface_normals_float32(
                self.cut_data._cpp_object,
                owner,
                self.level_set_index,
                points,
                offsets,
                parent_map,
            )
        else:
            raise ValueError(f"Unsupported runtime quadrature point dtype {dtype}.")

        gdim = self.cut_data.gdim
        values = _require_c_contiguous_array(
            values, np.dtype(np.float64), "surface normal values"
        )
        return values.reshape((points.shape[0], gdim))


class _CorrectionDistanceEvaluator:
    """Batch evaluator for signed correction distance along a direction."""

    def __init__(
        self,
        level_set: Function,
        cut_data: typing.Any,
        selector: str,
        direction: typing.Any,
        *,
        max_distance: float | None,
        max_distance_factor: float,
        tolerance: float,
        max_iterations: int,
    ) -> None:
        self.level_set = level_set
        self.cut_data = cut_data
        self.selector = str(selector)
        self.direction = direction
        self.level_set_index = _single_equality_level_set_index(cut_data, selector)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.max_distance_factor = float(max_distance_factor)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.version = 0
        if self.max_distance is not None and self.max_distance <= 0.0:
            raise ValueError("max_distance must be positive when supplied.")
        if self.max_distance is None and self.max_distance_factor <= 0.0:
            raise ValueError("max_distance_factor must be positive.")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")

    def cache_key(self, context: typing.Any) -> tuple[typing.Any, ...]:
        direction_source = getattr(self.direction, "_runintgen_source", None)
        direction_key_fn = getattr(direction_source, "cache_key", None)
        direction_key = (
            direction_key_fn(context)
            if direction_key_fn is not None
            else id(direction_source)
        )
        return (
            id(self.level_set),
            id(self.cut_data),
            self.selector,
            id(self.direction),
            direction_key,
            self.max_distance,
            self.max_distance_factor,
            self.tolerance,
            self.max_iterations,
            self.version,
        )

    def invalidate(self) -> None:
        self.version += 1

    def update(self, *, version: int | None = None) -> None:
        self.version = self.version + 1 if version is None else int(version)

    def evaluate(self, context: typing.Any) -> npt.NDArray[np.float64]:
        rules = context.rules
        if rules.parent_map is None:
            raise ValueError("correction_distance requires QuadratureRules.parent_map.")
        if rules.kind != "per_entity":
            raise ValueError(
                "correction_distance currently requires per-entity rules."
            )
        if rules.offsets is None:
            raise ValueError("correction_distance requires QuadratureRules.offsets.")

        directions = _evaluate_vector_quadrature_function(
            self.direction,
            context,
            label="correction_distance direction",
        )
        directions = np.ascontiguousarray(directions, dtype=np.float64)

        dtype = np.dtype(self.level_set.x.array.dtype)
        points = _points_as_2d_view(
            _require_c_contiguous_array(rules.points, dtype, "rules.points"),
            int(rules.tdim),
        )
        if not points.flags.c_contiguous:
            raise ValueError("rules.points must remain C-contiguous after reshaping.")
        offsets = _require_c_contiguous_array(
            rules.offsets, np.dtype(np.int32), "rules.offsets"
        )
        parent_map = _require_c_contiguous_array(
            rules.parent_map, np.dtype(np.int32), "rules.parent_map"
        )
        max_distance = -1.0 if self.max_distance is None else self.max_distance

        if dtype == np.dtype(np.float64):
            values = _cpp.level_set.evaluate_correction_distances_float64(
                self.level_set._cpp_object,
                points,
                offsets,
                parent_map,
                directions,
                max_distance,
                self.max_distance_factor,
                self.tolerance,
                self.max_iterations,
            )
        elif dtype == np.dtype(np.float32):
            values = _cpp.level_set.evaluate_correction_distances_float32(
                self.level_set._cpp_object,
                points,
                offsets,
                parent_map,
                directions,
                max_distance,
                self.max_distance_factor,
                self.tolerance,
                self.max_iterations,
            )
        else:
            raise ValueError(f"Unsupported level-set scalar dtype {dtype}.")

        return _require_c_contiguous_array(
            values, np.dtype(np.float64), "correction distance values"
        )


class _ConormalEvaluator:
    """Batch evaluator for surface DG conormals on runtime interior facets."""

    def __init__(
        self,
        normal_field: typing.Any,
        *,
        tolerance: float,
    ) -> None:
        self.normal_field = normal_field
        self.tolerance = float(tolerance)
        self.version = 0
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")

    def cache_key(self, context: typing.Any) -> tuple[typing.Any, ...]:
        normal_source = getattr(self.normal_field, "_runintgen_source", None)
        normal_key_fn = getattr(normal_source, "cache_key", None)
        normal_key = (
            normal_key_fn(context)
            if normal_key_fn is not None
            else id(normal_source)
        )
        side = None if context.metadata is None else context.metadata.get("side")
        return (
            id(self.normal_field),
            normal_key,
            side,
            self.tolerance,
            self.version,
        )

    def invalidate(self) -> None:
        self.version += 1

    def update(self, *, version: int | None = None) -> None:
        self.version = self.version + 1 if version is None else int(version)

    def evaluate(self, context: typing.Any) -> npt.NDArray[np.float64]:
        metadata = context.metadata or {}
        side = metadata.get("side")
        if side not in {"+", "-"}:
            raise ValueError(
                "conormal is side-aware and must be used as mu('+') or mu('-') "
                "on a runtime dS measure."
            )
        local_facets = metadata.get("local_facets")
        if local_facets is None:
            raise ValueError("conormal requires local facet metadata from dS assembly.")
        local_facets = _require_c_contiguous_array(
            local_facets, np.dtype(np.int32), "local_facets"
        )

        mesh = self.normal_field.function_space.mesh
        surface_normals = _evaluate_vector_quadrature_function(
            self.normal_field,
            context,
            label="conormal normal field",
        )
        surface_normals = np.ascontiguousarray(surface_normals, dtype=np.float64)
        rules = context.rules
        if rules.parent_map is None:
            raise ValueError("conormal requires QuadratureRules.parent_map.")
        if rules.kind != "per_entity":
            raise ValueError("conormal currently requires per-entity rules.")
        if rules.offsets is None:
            raise ValueError("conormal requires QuadratureRules.offsets.")
        offsets = _require_c_contiguous_array(
            rules.offsets, np.dtype(np.int32), "rules.offsets"
        )
        parent_map = _require_c_contiguous_array(
            rules.parent_map, np.dtype(np.int32), "rules.parent_map"
        )
        dtype = np.dtype(mesh.geometry.x.dtype)
        if dtype == np.dtype(np.float64):
            values = _cpp.level_set.evaluate_conormals_float64(
                mesh._cpp_object,
                surface_normals,
                offsets,
                parent_map,
                local_facets,
                self.tolerance,
            )
        elif dtype == np.dtype(np.float32):
            values = _cpp.level_set.evaluate_conormals_float32(
                mesh._cpp_object,
                surface_normals,
                offsets,
                parent_map,
                local_facets,
                self.tolerance,
            )
        else:
            raise ValueError(f"Unsupported mesh geometry dtype {dtype}.")

        values = _require_c_contiguous_array(
            values, np.dtype(np.float64), "conormal values"
        )
        return values.reshape(surface_normals.shape)


def level_set_value(level_set: Function, *, name: str | None = None) -> Function:
    """Return a lazy quadrature function for ``phi``."""
    mesh = level_set.function_space.mesh
    qf = QuadratureFunction(mesh, name=name or "level_set_value")
    qf.set_evaluator(_LevelSetValueEvaluator(level_set))
    return qf


def normal(level_set: Function, *, name: str | None = None, sign: float = 1.0) -> Function:
    """Return a lazy quadrature function for ``sign * grad(phi)/|grad(phi)|``."""
    mesh = level_set.function_space.mesh
    gdim = mesh.geometry.dim
    qf = QuadratureFunction(mesh, name=name or "normal", shape=(gdim,))
    qf.set_evaluator(NormalEvaluator(level_set, sign=sign))
    return qf


def surface_normal(
    cut_data: typing.Any,
    selector: str,
    *,
    name: str | None = None,
) -> Function:
    """Return a lazy quadrature function for the geometric cut-surface normal."""
    mesh = cut_data.mesh
    gdim = mesh.geometry.dim
    qf = QuadratureFunction(mesh, name=name or "surface_normal", shape=(gdim,))
    qf.set_evaluator(_SurfaceNormalEvaluator(cut_data, selector))
    return qf


def correction_distance(
    level_set: Function,
    cut_data: typing.Any,
    selector: str,
    *,
    direction: typing.Any,
    max_distance: float | None = None,
    max_distance_factor: float = 1.0,
    tolerance: float = 1.0e-12,
    max_iterations: int = 64,
    name: str | None = None,
) -> Function:
    """Return signed distance along ``direction`` to ``level_set == 0``."""
    mesh = level_set.function_space.mesh
    qf = QuadratureFunction(mesh, name=name or "correction_distance")
    qf.set_evaluator(
        _CorrectionDistanceEvaluator(
            level_set,
            cut_data,
            selector,
            direction,
            max_distance=max_distance,
            max_distance_factor=max_distance_factor,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )
    return qf


def conormal(
    normal_field: typing.Any,
    *,
    tolerance: float = 1.0e-14,
    name: str | None = None,
) -> Function:
    """Return a side-aware surface conormal QuadratureFunction."""
    spec = getattr(normal_field, "_runintgen_quadrature_function", None)
    if spec is None:
        raise TypeError("conormal expects a vector-valued QuadratureFunction.")
    if not spec.value_shape or len(spec.value_shape) != 1:
        raise ValueError("conormal expects a vector-valued QuadratureFunction.")

    mesh = normal_field.function_space.mesh
    gdim = mesh.geometry.dim
    if int(spec.value_shape[0]) != gdim:
        raise ValueError(
            "conormal normal-field dimension must match mesh geometry dimension."
        )
    qf = QuadratureFunction(mesh, name=name or "conormal", shape=(gdim,))
    qf.set_evaluator(_ConormalEvaluator(normal_field, tolerance=tolerance))
    return qf
