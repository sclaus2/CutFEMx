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
    "normal",
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
        raise ValueError("NormalEvaluator requires quadrature points with tdim > 0.")
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


def normal(level_set: Function, *, name: str | None = None, sign: float = 1.0) -> Function:
    """Return a lazy quadrature function for ``sign * grad(phi)/|grad(phi)|``."""
    mesh = level_set.function_space.mesh
    gdim = mesh.geometry.dim
    qf = QuadratureFunction(mesh, name=name or "normal", shape=(gdim,))
    qf.set_evaluator(NormalEvaluator(level_set, sign=sign))
    return qf
