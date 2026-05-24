# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import basix.ufl
import ufl

from dolfinx.fem import Function
from dolfinx.mesh import Mesh
from runintgen import QuadratureRules

from cutfemx import cutfemx_cpp as _cpp


class RuntimeQuadratureRules(QuadratureRules):
    """runintgen quadrature rules backed by a CutFEMx C++ owner."""

    def __getattribute__(self, name: str):
        if name == "physical_points":
            value = object.__getattribute__(self, "__dict__").get(
                "physical_points", None
            )
            if value is not None:
                return value

            owner = object.__getattribute__(self, "__dict__").get(
                "_cutfemx_owner", None
            )
            if owner is None:
                return None

            physical_points = owner.physical_points
            if physical_points.ndim != 2:
                raise ValueError("physical_points must have shape (gdim, total_nq).")
            gdim = int(owner.gdim)
            if physical_points.shape[0] != gdim:
                raise ValueError("physical_points first dimension must equal gdim.")
            if physical_points.shape[1] != self.total_points:
                raise ValueError("physical_points second dimension must equal total_nq.")

            object.__setattr__(self, "gdim", gdim)
            object.__setattr__(self, "physical_points", physical_points)
            return physical_points

        return super().__getattribute__(name)

    def with_physical_points(self) -> "RuntimeQuadratureRules":
        """Fill the C++ physical-point cache and return this rule object."""
        _ = self.physical_points
        return self


class CutMesh:
    """Python handle for a CutFEMx visualisation mesh."""

    def __init__(self, cpp_object):
        self._cpp_object = cpp_object
        self._mesh = None
        cpp_mesh = cpp_object.cut_mesh
        if cpp_mesh is not None:
            cell_type = cpp_mesh.topology.cell_type
            gdim = cpp_mesh.geometry.dim
            dtype = cpp_mesh.geometry.x.dtype
            domain = ufl.Mesh(
                basix.ufl.element(
                    "Lagrange", cell_type.name, 1, shape=(gdim,), dtype=dtype
                )
            )
            self._mesh = Mesh(cpp_mesh, domain)

    @property
    def mesh(self) -> Mesh | None:
        """DOLFINx mesh for the selected cut geometry."""
        return self._mesh

    @property
    def parent_index(self) -> npt.NDArray[np.int32]:
        """Parent background-cell index for each output cell."""
        return self._cpp_object.parent_index

    @property
    def is_cut_cell(self) -> npt.NDArray[np.int8]:
        """One for cut output cells, zero for uncut background cells."""
        return self._cpp_object.is_cut_cell


class CutData:
    """Python handle for CutFEMx cut data."""

    def __init__(self, cpp_object):
        self._cpp_object = cpp_object

    def update(self) -> None:
        """Refresh the current cut state from the held level-set values."""
        self._cpp_object.update()

    @property
    def tdim(self) -> int:
        return int(self._cpp_object.tdim)

    @property
    def gdim(self) -> int:
        return int(self._cpp_object.gdim)

    @property
    def num_local_cells(self) -> int:
        return int(self._cpp_object.num_local_cells)

    @property
    def level_set_names(self) -> tuple[str, ...]:
        return tuple(self._cpp_object.level_set_names)


def _candidate_entities(
    entities: npt.ArrayLike | None, entity_dim: int | None
) -> tuple[npt.NDArray[np.int32] | None, int | None]:
    if entities is None:
        if entity_dim is not None:
            raise ValueError("entity_dim is only valid when entities are supplied")
        return None, None

    if entity_dim is None:
        raise ValueError("entity_dim must be supplied when entities are supplied")

    return np.ascontiguousarray(np.asarray(entities, dtype=np.int32)), entity_dim


def _normalise_level_sets(level_set: Function | Sequence[Function]) -> list[Function]:
    if isinstance(level_set, Function):
        return [level_set]
    if isinstance(level_set, (str, bytes)):
        raise TypeError(
            "cutfemx.cut expects a Function or a non-empty sequence of Functions"
        )
    if not isinstance(level_set, Sequence):
        raise TypeError(
            "cutfemx.cut expects a Function or a non-empty sequence of Functions"
        )

    level_sets = list(level_set)
    if not level_sets:
        raise ValueError("cutfemx.cut requires at least one level-set function")
    for item in level_sets:
        if not isinstance(item, Function):
            raise TypeError(
                "cutfemx.cut sequence entries must be dolfinx.fem.Function objects"
            )
    return level_sets


def cut(
    level_set: Function | Sequence[Function],
    entities: npt.ArrayLike | None = None,
    entity_dim: int | None = None,
) -> CutData:
    """Cut one or more scalar level-set functions on cells or selected entities."""
    level_sets = _normalise_level_sets(level_set)
    candidate_entities, dim = _candidate_entities(entities, entity_dim)
    if len(level_sets) == 1:
        if candidate_entities is None:
            return CutData(_cpp.cut(level_sets[0]._cpp_object))
        return CutData(
            _cpp.cut_entities(level_sets[0]._cpp_object, candidate_entities, dim)
        )
    if candidate_entities is None:
        return CutData(_cpp.cut_multi([phi._cpp_object for phi in level_sets]))
    return CutData(
        _cpp.cut_multi_entities(
            [phi._cpp_object for phi in level_sets], candidate_entities, dim
        )
    )


def update(cut_data: CutData) -> None:
    """Refresh cut data from the current level-set values."""
    _cpp.update_cut(cut_data._cpp_object)


def locate_entities(
    cut_data: CutData,
    ls_part: str,
) -> npt.NDArray[np.int32]:
    """Locate background entities classified by a level-set selector."""
    return _cpp.locate_entities(cut_data._cpp_object, ls_part)


def create_cut_mesh(
    cut_data: CutData,
    ls_part: str,
    mode: str | None = None,
) -> CutMesh:
    """Create a cut visualisation mesh for the selected level-set part."""
    mode_arg = "auto" if mode is None else mode
    return CutMesh(_cpp.create_cut_mesh(cut_data._cpp_object, ls_part, mode_arg))


def runtime_quadrature(
    cut_data: CutData,
    ls_part: str,
    order: int,
):
    """Create runintgen-compatible runtime quadrature for selected entities."""
    cpp_rules = _cpp.runtime_quadrature(cut_data._cpp_object, ls_part, order)

    rules = RuntimeQuadratureRules(
        kind=cpp_rules.kind,
        tdim=cpp_rules.tdim,
        points=cpp_rules.points,
        weights=cpp_rules.weights,
        offsets=cpp_rules.offsets,
        parent_map=cpp_rules.parent_map,
    )
    rules._cutfemx_owner = cpp_rules
    return rules
