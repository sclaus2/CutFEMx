# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from runintgen import QuadratureRules

import basix.ufl
import ufl
from cutfemx import cutfemx_cpp as _cpp
from dolfinx.fem import Function
from dolfinx.mesh import Mesh


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

    def with_physical_points(self) -> RuntimeQuadratureRules:
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

    def __init__(
        self,
        cpp_object,
        *,
        level_sets: Sequence[Function] | None = None,
        entities: npt.NDArray[np.int32] | None = None,
        entity_dim: int | None = None,
    ):
        self._cpp_object = cpp_object
        self._level_sets = tuple(level_sets or ())
        self._entities = None if entities is None else np.asarray(entities, dtype=np.int32)
        self._entity_dim = entity_dim

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

    @property
    def level_sets(self) -> tuple[Function, ...]:
        return self._level_sets

    @property
    def mesh(self) -> Mesh:
        if not self._level_sets:
            raise RuntimeError("CutData does not retain a Python level-set owner.")
        return self._level_sets[0].function_space.mesh

    @property
    def entity_dim(self) -> int | None:
        return self._entity_dim

    @property
    def entities(self) -> npt.NDArray[np.int32] | None:
        return self._entities


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
            return CutData(_cpp.cut(level_sets[0]._cpp_object), level_sets=level_sets)
        return CutData(
            _cpp.cut_entities(level_sets[0]._cpp_object, candidate_entities, dim),
            level_sets=level_sets,
            entities=candidate_entities,
            entity_dim=dim,
        )
    if candidate_entities is None:
        return CutData(
            _cpp.cut_multi([phi._cpp_object for phi in level_sets]),
            level_sets=level_sets,
        )
    return CutData(
        _cpp.cut_multi_entities(
            [phi._cpp_object for phi in level_sets], candidate_entities, dim
        ),
        level_sets=level_sets,
        entities=candidate_entities,
        entity_dim=dim,
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


def interior_facets_for_cells(
    msh: Mesh,
    cells: npt.ArrayLike,
    *,
    include_ghosts: bool = False,
) -> npt.NDArray[np.int32]:
    """Return raw local interior facet ids touched by ``cells``."""
    local_cells = np.ascontiguousarray(np.asarray(cells, dtype=np.int32).ravel())
    dtype = np.dtype(msh.geometry.x.dtype)
    if dtype == np.dtype(np.float64):
        return _cpp.interior_facets_for_cells_float64(
            msh._cpp_object, local_cells, include_ghosts
        )
    if dtype == np.dtype(np.float32):
        return _cpp.interior_facets_for_cells_float32(
            msh._cpp_object, local_cells, include_ghosts
        )
    raise ValueError(f"Unsupported mesh geometry dtype {dtype}.")


def ghost_penalty_facets(
    cut_data: CutData,
    selector: str,
    *,
    depth: int = 1,
    include_ghosts: bool = False,
) -> npt.NDArray[np.int32]:
    """Return owned raw interior facet ids for the cut-cell stabilization band."""
    if depth != 1:
        raise NotImplementedError("ghost_penalty_facets currently supports depth=1.")
    if cut_data.entity_dim is not None and cut_data.entity_dim != cut_data.mesh.topology.dim:
        raise ValueError("ghost_penalty_facets expects cell-hosted CutData.")

    msh = cut_data.mesh
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    cell_to_facet = msh.topology.connectivity(tdim, fdim)
    facet_to_cell = msh.topology.connectivity(fdim, tdim)
    if cell_to_facet is None or facet_to_cell is None:
        raise RuntimeError("Facet-cell connectivity is unavailable.")

    cut_cells = locate_entities(cut_data, "phi=0")
    selected_cells = locate_entities(cut_data, selector)
    active_cells = set(np.concatenate([cut_cells, selected_cells]).tolist())
    num_owned_facets = msh.topology.index_map(fdim).size_local

    facets: set[int] = set()
    for cell in cut_cells:
        for facet in cell_to_facet.links(int(cell)):
            facet = int(facet)
            if not include_ghosts and facet >= num_owned_facets:
                continue
            adjacent = [int(c) for c in facet_to_cell.links(facet)]
            if len(adjacent) != 2:
                continue
            if all(c in active_cells for c in adjacent):
                facets.add(facet)
    return np.asarray(sorted(facets), dtype=np.int32)
