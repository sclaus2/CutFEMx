# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

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


class MeshCutter:
    """Python handle for the C++ CutFEMx mesh cutter."""

    def __init__(self, cpp_object):
        self._cpp_object = cpp_object

    def update(self) -> None:
        """Refresh the current cut state from the held level-set values."""
        self._cpp_object.update()

    def locate_entities(
        self,
        ls_part: str,
        entities: npt.ArrayLike | None = None,
        entity_dim: int | None = None,
    ) -> npt.NDArray[np.int32]:
        """Locate background entities using the mesh-cutter API."""
        if entities is None:
            if entity_dim is not None:
                raise ValueError("entity_dim is only valid when entities are supplied")
            return self._cpp_object.locate_entities(ls_part)

        if entity_dim is None:
            raise ValueError("entity_dim must be supplied when entities are supplied")

        candidate_entities = np.ascontiguousarray(np.asarray(entities, dtype=np.int32))
        return self._cpp_object.locate_entities_subset(
            ls_part, candidate_entities, entity_dim
        )

    def create_cut_mesh(
        self,
        ls_part: str,
        entities: npt.ArrayLike | None = None,
        entity_dim: int | None = None,
        mode: str = "full",
    ) -> CutMesh:
        """Create a cut visualisation mesh for the selected level-set part."""
        if entities is None:
            if entity_dim is not None:
                raise ValueError("entity_dim is only valid when entities are supplied")
            return CutMesh(self._cpp_object.create_cut_mesh(ls_part, mode))

        if entity_dim is None:
            raise ValueError("entity_dim must be supplied when entities are supplied")

        candidate_entities = np.ascontiguousarray(np.asarray(entities, dtype=np.int32))
        return CutMesh(
            self._cpp_object.create_cut_mesh_subset(
                ls_part, candidate_entities, entity_dim, mode
            )
        )

    def runtime_quadrature(
        self,
        ls_part: str,
        order: int,
        entities: npt.ArrayLike | None = None,
        entity_dim: int | None = None,
    ):
        """Create runintgen-compatible runtime quadrature for cut entities."""
        if entities is None:
            if entity_dim is not None:
                raise ValueError("entity_dim is only valid when entities are supplied")
            cpp_rules = self._cpp_object.runtime_quadrature(ls_part, order)
        else:
            if entity_dim is None:
                raise ValueError("entity_dim must be supplied when entities are supplied")
            candidate_entities = np.ascontiguousarray(
                np.asarray(entities, dtype=np.int32)
            )
            cpp_rules = self._cpp_object.runtime_quadrature_subset(
                ls_part, order, candidate_entities, entity_dim
            )

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


def mesh_cutter(level_set: Function) -> MeshCutter:
    """Create a persistent mesh cutter for a scalar level-set function."""
    return MeshCutter(_cpp.mesh_cutter(level_set._cpp_object))
