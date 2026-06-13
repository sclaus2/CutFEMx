"""Generate PyVista screenshots for the tutorial pages.

The scenes use the same DOLFINx mesh construction and CutFEMx classification
steps as the demos. 2D setup views use a top-down camera; final 3D or warped
solution views use an angled camera.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import TYPE_CHECKING

from mpi4py import MPI

import cutfemx
import numpy as np
import pyvista as pv
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

import basix.ufl
import ufl
from dolfinx import default_scalar_type, fem, la, mesh

if TYPE_CHECKING:
    from collections.abc import Callable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "_static" / "tutorials"

pv.OFF_SCREEN = True

BACKGROUND = "#ffffff"
MESH_EDGE = "#334155"
INTERIOR = "#68b984"
CUT = "#f59e42"
INTERFACE = "#0f766e"
QUADRATURE = "#2563eb"
SURFACE_QUADRATURE = "#be185d"
SKELETON = "#ea580c"
GHOST = "#dc2626"
POISSON_SOLUTION_WARP_FACTOR = 0.45
INTERFACE_POISSON_SOLUTION_WARP_FACTOR = 1.0
ELASTICITY_DEFORMATION_SCALE = 1.0
_STL_DISTANCE_STATE: dict | None = None


def _circle_level_set(center: tuple[float, float], radius: float) -> Callable[[np.ndarray], np.ndarray]:
    def phi(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius

    return phi


def _flower_level_set(
    center: tuple[float, float],
    base_radius: float,
    amplitude: float,
    petals: int,
) -> Callable[[np.ndarray], np.ndarray]:
    def phi(x: np.ndarray) -> np.ndarray:
        dx = x[0] - center[0]
        dy = x[1] - center[1]
        theta = np.arctan2(dy, dx)
        boundary_radius = base_radius + amplitude * np.cos(petals * theta)
        return np.sqrt(dx**2 + dy**2) - boundary_radius

    return phi


def _sphere_level_set(center: tuple[float, float, float], radius: float) -> Callable[[np.ndarray], np.ndarray]:
    def phi(x: np.ndarray) -> np.ndarray:
        return (
            (x[0] - center[0]) ** 2
            + (x[1] - center[1]) ** 2
            + (x[2] - center[2]) ** 2
            - radius**2
        )

    return phi


def _dogbone_level_set(
    half_length: float,
    grip_radius: float,
    waist_radius: float,
    gauge_half_length: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def phi(x: np.ndarray) -> np.ndarray:
        t = (np.abs(x[0]) - gauge_half_length) / (half_length - gauge_half_length)
        t = np.clip(t, 0.0, 1.0)
        shoulder = t * t * (3.0 - 2.0 * t)
        radius = waist_radius + (grip_radius - waist_radius) * shoulder
        return np.sqrt(x[1] ** 2 + x[2] ** 2) - radius

    return phi


def _mesh_grid(msh: mesh.Mesh, cell_ids: np.ndarray | None = None) -> pv.UnstructuredGrid:
    points = np.asarray(msh.geometry.x, dtype=np.float64)
    dofmap = np.asarray(msh.geometry.dofmaps[0], dtype=np.int64)
    if cell_ids is not None:
        dofmap = dofmap[np.asarray(cell_ids, dtype=np.int32)]

    cell_name = msh.topology.cell_name()
    cell_types = {
        "interval": pv.CellType.LINE,
        "triangle": pv.CellType.TRIANGLE,
        "tetrahedron": pv.CellType.TETRA,
        "hexahedron": pv.CellType.HEXAHEDRON,
    }
    if cell_name not in cell_types:
        raise ValueError(f"Unsupported mesh cell type: {cell_name}")

    cells = np.hstack((np.full((dofmap.shape[0], 1), dofmap.shape[1], dtype=np.int64), dofmap))
    return pv.UnstructuredGrid(
        cells.ravel(),
        np.full(dofmap.shape[0], cell_types[cell_name], dtype=np.uint8),
        points,
    )


def _surface(dataset: pv.DataSet) -> pv.PolyData:
    return dataset.extract_surface(algorithm="dataset_surface")


def _facet_polydata(msh: mesh.Mesh, facet_ids: np.ndarray) -> pv.PolyData | None:
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    if facet_ids.size == 0:
        return None

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, 0)
    connectivity = msh.topology.connectivity(fdim, 0)
    points = np.asarray(msh.geometry.x, dtype=np.float64)
    entities = [np.asarray(connectivity.links(int(facet)), dtype=np.int64) for facet in facet_ids]
    if not entities:
        return None

    if all(entity.size == 2 for entity in entities):
        used = np.unique(np.concatenate(entities))
        remap = {int(old): new for new, old in enumerate(used)}
        local_points = points[used]
        poly = pv.PolyData(local_points)
        lines = np.asarray(
            [value for entity in entities for value in (2, remap[int(entity[0])], remap[int(entity[1])])],
            dtype=np.int64,
        )
        poly.lines = lines
        return poly

    used = np.unique(np.concatenate(entities))
    remap = {int(old): new for new, old in enumerate(used)}
    local_points = points[used]
    faces = np.asarray(
        [value for entity in entities for value in (entity.size, *[remap[int(v)] for v in entity])],
        dtype=np.int64,
    )
    return pv.PolyData(local_points, faces=faces)


def _facet_band_polydata(
    msh: mesh.Mesh,
    facet_ids: np.ndarray,
    *,
    width: float,
    z: float = 0.06,
) -> pv.PolyData | None:
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    if facet_ids.size == 0:
        return None

    fdim = msh.topology.dim - 1
    if fdim != 1:
        return None

    msh.topology.create_connectivity(fdim, 0)
    connectivity = msh.topology.connectivity(fdim, 0)
    mesh_points = np.asarray(msh.geometry.x, dtype=np.float64)
    band_points: list[np.ndarray] = []
    faces: list[int] = []
    half_width = 0.5 * width

    for facet in facet_ids:
        vertices = np.asarray(connectivity.links(int(facet)), dtype=np.int64)
        if vertices.size != 2:
            continue

        p0 = mesh_points[vertices[0]].copy()
        p1 = mesh_points[vertices[1]].copy()
        direction = p1[:2] - p0[:2]
        length = np.linalg.norm(direction)
        if length == 0.0:
            continue

        normal = half_width * np.array([-direction[1], direction[0]], dtype=np.float64) / length
        base = len(band_points)
        q0 = np.array([p0[0] + normal[0], p0[1] + normal[1], z], dtype=np.float64)
        q1 = np.array([p1[0] + normal[0], p1[1] + normal[1], z], dtype=np.float64)
        q2 = np.array([p1[0] - normal[0], p1[1] - normal[1], z], dtype=np.float64)
        q3 = np.array([p0[0] - normal[0], p0[1] - normal[1], z], dtype=np.float64)
        band_points.extend((q0, q1, q2, q3))
        faces.extend((4, base, base + 1, base + 2, base + 3))

    if not band_points:
        return None
    return pv.PolyData(np.asarray(band_points, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64))


def _facet_segments(msh: mesh.Mesh, facet_ids: np.ndarray) -> list[list[list[float]]]:
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    if facet_ids.size == 0:
        return []

    fdim = msh.topology.dim - 1
    if fdim != 1:
        return []

    msh.topology.create_connectivity(fdim, 0)
    connectivity = msh.topology.connectivity(fdim, 0)
    mesh_points = np.asarray(msh.geometry.x, dtype=np.float64)
    segments: list[list[list[float]]] = []
    for facet in facet_ids:
        vertices = np.asarray(connectivity.links(int(facet)), dtype=np.int64)
        if vertices.size != 2:
            continue
        p0, p1 = mesh_points[vertices, :2]
        segments.append([p0.astype(float).tolist(), p1.astype(float).tolist()])
    return segments


def _clipped_facet_segments(
    msh: mesh.Mesh,
    facet_ids: np.ndarray,
    levelset: Callable[[np.ndarray], np.ndarray],
    *,
    eps: float = 1.0e-12,
) -> list[list[list[float]]]:
    facet_ids = np.asarray(facet_ids, dtype=np.int32)
    if facet_ids.size == 0:
        return []

    fdim = msh.topology.dim - 1
    if fdim != 1:
        return []

    msh.topology.create_connectivity(fdim, 0)
    connectivity = msh.topology.connectivity(fdim, 0)
    mesh_points = np.asarray(msh.geometry.x, dtype=np.float64)
    segments: list[list[list[float]]] = []

    for facet in facet_ids:
        vertices = np.asarray(connectivity.links(int(facet)), dtype=np.int64)
        if vertices.size != 2:
            continue

        p0, p1 = mesh_points[vertices, :2]
        values = np.asarray(levelset(mesh_points[vertices].T), dtype=np.float64)
        phi0, phi1 = float(values[0]), float(values[1])
        inside0 = phi0 < -eps
        inside1 = phi1 < -eps
        on0 = abs(phi0) <= eps
        on1 = abs(phi1) <= eps

        if (inside0 or on0) and (inside1 or on1):
            segment = [p0, p1]
        elif inside0 or on0:
            if abs(phi0 - phi1) <= eps:
                continue
            t = phi0 / (phi0 - phi1)
            segment = [p0, p0 + t * (p1 - p0)]
        elif inside1 or on1:
            if abs(phi1 - phi0) <= eps:
                continue
            t = phi1 / (phi1 - phi0)
            segment = [p1, p1 + t * (p0 - p1)]
        else:
            continue

        if np.linalg.norm(segment[1] - segment[0]) <= eps:
            continue
        segments.append([segment[0].astype(float).tolist(), segment[1].astype(float).tolist()])

    return segments


def _segments_polydata(
    segments: list[list[list[float]]],
    *,
    z: float = 0.07,
) -> pv.PolyData | None:
    if not segments:
        return None

    points: list[list[float]] = []
    lines: list[int] = []
    for segment in segments:
        base = len(points)
        p0, p1 = segment
        points.append([float(p0[0]), float(p0[1]), z])
        points.append([float(p1[0]), float(p1[1]), z])
        lines.extend((2, base, base + 1))

    poly = pv.PolyData(np.asarray(points, dtype=np.float64))
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def _quadrature_points(rules) -> np.ndarray:
    points = np.asarray(rules.physical_points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim == 2 and points.shape[0] in (2, 3):
        points = points.T
    if points.shape[1] == 2:
        points = np.column_stack((points, np.full(points.shape[0], 0.02)))
    return points


def _solve_runtime_system(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
    *,
    name: str,
    bcs: list[fem.DirichletBC] | None = None,
) -> tuple[fem.Function, cutfemx.fem.ActiveDomain]:
    if V.mesh.comm.size != 1:
        raise RuntimeError("Tutorial scene solves are serial-only.")

    bcs = [] if bcs is None else bcs
    A = cutfemx.fem.assemble_matrix(a, bcs=bcs)
    A.scatter_reverse()
    b = cutfemx.fem.assemble_vector(L)
    if bcs:
        cutfemx.fem.apply_lifting(b.array, [a], [bcs])
    b.scatter_reverse(la.InsertMode.add)
    if bcs:
        fem.set_bc(b.array, bcs)

    active = cutfemx.fem.active_domain(a)
    cutfemx.fem.deactivate_outside(A, b, active)

    uh = fem.Function(V, name=name)
    uh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    uh.x.scatter_forward()
    return uh, active


def _function_grid(u: fem.Function, scalar_name: str = "u") -> pv.UnstructuredGrid:
    from dolfinx import plot

    cells, types, coordinates = plot.vtk_mesh(u.function_space)
    grid = pv.UnstructuredGrid(cells, types, coordinates)
    values = np.asarray(u.x.array)
    if np.iscomplexobj(values):
        values = values.real
    values = np.asarray(values, dtype=np.float64)
    if values.size == grid.n_points:
        grid.point_data[scalar_name] = values
    elif values.size == grid.n_cells:
        grid.cell_data[scalar_name] = values
    elif values.size > grid.n_points:
        grid.point_data[scalar_name] = values[: grid.n_points]
    else:
        raise RuntimeError(
            f"Cannot attach {values.size} values to grid with "
            f"{grid.n_points} points and {grid.n_cells} cells."
        )
    grid.set_active_scalars(scalar_name)
    return grid


def _cut_scalar_function_grid(
    cut_data,
    selector: str,
    u: fem.Function,
    *,
    mode: str = "full",
    scalar_name: str = "u",
) -> pv.UnstructuredGrid:
    cut_mesh = cutfemx.create_cut_mesh(cut_data, selector, mode=mode)
    if cut_mesh.mesh is None:
        raise RuntimeError(f"Cannot plot empty cut mesh for selector {selector!r}.")
    return _function_grid(cutfemx.fem.cut_function(u, cut_mesh), scalar_name)


def _attach_vector_point_data(
    grid: pv.UnstructuredGrid,
    vector_values: np.ndarray,
    *,
    vector_name: str,
    magnitude_name: str,
) -> None:
    values = np.asarray(vector_values, dtype=np.float64)
    if values.shape[0] < grid.n_points:
        raise RuntimeError(
            f"Cannot attach {values.shape[0]} vectors to a grid with {grid.n_points} points."
        )
    point_vectors = values[: grid.n_points]
    grid.point_data[vector_name] = point_vectors
    grid.point_data[magnitude_name] = np.linalg.norm(point_vectors, axis=1)
    grid.set_active_scalars(magnitude_name)


def _mesh_triangles(msh: mesh.Mesh, cell_ids: np.ndarray | None = None) -> list[list[int]]:
    dofmap = np.asarray(msh.geometry.dofmaps[0], dtype=np.int64)
    if cell_ids is not None:
        dofmap = dofmap[np.asarray(cell_ids, dtype=np.int32)]
    if dofmap.shape[1] != 3:
        return []
    return dofmap.astype(int).tolist()


def _grid_triangles(grid: pv.UnstructuredGrid) -> list[list[int]]:
    cells = np.asarray(grid.cells, dtype=np.int64)
    triangles: list[list[int]] = []
    offset = 0
    while offset < cells.size:
        nvertices = int(cells[offset])
        ids = cells[offset + 1 : offset + 1 + nvertices]
        if nvertices == 3:
            triangles.append(ids.astype(int).tolist())
        offset += nvertices + 1
    return triangles


def _edges_from_triangles(triangles: list[list[int]]) -> list[list[int]]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in triangles:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return [list(edge) for edge in sorted(edges)]


def _boundary_edges_from_triangles(triangles: list[list[int]]) -> list[list[int]]:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in triangles:
        for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            counts[edge] = counts.get(edge, 0) + 1
    return [list(edge) for edge, count in sorted(counts.items()) if count == 1]


def _merge_duplicate_triangle_points(
    points: np.ndarray, triangles: list[list[int]], *, tol: float = 1.0e-12
) -> tuple[np.ndarray, list[list[int]]]:
    key_to_new: dict[tuple[int, ...], int] = {}
    old_to_new: dict[int, int] = {}
    merged_points: list[np.ndarray] = []
    for old_id, point in enumerate(points):
        key = tuple(np.rint(point / tol).astype(np.int64).tolist())
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(merged_points)
            key_to_new[key] = new_id
            merged_points.append(point)
        old_to_new[old_id] = new_id
    merged_triangles = [[old_to_new[idx] for idx in triangle] for triangle in triangles]
    return np.asarray(merged_points, dtype=np.float64), merged_triangles


def _warped_surface_view_data(
    grids: list[pv.UnstructuredGrid], *, scalar: str = "u", warp_factor: float = 1.0
) -> dict:
    point_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []
    triangles: list[list[int]] = []
    offset = 0
    for grid in grids:
        points = np.asarray(grid.points, dtype=np.float64).copy()
        values = np.asarray(grid[scalar], dtype=np.float64)
        points[:, 2] = warp_factor * values
        point_blocks.append(points)
        value_blocks.append(values)
        for triangle in _grid_triangles(grid):
            triangles.append([idx + offset for idx in triangle])
        offset += points.shape[0]

    if not point_blocks or not triangles:
        return {"bounds": [0, 1, 0, 1, 0, 1], "points": [], "cells": [], "boundaryEdges": []}

    points = np.vstack(point_blocks)
    values = np.concatenate(value_blocks)
    interactive_points, interactive_triangles = _merge_duplicate_triangle_points(points, triangles)
    vmin = float(values.min())
    vmax = float(values.max())
    cells = [
        {
            "ids": remapped_triangle,
            "color": _viridis_color(float(values[original_triangle].mean()), vmin, vmax),
        }
        for original_triangle, remapped_triangle in zip(triangles, interactive_triangles)
    ]
    return {
        "bounds": [
            float(interactive_points[:, 0].min()),
            float(interactive_points[:, 0].max()),
            float(interactive_points[:, 1].min()),
            float(interactive_points[:, 1].max()),
            float(interactive_points[:, 2].min()),
            float(interactive_points[:, 2].max()),
        ],
        "points": interactive_points.tolist(),
        "cells": cells,
        "boundaryEdges": _boundary_edges_from_triangles(interactive_triangles),
    }


def _as_2d(points: np.ndarray) -> list[list[float]]:
    if points.size == 0:
        return []
    return np.asarray(points[:, :2], dtype=float).tolist()


def _plot_bounds(points: np.ndarray, *, padding: float = 0.06) -> list[float]:
    xy = np.asarray(points[:, :2], dtype=np.float64)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    width = xmax - xmin
    height = ymax - ymin
    pad = padding * max(width, height)
    return [float(xmin - pad), float(xmax + pad), float(ymin - pad), float(ymax + pad)]


def _circle_points(center: tuple[float, float], radius: float) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    return np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta))).tolist()


def _flower_points(
    center: tuple[float, float],
    base_radius: float,
    amplitude: float,
    petals: int,
) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    radius = base_radius + amplitude * np.cos(petals * theta)
    return np.column_stack((center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta))).tolist()


def _hex_from_rgb(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _viridis_color(value: float, vmin: float, vmax: float) -> str:
    stops = [
        (0.00, (68, 1, 84)),
        (0.13, (71, 44, 122)),
        (0.25, (59, 81, 139)),
        (0.38, (44, 113, 142)),
        (0.50, (33, 144, 141)),
        (0.63, (39, 173, 129)),
        (0.75, (92, 200, 99)),
        (0.88, (170, 220, 50)),
        (1.00, (253, 231, 37)),
    ]
    if vmax <= vmin:
        t = 0.5
    else:
        t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    for (left_t, left_color), (right_t, right_color) in zip(stops[:-1], stops[1:]):
        if left_t <= t <= right_t:
            alpha = (t - left_t) / (right_t - left_t)
            rgb = tuple(
                int(round((1.0 - alpha) * left_color[i] + alpha * right_color[i]))
                for i in range(3)
            )
            return _hex_from_rgb(rgb)
    return _hex_from_rgb(stops[-1][1])


def _write_interactive_2d(path: Path, title: str, data: dict) -> None:
    data_json = json.dumps(data, separators=(",", ":"))
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      background: #ffffff;
      overflow: hidden;
    }
    .toolbar {
      position: absolute;
      top: 10px;
      right: 12px;
      z-index: 2;
      display: flex;
      gap: 8px;
    }
    button {
      border: 1px solid #cbd5e1;
      background: rgba(255, 255, 255, 0.92);
      color: #334155;
      border-radius: 4px;
      padding: 6px 10px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
    }
    button:hover {
      background: #f8fafc;
    }
    svg {
      width: 100%;
      height: 100%;
      display: block;
      touch-action: none;
      cursor: grab;
      background: #ffffff;
    }
    svg.dragging {
      cursor: grabbing;
    }
    .mesh-edge, .cut-edge, .interface, .line-layer {
      vector-effect: non-scaling-stroke;
      fill: none;
    }
    .mesh-edge {
      stroke: #334155;
      stroke-width: 0.8;
      opacity: 0.92;
    }
    .cut-cell {
      fill: #f59e42;
      fill-opacity: 0.24;
      stroke: #7c2d12;
      stroke-width: 0.7;
      vector-effect: non-scaling-stroke;
    }
    .solution-cell {
      stroke: #334155;
      stroke-width: 0.55;
      vector-effect: non-scaling-stroke;
    }
    .interface {
      stroke: #0f766e;
      stroke-width: 2.5;
    }
    .q-volume {
      fill: #2563eb;
      opacity: 0.95;
    }
    .q-interface {
      fill: #be185d;
      opacity: 0.95;
    }
    .point-layer {
      opacity: 0.95;
    }
  </style>
</head>
<body>
  <div class="toolbar"><button type="button" id="reset">Reset</button></div>
  <svg id="plot" role="img" aria-label="__TITLE__"></svg>
  <script>
    const data = __DATA__;
    const svg = document.getElementById("plot");
    const reset = document.getElementById("reset");
    const ns = "http://www.w3.org/2000/svg";
    const initial = {
      xmin: data.bounds[0],
      xmax: data.bounds[1],
      ymin: data.bounds[2],
      ymax: data.bounds[3],
    };
    const view = { ...initial };

    function svgPoint(clientX, clientY) {
      const point = svg.createSVGPoint();
      point.x = clientX;
      point.y = clientY;
      const transformed = point.matrixTransform(svg.getScreenCTM().inverse());
      return [transformed.x, -transformed.y];
    }

    function setViewBox() {
      svg.setAttribute(
        "viewBox",
        `${view.xmin} ${-view.ymax} ${view.xmax - view.xmin} ${view.ymax - view.ymin}`
      );
    }

    function pointString(points, ids) {
      return ids.map((id) => `${points[id][0]},${-points[id][1]}`).join(" ");
    }

    function pathFromEdges(points, edges) {
      return edges.map(([a, b]) => `M${points[a][0]},${-points[a][1]}L${points[b][0]},${-points[b][1]}`).join("");
    }

    function pathFromSegments(segments) {
      return segments.map(([[x0, y0], [x1, y1]]) => `M${x0},${-y0}L${x1},${-y1}`).join("");
    }

    function pathFromPolyline(points) {
      return points.map((p, index) => `${index === 0 ? "M" : "L"}${p[0]},${-p[1]}`).join("") + "Z";
    }

    function add(tag, attrs, parent = svg) {
      const el = document.createElementNS(ns, tag);
      for (const [key, value] of Object.entries(attrs)) {
        el.setAttribute(key, value);
      }
      parent.appendChild(el);
      return el;
    }

    function render() {
      setViewBox();

      if (data.solutionCells) {
        for (const cell of data.solutionCells) {
          add("polygon", {
            points: pointString(data.solutionPoints, cell.ids),
            fill: cell.color,
            class: "solution-cell",
          });
        }
      }

      if (data.cutCells) {
        for (const ids of data.cutCells.triangles) {
          add("polygon", {
            points: pointString(data.cutCells.points, ids),
            class: "cut-cell",
          });
        }
      }

      add("path", {
        d: pathFromEdges(data.background.points, data.background.edges),
        class: "mesh-edge",
      });

      if (data.lineLayers) {
        for (const layer of data.lineLayers) {
          const attrs = {
            d: pathFromSegments(layer.segments),
            class: "line-layer",
            stroke: layer.color,
            "stroke-width": layer.width || 1.5,
            opacity: layer.opacity == null ? 1.0 : layer.opacity,
          };
          if (layer.dash) {
            attrs["stroke-dasharray"] = layer.dash;
          }
          add("path", attrs);
        }
      }

      if (data.interface) {
        add("path", {
          d: pathFromPolyline(data.interface),
          class: "interface",
        });
      }

      if (data.volumePoints) {
        for (const p of data.volumePoints) {
          add("circle", { cx: p[0], cy: -p[1], r: data.pointRadius, class: "q-volume" });
        }
      }
      if (data.interfacePoints) {
        for (const p of data.interfacePoints) {
          add("circle", { cx: p[0], cy: -p[1], r: data.interfacePointRadius, class: "q-interface" });
        }
      }
      if (data.pointLayers) {
        for (const layer of data.pointLayers) {
          for (const p of layer.points) {
            add("circle", {
              cx: p[0],
              cy: -p[1],
              r: layer.radius || data.pointRadius,
              class: "point-layer",
              fill: layer.color,
              opacity: layer.opacity == null ? 0.95 : layer.opacity,
            });
          }
        }
      }
    }

    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const [x, y] = svgPoint(event.clientX, event.clientY);
      const factor = event.deltaY < 0 ? 0.82 : 1.22;
      view.xmin = x + (view.xmin - x) * factor;
      view.xmax = x + (view.xmax - x) * factor;
      view.ymin = y + (view.ymin - y) * factor;
      view.ymax = y + (view.ymax - y) * factor;
      setViewBox();
    }, { passive: false });

    let drag = null;
    svg.addEventListener("pointerdown", (event) => {
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("dragging");
      drag = { start: svgPoint(event.clientX, event.clientY), view: { ...view } };
    });
    svg.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const [x, y] = svgPoint(event.clientX, event.clientY);
      const dx = drag.start[0] - x;
      const dy = drag.start[1] - y;
      view.xmin = drag.view.xmin + dx;
      view.xmax = drag.view.xmax + dx;
      view.ymin = drag.view.ymin + dy;
      view.ymax = drag.view.ymax + dy;
      setViewBox();
    });
    svg.addEventListener("pointerup", () => {
      drag = null;
      svg.classList.remove("dragging");
    });
    svg.addEventListener("pointercancel", () => {
      drag = null;
      svg.classList.remove("dragging");
    });
    reset.addEventListener("click", () => {
      Object.assign(view, initial);
      setViewBox();
    });

    render();
  </script>
</body>
</html>
"""
    html = template.replace("__TITLE__", title).replace("__DATA__", data_json)
    path.write_text(html, encoding="utf-8")


def _write_interactive_warped_surface(path: Path, title: str, data: dict) -> None:
    data_json = json.dumps(data, separators=(",", ":"))
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #ffffff;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      touch-action: none;
      background: #ffffff;
    }
    canvas.dragging {
      cursor: grabbing;
    }
    .toolbar {
      position: absolute;
      top: 10px;
      right: 12px;
      z-index: 2;
      display: flex;
      gap: 8px;
    }
    button {
      border: 1px solid #cbd5e1;
      background: rgba(255, 255, 255, 0.92);
      color: #334155;
      border-radius: 4px;
      padding: 6px 10px;
      font: inherit;
      cursor: pointer;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
    }
    button:hover {
      background: #f8fafc;
    }
  </style>
</head>
<body>
  <div class="toolbar"><button type="button" id="reset">Reset</button></div>
  <canvas id="plot" aria-label="__TITLE__"></canvas>
  <script>
    const data = __DATA__;
    const canvas = document.getElementById("plot");
    const reset = document.getElementById("reset");
    const ctx = canvas.getContext("2d");
    const initial = { yaw: -0.62, pitch: 0.78, zoom: 1.0 };
    const view = { ...initial };
    const bounds = data.bounds;
    const center = [
      0.5 * (bounds[0] + bounds[1]),
      0.5 * (bounds[2] + bounds[3]),
      0.5 * (bounds[4] + bounds[5]),
    ];
    const span = Math.max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]);
    const shadeStrength = data.shadeStrength ?? 0.0;
    let dragging = null;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render();
    }

    function hexToRgb(hex) {
      const value = hex.replace("#", "");
      return [
        parseInt(value.slice(0, 2), 16),
        parseInt(value.slice(2, 4), 16),
        parseInt(value.slice(4, 6), 16),
      ];
    }

    function shaded(hex, shade) {
      const rgb = hexToRgb(hex);
      return `rgb(${rgb.map((v) => Math.max(0, Math.min(255, Math.round(v * shade)))).join(",")})`;
    }

    function rotatePoint(point) {
      const x0 = point[0] - center[0];
      const y0 = point[1] - center[1];
      const z0 = point[2] - center[2];
      const cy = Math.cos(view.yaw);
      const sy = Math.sin(view.yaw);
      const cp = Math.cos(view.pitch);
      const sp = Math.sin(view.pitch);
      const x1 = cy * x0 - sy * y0;
      const y1 = sy * x0 + cy * y0;
      const z1 = z0;
      return [x1, cp * y1 - sp * z1, sp * y1 + cp * z1];
    }

    function project(point, scale, width, height) {
      return [0.5 * width + scale * point[0], 0.53 * height - scale * point[1]];
    }

    function faceShade(p0, p1, p2) {
      const ux = p1[0] - p0[0];
      const uy = p1[1] - p0[1];
      const uz = p1[2] - p0[2];
      const vx = p2[0] - p0[0];
      const vy = p2[1] - p0[1];
      const vz = p2[2] - p0[2];
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      const length = Math.hypot(nx, ny, nz) || 1.0;
      const light = Math.max(0.0, (0.25 * nx - 0.35 * ny + 0.90 * nz) / length);
      return 0.78 + 0.30 * light;
    }

    function render() {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      const scale = view.zoom * Math.min(width, height) / (span * 1.34);
      const rotated = data.points.map(rotatePoint);
      const projected = rotated.map((p) => project(p, scale, width, height));
      const cells = data.cells.map((cell) => {
        const ids = cell.ids;
        return {
          ids,
          color: cell.color,
          depth: (rotated[ids[0]][2] + rotated[ids[1]][2] + rotated[ids[2]][2]) / 3.0,
          shade: shadeStrength > 0.0
            ? 1.0 + shadeStrength * (faceShade(rotated[ids[0]], rotated[ids[1]], rotated[ids[2]]) - 1.0)
            : 1.0,
        };
      }).sort((a, b) => a.depth - b.depth);

      for (const cell of cells) {
        const [a, b, c] = cell.ids;
        ctx.beginPath();
        ctx.moveTo(projected[a][0], projected[a][1]);
        ctx.lineTo(projected[b][0], projected[b][1]);
        ctx.lineTo(projected[c][0], projected[c][1]);
        ctx.closePath();
        ctx.fillStyle = shaded(cell.color, cell.shade);
        ctx.fill();
        ctx.strokeStyle = "rgba(51, 65, 85, 0.58)";
        ctx.lineWidth = 0.55;
        ctx.stroke();
      }

      ctx.strokeStyle = "rgba(15, 118, 110, 0.82)";
      ctx.lineWidth = 1.7;
      ctx.beginPath();
      for (const edge of data.boundaryEdges) {
        const a = projected[edge[0]];
        const b = projected[edge[1]];
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();

      if (data.paths) {
        for (const path of data.paths) {
          const pathPoints = path.points.map((point) => project(rotatePoint(point), scale, width, height));
          if (pathPoints.length < 2) continue;
          ctx.beginPath();
          ctx.moveTo(pathPoints[0][0], pathPoints[0][1]);
          for (const point of pathPoints.slice(1)) {
            ctx.lineTo(point[0], point[1]);
          }
          if (path.closed) {
            ctx.closePath();
          }
          ctx.strokeStyle = path.color || "#0f766e";
          ctx.lineWidth = path.width || 2.2;
          ctx.lineJoin = "round";
          ctx.lineCap = "round";
          ctx.stroke();
        }
      }
    }

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("dragging");
      dragging = { x: event.clientX, y: event.clientY, yaw: view.yaw, pitch: view.pitch };
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      view.yaw = dragging.yaw + 0.012 * (event.clientX - dragging.x);
      view.pitch = Math.max(-1.35, Math.min(1.35, dragging.pitch + 0.012 * (event.clientY - dragging.y)));
      render();
    });

    canvas.addEventListener("pointerup", () => {
      dragging = null;
      canvas.classList.remove("dragging");
    });

    canvas.addEventListener("pointercancel", () => {
      dragging = null;
      canvas.classList.remove("dragging");
    });

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      view.zoom *= event.deltaY < 0 ? 1.12 : 0.89;
      view.zoom = Math.max(0.42, Math.min(3.0, view.zoom));
      render();
    }, { passive: false });

    reset.addEventListener("click", () => {
      Object.assign(view, initial);
      render();
    });

    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    html = template.replace("__TITLE__", title).replace("__DATA__", data_json)
    path.write_text(html, encoding="utf-8")


def _closed_polyline(points_2d: list[list[float]], z: float = 0.03) -> pv.PolyData:
    points_2d_array = np.asarray(points_2d, dtype=np.float64)
    points = np.column_stack(
        (
            points_2d_array[:, 0],
            points_2d_array[:, 1],
            np.full(points_2d_array.shape[0], z, dtype=np.float64),
        )
    )
    poly = pv.PolyData(points)
    poly.lines = np.asarray([points.shape[0], *range(points.shape[0])], dtype=np.int64)
    return poly


def _circle_polyline(center: tuple[float, float], radius: float, z: float = 0.03) -> pv.PolyData:
    return _closed_polyline(_circle_points(center, radius), z=z)


def _box_edges(bounds: tuple[float, float, float, float, float, float]) -> pv.PolyData:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    points = np.asarray(
        [
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
        ],
        dtype=np.float64,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    poly = pv.PolyData(points)
    poly.lines = np.asarray([value for edge in edges for value in (2, edge[0], edge[1])], dtype=np.int64)
    return poly


def _add_background_mesh(
    plotter: pv.Plotter,
    msh: mesh.Mesh,
    cell_ids: np.ndarray | None = None,
    *,
    fill_color: str = "#e2e8f0",
    fill_opacity: float = 0.04,
    edge_color: str = MESH_EDGE,
    edge_opacity: float = 0.95,
    line_width: float = 0.9,
) -> None:
    grid = _mesh_grid(msh, cell_ids)
    plotter.add_mesh(grid, color=fill_color, show_edges=False, opacity=fill_opacity)
    plotter.add_mesh(grid, style="wireframe", color=edge_color, line_width=line_width, opacity=edge_opacity)


def _add_points(
    plotter: pv.Plotter,
    points: np.ndarray,
    *,
    color: str,
    point_size: float = 4.0,
    opacity: float = 1.0,
) -> None:
    if points.size == 0:
        return
    plotter.add_mesh(
        pv.PolyData(points),
        color=color,
        point_size=point_size,
        opacity=opacity,
        render_points_as_spheres=True,
    )


def _add_facet_mesh(
    plotter: pv.Plotter,
    msh: mesh.Mesh,
    facet_ids: np.ndarray,
    *,
    color: str,
    line_width: float = 4.0,
    opacity: float = 1.0,
) -> None:
    poly = _facet_polydata(msh, facet_ids)
    if poly is None:
        return
    plotter.add_mesh(poly, color=color, line_width=line_width, opacity=opacity)


def _add_segments(
    plotter: pv.Plotter,
    segments: list[list[list[float]]],
    *,
    color: str,
    line_width: float = 4.0,
    opacity: float = 1.0,
    z: float = 0.07,
) -> None:
    poly = _segments_polydata(segments, z=z)
    if poly is None:
        return
    plotter.add_mesh(poly, color=color, line_width=line_width, opacity=opacity)


def _add_facet_bands(
    plotter: pv.Plotter,
    msh: mesh.Mesh,
    facet_ids: np.ndarray,
    *,
    color: str,
    width: float,
    opacity: float = 1.0,
) -> None:
    poly = _facet_band_polydata(msh, facet_ids, width=width)
    if poly is None:
        _add_facet_mesh(plotter, msh, facet_ids, color=color, line_width=3.0, opacity=opacity)
        return
    plotter.add_mesh(poly, color=color, opacity=opacity, show_edges=False)


def _prepare_plotter(
    path: Path,
    title: str,
    draw: Callable[[pv.Plotter], None],
    *,
    flat: bool = True,
    camera_distance: float | None = None,
    camera_zoom: float | None = None,
    view: str | None = None,
) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 850))
    plotter.set_background(BACKGROUND)
    draw(plotter)
    plotter.add_title(title, font_size=12, color="#334155")
    plotter.add_axes(line_width=2, color="#334155")
    if view == "yz":
        plotter.view_yz()
        plotter.camera.parallel_projection = True
        plotter.camera.zoom(1.12 if camera_zoom is None else camera_zoom)
    elif flat:
        plotter.view_xy()
        plotter.camera.parallel_projection = True
        plotter.camera.zoom(1.12 if camera_zoom is None else camera_zoom)
    else:
        plotter.view_isometric()
        plotter.camera.parallel_projection = True
        if camera_distance is not None:
            plotter.camera.distance = camera_distance
        plotter.camera.zoom(1.15 if camera_zoom is None else camera_zoom)
    return plotter


def _export_scene(
    path: Path,
    title: str,
    draw: Callable[[pv.Plotter], None],
    *,
    flat: bool = True,
    camera_distance: float | None = None,
    camera_zoom: float | None = None,
    view: str | None = None,
) -> None:
    image_path = path.with_suffix(".png")
    plotter = _prepare_plotter(
        path,
        title,
        draw,
        flat=flat,
        camera_distance=camera_distance,
        camera_zoom=camera_zoom,
        view=view,
    )
    plotter.screenshot(image_path.as_posix())
    plotter.close()


def _poisson_state(
    *,
    n: int,
    center: tuple[float, float],
    radius: float,
    levelset_degree: int,
    order: int,
    levelset: Callable[[np.ndarray], np.ndarray] | None = None,
    interface_points: list[list[float]] | None = None,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((-1.0, -1.0), (1.0, 1.0)),
    cells: tuple[int, int] | None = None,
):
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        bounds,
        cells or (n, n),
        cell_type=mesh.CellType.triangle,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", levelset_degree))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(levelset or _circle_level_set(center, radius))
    phi.x.scatter_forward()

    cut_data = cutfemx.cut(phi)
    inside = cutfemx.locate_entities(cut_data, "phi<0")
    cut = cutfemx.locate_entities(cut_data, "phi=0")
    active = cutfemx.locate_entities(cut_data, "phi<=0")
    volume_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
    return {
        "mesh": msh,
        "phi": phi,
        "cut_data": cut_data,
        "inside": inside,
        "cut": cut,
        "active": active,
        "volume_rules": volume_rules,
        "interface_rules": interface_rules,
        "ghost_facets": ghost_facets,
        "center": center,
        "radius": radius,
        "interface_points": interface_points or _circle_points(center, radius),
    }


def _poisson_flower_state() -> dict:
    center = (0.0, 0.0)
    base_radius = 0.46
    amplitude = 0.15
    petals = 6
    return _poisson_state(
        n=32,
        center=center,
        radius=base_radius,
        levelset_degree=1,
        order=4,
        levelset=_flower_level_set(center, base_radius, amplitude, petals),
        interface_points=_flower_points(center, base_radius, amplitude, petals),
    )


def _add_2d_background(plotter: pv.Plotter, state, *, opacity: float = 0.18) -> None:
    _add_background_mesh(
        plotter,
        state["mesh"],
        fill_opacity=min(0.16, 0.30 * opacity),
        edge_opacity=0.92,
        line_width=0.88,
    )
    plotter.add_mesh(_closed_polyline(state["interface_points"]), color=INTERFACE, line_width=4)


def _add_cut_cell_classes(plotter: pv.Plotter, state) -> None:
    plotter.add_mesh(
        _mesh_grid(state["mesh"], state["inside"]),
        color=INTERIOR,
        show_edges=True,
        edge_color="#475569",
        line_width=0.45,
        opacity=0.62,
    )
    plotter.add_mesh(
        _mesh_grid(state["mesh"], state["cut"]),
        color=CUT,
        show_edges=True,
        edge_color="#7c2d12",
        line_width=0.6,
        opacity=0.82,
    )


def _solve_continuous_poisson(state: dict) -> fem.Function:
    uh = state.get("poisson_uh")
    if uh is not None:
        return uh

    msh = state["mesh"]
    phi = state["phi"]
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[state["inside"], state["volume_rules"]],
    )
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=state["interface_rules"])
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=state["ghost_facets"])

    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * u_exact
    gamma = 40.0
    gamma_g = 0.1

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += (
        -ufl.dot(ufl.grad(u), n_gamma) * v
        - ufl.dot(ufl.grad(v), n_gamma) * u
        + gamma / h * u * v
    ) * dx_gamma
    if state["ghost_facets"].size > 0:
        a += (
            gamma_g
            * h_avg
            * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
            * dS_ghost
        )

    L = f * v * dx_omega
    L += (-ufl.dot(ufl.grad(v), n_gamma) * u_exact + gamma / h * u_exact * v) * dx_gamma

    uh, _ = _solve_runtime_system(cutfemx.fem.form(a), cutfemx.fem.form(L), V, name="u_h")
    state["poisson_uh"] = uh
    return uh


def _continuous_poisson_solution_grid(state: dict) -> pv.UnstructuredGrid:
    return _cut_scalar_function_grid(state["cut_data"], "phi<0", _solve_continuous_poisson(state))


def poisson_scene() -> None:
    state = _poisson_flower_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state, opacity=0.12)
        plotter.add_mesh(
            _continuous_poisson_solution_grid(state),
            scalars="u",
            cmap="viridis",
            show_edges=True,
            edge_color=MESH_EDGE,
            line_width=0.45,
        )
        plotter.add_mesh(_closed_polyline(state["interface_points"], z=0.08), color=INTERFACE, line_width=4)

    _export_scene(
        OUTPUT_DIR / "poisson-scene.png",
        "Flat cut-mesh solution",
        draw,
    )


def poisson_background_scene() -> None:
    state = _poisson_flower_state()
    _export_scene(
        OUTPUT_DIR / "poisson-background-scene.png",
        "Background triangular mesh and level set",
        lambda plotter: _add_2d_background(plotter, state, opacity=0.42),
    )


def poisson_interior_scene() -> None:
    state = _poisson_flower_state()
    _export_scene(
        OUTPUT_DIR / "poisson-interior-cells-scene.png",
        "Interior cells",
        lambda plotter: (_add_2d_background(plotter, state), plotter.add_mesh(_mesh_grid(state["mesh"], state["inside"]), color=INTERIOR, show_edges=True, edge_color="#475569", opacity=0.72)),
    )


def poisson_cut_cells_scene() -> None:
    state = _poisson_flower_state()
    _export_scene(
        OUTPUT_DIR / "poisson-cut-cells-scene.png",
        "Cut cells",
        lambda plotter: (_add_2d_background(plotter, state), plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.86)),
    )


def poisson_quadrature_scene() -> None:
    state = _poisson_flower_state()
    mesh_points = np.asarray(state["mesh"].geometry.x, dtype=np.float64)
    mesh_triangles = _mesh_triangles(state["mesh"])
    cut_triangles = _mesh_triangles(state["mesh"], state["cut"])
    volume_points = _quadrature_points(state["volume_rules"])
    interface_points = _quadrature_points(state["interface_rules"])

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.28)
        _add_points(plotter, volume_points, color=QUADRATURE, point_size=3.4)
        _add_points(plotter, interface_points, color=SURFACE_QUADRATURE, point_size=4.4)

    _export_scene(OUTPUT_DIR / "poisson-quadrature-scene.png", "Runtime quadrature rules", draw)
    _write_interactive_2d(
        OUTPUT_DIR / "poisson-quadrature-view.html",
        "Interactive Poisson quadrature rules",
        {
            "bounds": _plot_bounds(mesh_points),
            "background": {
                "points": _as_2d(mesh_points),
                "edges": _edges_from_triangles(mesh_triangles),
            },
            "cutCells": {
                "points": _as_2d(mesh_points),
                "triangles": cut_triangles,
            },
            "interface": state["interface_points"],
            "volumePoints": _as_2d(volume_points),
            "interfacePoints": _as_2d(interface_points),
            "pointRadius": 0.0065,
            "interfacePointRadius": 0.008,
        },
    )


def poisson_ghost_scene() -> None:
    state = _poisson_flower_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state)
        _add_cut_cell_classes(plotter, state)
        _add_facet_bands(plotter, state["mesh"], state["ghost_facets"], color=GHOST, width=0.010)

    _export_scene(OUTPUT_DIR / "poisson-ghost-facets-scene.png", "Ghost penalty facets", draw)


def poisson_solution_scene() -> None:
    state = _poisson_flower_state()
    grid = _continuous_poisson_solution_grid(state)
    warp_factor = POISSON_SOLUTION_WARP_FACTOR
    view_data = _warped_surface_view_data([grid], warp_factor=warp_factor)
    view_data["shadeStrength"] = 0.0

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(
            grid.warp_by_scalar("u", factor=warp_factor),
            scalars="u",
            cmap="viridis",
            show_edges=True,
            edge_color="#475569",
            line_width=0.35,
            lighting=False,
        )

    _export_scene(OUTPUT_DIR / "poisson-solution-scene.png", "Cut-domain solution", draw, flat=False)
    _write_interactive_warped_surface(
        OUTPUT_DIR / "poisson-solution-view.html",
        "Interactive warped Poisson solution",
        view_data,
    )


def _dg_state():
    center = (0.08, -0.04)
    radius = 0.61
    levelset = _circle_level_set(center, radius)
    state = _poisson_state(
        n=24,
        center=center,
        radius=radius,
        levelset_degree=1,
        order=4,
        levelset=levelset,
    )
    msh = state["mesh"]
    facet_dim = msh.topology.dim - 1
    skeleton_facets = cutfemx.interior_facets_for_cells(msh, state["active"])
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(levelset)
    phi.x.scatter_forward()
    skeleton_cut = cutfemx.cut(phi, skeleton_facets, facet_dim)
    state["skeleton_facets"] = skeleton_facets
    state["omega_interior_facets"] = cutfemx.locate_entities(skeleton_cut, "phi<0")
    state["omega_cut_facet_rules"] = cutfemx.runtime_quadrature(skeleton_cut, "phi<0", 4)
    state["cut_skeleton_facets"] = cutfemx.locate_entities(skeleton_cut, "phi=0")
    state["omega_interior_segments"] = _facet_segments(msh, state["omega_interior_facets"])
    state["cut_skeleton_support_segments"] = _facet_segments(msh, state["cut_skeleton_facets"])
    state["omega_cut_segments"] = _clipped_facet_segments(
        msh,
        state["cut_skeleton_facets"],
        levelset,
    )
    return state


def dg_poisson_scene() -> None:
    state = _dg_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state)
        _add_cut_cell_classes(plotter, state)
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["omega_interior_facets"],
            color=SKELETON,
            line_width=2.4,
            opacity=0.72,
        )
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["cut_skeleton_facets"],
            color="#64748b",
            line_width=1.3,
            opacity=0.62,
        )
        _add_segments(plotter, state["omega_cut_segments"], color="#c2410c", line_width=5.0)

    _export_scene(OUTPUT_DIR / "dg-poisson-scene.png", "DG Poisson active skeleton", draw)


def dg_poisson_geometry_scene() -> None:
    state = _dg_state()
    _export_scene(
        OUTPUT_DIR / "dg-poisson-geometry-scene.png",
        "DG active and cut cells",
        lambda plotter: (_add_2d_background(plotter, state), _add_cut_cell_classes(plotter, state)),
    )


def dg_poisson_quadrature_scene() -> None:
    state = _dg_state()
    mesh_points = np.asarray(state["mesh"].geometry.x, dtype=np.float64)
    mesh_triangles = _mesh_triangles(state["mesh"])
    cut_triangles = _mesh_triangles(state["mesh"], state["cut"])
    volume_points = _quadrature_points(state["volume_rules"])
    interface_points = _quadrature_points(state["interface_rules"])
    skeleton_points = _quadrature_points(state["omega_cut_facet_rules"])

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.28)
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["omega_interior_facets"],
            color=SKELETON,
            line_width=1.8,
            opacity=0.36,
        )
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["cut_skeleton_facets"],
            color="#64748b",
            line_width=1.1,
            opacity=0.52,
        )
        _add_segments(plotter, state["omega_cut_segments"], color="#c2410c", line_width=4.2, opacity=0.9)
        _add_points(plotter, volume_points, color=QUADRATURE, point_size=3.8)
        _add_points(plotter, interface_points, color=SURFACE_QUADRATURE, point_size=4.8)
        _add_points(plotter, skeleton_points, color=SKELETON, point_size=4.2)

    _export_scene(OUTPUT_DIR / "dg-poisson-quadrature-scene.png", "DG volume, interface, and skeleton rules", draw)
    _write_interactive_2d(
        OUTPUT_DIR / "dg-poisson-quadrature-view.html",
        "Interactive DG Poisson quadrature rules",
        {
            "bounds": _plot_bounds(mesh_points),
            "background": {
                "points": _as_2d(mesh_points),
                "edges": _edges_from_triangles(mesh_triangles),
            },
            "cutCells": {
                "points": _as_2d(mesh_points),
                "triangles": cut_triangles,
            },
            "interface": state["interface_points"],
            "lineLayers": [
                {
                    "segments": state["omega_interior_segments"],
                    "color": SKELETON,
                    "width": 1.4,
                    "opacity": 0.48,
                },
                {
                    "segments": state["cut_skeleton_support_segments"],
                    "color": "#64748b",
                    "width": 1.1,
                    "opacity": 0.72,
                },
                {
                    "segments": state["omega_cut_segments"],
                    "color": "#c2410c",
                    "width": 3.2,
                    "opacity": 0.95,
                },
            ],
            "pointLayers": [
                {
                    "points": _as_2d(volume_points),
                    "color": QUADRATURE,
                    "radius": 0.007,
                    "opacity": 0.9,
                },
                {
                    "points": _as_2d(interface_points),
                    "color": SURFACE_QUADRATURE,
                    "radius": 0.0085,
                    "opacity": 0.95,
                },
                {
                    "points": _as_2d(skeleton_points),
                    "color": SKELETON,
                    "radius": 0.008,
                    "opacity": 0.95,
                },
            ],
            "pointRadius": 0.007,
            "interfacePointRadius": 0.0085,
        },
    )


def dg_poisson_skeleton_scene() -> None:
    state = _dg_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state)
        _add_background_mesh(plotter, state["mesh"], state["active"], fill_opacity=0.08, edge_opacity=0.92, line_width=0.88)
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["omega_interior_facets"],
            color=SKELETON,
            line_width=2.4,
            opacity=0.72,
        )
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["cut_skeleton_facets"],
            color="#64748b",
            line_width=1.4,
            opacity=0.7,
        )
        _add_segments(plotter, state["omega_cut_segments"], color="#c2410c", line_width=5.2)

    _export_scene(OUTPUT_DIR / "dg-poisson-skeleton-scene.png", "Clipped active DG skeleton", draw)


def dg_poisson_penalty_scene() -> None:
    state = _dg_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state)
        _add_cut_cell_classes(plotter, state)
        _add_facet_mesh(
            plotter,
            state["mesh"],
            state["omega_interior_facets"],
            color=SKELETON,
            line_width=1.5,
            opacity=0.26,
        )
        _add_segments(plotter, state["omega_cut_segments"], color=SKELETON, line_width=3.0, opacity=0.34)
        _add_facet_bands(plotter, state["mesh"], state["ghost_facets"], color=GHOST, width=0.010)

    _export_scene(OUTPUT_DIR / "dg-poisson-penalty-scene.png", "DG ghost penalty facets", draw)


def _solve_dg_poisson(state: dict) -> fem.Function:
    uh = state.get("dg_poisson_uh")
    if uh is not None:
        return uh

    msh = state["mesh"]
    phi = state["phi"]
    degree = 1
    penalty = 20.0
    interface_penalty = 40.0
    ghost_penalty = 0.1

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[state["inside"], state["volume_rules"]],
    )
    dS_omega = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[state["omega_interior_facets"], state["omega_cut_facet_rules"]],
    )
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=state["interface_rules"])
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=state["ghost_facets"])

    V = fem.functionspace(msh, ("DG", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    n_facet = ufl.FacetNormal(msh)
    n_gamma = cutfemx.normal(phi)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * u_exact
    sigma = penalty * degree**2
    sigma_gamma = interface_penalty * degree**2
    jump_u = ufl.jump(u, n_facet)
    jump_v = ufl.jump(v, n_facet)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += -ufl.inner(ufl.avg(ufl.grad(u)), jump_v) * dS_omega
    a += -ufl.inner(ufl.avg(ufl.grad(v)), jump_u) * dS_omega
    a += sigma / h_avg * ufl.inner(jump_u, jump_v) * dS_omega
    a += (
        -ufl.dot(ufl.grad(u), n_gamma) * v
        - ufl.dot(ufl.grad(v), n_gamma) * u
        + sigma_gamma / h * u * v
    ) * dx_gamma
    if state["ghost_facets"].size > 0:
        a += (
            ghost_penalty
            * h_avg
            * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
            * dS_ghost
        )

    L = f * v * dx_omega
    L += (-ufl.dot(ufl.grad(v), n_gamma) * u_exact + sigma_gamma / h * u_exact * v) * dx_gamma

    uh, _ = _solve_runtime_system(cutfemx.fem.form(a), cutfemx.fem.form(L), V, name="u_h")
    state["dg_poisson_uh"] = uh
    return uh


def dg_poisson_solution_scene() -> None:
    state = _dg_state()

    def draw(plotter: pv.Plotter) -> None:
        grid = _cut_scalar_function_grid(state["cut_data"], "phi<0", _solve_dg_poisson(state))
        plotter.add_mesh(grid, scalars="u", cmap="plasma", show_edges=True, edge_color="#475569", line_width=0.35)
        plotter.add_mesh(_circle_polyline(state["center"], state["radius"]), color=INTERFACE, line_width=4)

    _export_scene(OUTPUT_DIR / "dg-poisson-solution-scene.png", "Flat DG scalar solution", draw)


def _surface_state():
    center = (0.05, -0.03, 0.02)
    radius = 0.62
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
        (10, 10, 10),
        cell_type=mesh.CellType.hexahedron,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 2))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(_sphere_level_set(center, radius))
    phi.x.scatter_forward()
    cell_cut = cutfemx.cut(phi)
    cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
    gamma_rules = cutfemx.runtime_quadrature(cell_cut, "phi=0", 4, backend="algoim")
    facet_dim = msh.topology.dim - 1
    skeleton_facets = cutfemx.interior_facets_for_cells(msh, cut_cells)
    facet_cut = cutfemx.cut(phi, skeleton_facets, facet_dim)
    skeleton_rules = cutfemx.runtime_quadrature(facet_cut, "phi=0", 4, backend="algoim")
    surface_skeleton_facets = cutfemx.locate_entities(facet_cut, "phi=0")
    gamma_mesh = cutfemx.create_cut_mesh(cell_cut, "phi=0", mode="cut_only").mesh
    return {
        "mesh": msh,
        "phi": phi,
        "cell_cut": cell_cut,
        "gamma_mesh": gamma_mesh,
        "cut_cells": cut_cells,
        "gamma_rules": gamma_rules,
        "skeleton_rules": skeleton_rules,
        "skeleton_facets": skeleton_facets,
        "surface_skeleton_facets": surface_skeleton_facets,
        "center": center,
        "radius": radius,
    }


def _solve_surface_poisson_dg(state: dict) -> fem.Function:
    uh = state.get("surface_poisson_uh")
    if uh is not None:
        return uh

    msh = state["mesh"]
    phi = state["phi"]
    degree = 1
    penalty = 20.0
    ghost_stabilization = 0.1
    radius = state["radius"]
    center = state["center"]

    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_data=state["gamma_rules"])
    dS_gamma = ufl.Measure("dS", domain=msh, subdomain_data=state["skeleton_rules"])
    dS_ghost = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=2,
        subdomain_data=state["surface_skeleton_facets"],
    )

    V = fem.functionspace(msh, ("DG", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    n_gamma = cutfemx.normal(phi)
    mu = cutfemx.conormal(n_gamma)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    identity = ufl.Identity(msh.geometry.dim)
    P = identity - ufl.outer(n_gamma, n_gamma)
    grad_G_u = ufl.dot(P, ufl.grad(u))
    grad_G_v = ufl.dot(P, ufl.grad(v))

    n_p = n_gamma("+")
    n_m = n_gamma("-")
    P_p = identity - ufl.outer(n_p, n_p)
    P_m = identity - ufl.outer(n_m, n_m)
    avg_grad_G_u = 0.5 * (
        ufl.dot(P_p, ufl.grad(u)("+")) + ufl.dot(P_m, ufl.grad(u)("-"))
    )
    avg_grad_G_v = 0.5 * (
        ufl.dot(P_p, ufl.grad(v)("+")) + ufl.dot(P_m, ufl.grad(v)("-"))
    )
    jump_u_mu = ufl.jump(u, mu)
    jump_v_mu = ufl.jump(v, mu)

    u_surface = x[0] - center[0]
    f = (1.0 + 2.0 / radius**2) * u_surface

    a = (ufl.inner(grad_G_u, grad_G_v) + u * v) * dx_gamma
    a += -ufl.inner(avg_grad_G_u, jump_v_mu) * dS_gamma
    a += -ufl.inner(avg_grad_G_v, jump_u_mu) * dS_gamma
    a += penalty * degree**2 / h_avg * ufl.inner(jump_u_mu, jump_v_mu) * dS_gamma

    if state["surface_skeleton_facets"].size > 0:
        a += (
            ghost_stabilization
            * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
            * dS_ghost
        )

    L = f * v * dx_gamma
    uh, _ = _solve_runtime_system(cutfemx.fem.form(a), cutfemx.fem.form(L), V, name="u_h")
    state["surface_poisson_uh"] = uh
    return uh


def _surface_solution_grid(state) -> pv.UnstructuredGrid:
    return _cut_scalar_function_grid(
        state["cell_cut"],
        "phi=0",
        _solve_surface_poisson_dg(state),
        mode="cut_only",
    )


def surface_poisson_scene() -> None:
    state = _surface_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_surface(_mesh_grid(state["mesh"])), style="wireframe", color=MESH_EDGE, opacity=0.35)
        plotter.add_mesh(_surface_solution_grid(state), scalars="u", cmap="viridis", show_edges=True, edge_color="#475569")

    _export_scene(OUTPUT_DIR / "surface-poisson-dg-scene.png", "Surface Poisson DG on a sphere", draw, flat=False)


def surface_poisson_geometry_scene() -> None:
    state = _surface_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_surface(_mesh_grid(state["mesh"])), style="wireframe", color=MESH_EDGE, opacity=0.45)
        plotter.add_mesh(_mesh_grid(state["gamma_mesh"]), color="#93c5fd", show_edges=True, edge_color="#475569", opacity=0.76)

    _export_scene(OUTPUT_DIR / "surface-poisson-geometry-scene.png", "Cut surface mesh in the hexahedral box", draw, flat=False)


def surface_poisson_quadrature_scene() -> None:
    state = _surface_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_mesh_grid(state["gamma_mesh"]), color="#bfdbfe", show_edges=True, edge_color="#64748b", opacity=0.42)
        _add_points(plotter, _quadrature_points(state["gamma_rules"]), color=SURFACE_QUADRATURE, point_size=5.0)

    _export_scene(OUTPUT_DIR / "surface-poisson-quadrature-scene.png", "Surface quadrature on cut cells", draw, flat=False)


def surface_poisson_skeleton_scene() -> None:
    state = _surface_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_mesh_grid(state["gamma_mesh"]), color="#bfdbfe", show_edges=True, edge_color="#64748b", opacity=0.38)
        _add_points(plotter, _quadrature_points(state["skeleton_rules"]), color=SKELETON, point_size=5.0)

    _export_scene(OUTPUT_DIR / "surface-poisson-skeleton-scene.png", "Surface DG skeleton rules", draw, flat=False)


def surface_poisson_solution_scene() -> None:
    state = _surface_state()
    _export_scene(
        OUTPUT_DIR / "surface-poisson-solution-scene.png",
        "Surface solution field",
        lambda plotter: plotter.add_mesh(_surface_solution_grid(state), scalars="u", cmap="viridis", show_edges=True, edge_color="#475569"),
        flat=False,
    )


def _stokes_state():
    center = (-1.2, 0.0)
    radius = 0.3
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        ((-3.0, -1.0), (5.0, 1.0)),
        (96, 24),
        cell_type=mesh.CellType.triangle,
    )
    Q = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(Q, name="phi")
    phi.interpolate(_circle_level_set(center, radius))
    phi.x.scatter_forward()
    cut_data = cutfemx.cut(phi)
    fluid_cells = cutfemx.locate_entities(cut_data, "phi>0")
    cut_cells = cutfemx.locate_entities(cut_data, "phi=0")
    uncut_fluid_cells = np.setdiff1d(fluid_cells, cut_cells)
    fluid_rules = cutfemx.runtime_quadrature(cut_data, "phi>0", 4)
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", 4)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi>0")
    active_cells = np.union1d(fluid_cells, cut_cells)
    pressure_facets = cutfemx.interior_facets_for_cells(msh, active_cells)
    return {
        "mesh": msh,
        "level_set": phi,
        "cut_data": cut_data,
        "fluid_cells": fluid_cells,
        "cut_cells": cut_cells,
        "uncut_fluid_cells": uncut_fluid_cells,
        "fluid_rules": fluid_rules,
        "interface_rules": interface_rules,
        "ghost_facets": ghost_facets,
        "pressure_facets": pressure_facets,
        "center": center,
        "radius": radius,
    }


def _stokes_background(plotter: pv.Plotter, state) -> None:
    _add_background_mesh(plotter, state["mesh"], fill_opacity=0.04, edge_opacity=0.92, line_width=0.78)
    plotter.add_mesh(_circle_polyline(state["center"], state["radius"]), color=INTERFACE, line_width=4)


def stokes_scene() -> None:
    state = _stokes_state()

    def draw(plotter: pv.Plotter) -> None:
        _stokes_background(plotter, state)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["uncut_fluid_cells"]), color="#bfdbfe", show_edges=True, edge_color="#64748b", opacity=0.32)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut_cells"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.72)

    _export_scene(OUTPUT_DIR / "stokes-scene.png", "Stokes flow around a cut obstacle", draw, camera_distance=7.0)


def stokes_geometry_scene() -> None:
    state = _stokes_state()
    _export_scene(OUTPUT_DIR / "stokes-geometry-scene.png", "Triangular channel mesh and obstacle level set", lambda plotter: _stokes_background(plotter, state), camera_distance=7.0)


def stokes_cut_interface_scene() -> None:
    state = _stokes_state()

    def draw(plotter: pv.Plotter) -> None:
        _stokes_background(plotter, state)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut_cells"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.82)
        _add_points(plotter, _quadrature_points(state["interface_rules"]), color=SURFACE_QUADRATURE, point_size=5.0)

    _export_scene(OUTPUT_DIR / "stokes-cut-interface-scene.png", "Obstacle interface quadrature", draw, camera_distance=7.0)


def stokes_stabilization_scene() -> None:
    state = _stokes_state()

    def draw(plotter: pv.Plotter) -> None:
        _stokes_background(plotter, state)
        _add_facet_mesh(plotter, state["mesh"], state["pressure_facets"], color=SKELETON, line_width=1.5, opacity=0.22)
        _add_facet_bands(plotter, state["mesh"], state["ghost_facets"], color=GHOST, width=0.012)

    _export_scene(OUTPUT_DIR / "stokes-stabilization-scene.png", "Velocity and pressure stabilization facets", draw, camera_distance=7.0)


def _stokes_traction(u, p, nu, n):
    return nu * ufl.dot(ufl.grad(u), n) - p * n


def _solve_stokes_velocity(state: dict) -> fem.Function:
    velocity = state.get("velocity")
    if velocity is not None:
        return velocity

    msh = state["mesh"]
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    P1_vec = basix.ufl.element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    W = fem.functionspace(msh, basix.ufl.mixed_element([P1_vec, P1]))
    V, _ = W.sub(0).collapse()

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[state["uncut_fluid_cells"], state["fluid_rules"]],
    )
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=state["interface_rules"])
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=state["ghost_facets"])
    dS_pressure = ufl.Measure("dS", domain=msh, subdomain_id=3, subdomain_data=state["pressure_facets"])

    w = ufl.TrialFunction(W)
    u, p = ufl.split(w)
    v, q = ufl.split(ufl.TestFunction(W))
    nu = fem.Constant(msh, default_scalar_type(1.0))
    f = fem.Constant(msh, default_scalar_type((0.0, 0.0)))
    n_gamma = -cutfemx.normal(state["level_set"])
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    gamma_u = 100.0
    gamma_p = 0.1
    gamma_g = 0.1

    a = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += -p * ufl.div(v) * dx_omega
    a += ufl.div(u) * q * dx_omega
    a += -ufl.inner(_stokes_traction(u, p, nu, n_gamma), v) * dx_gamma
    a += -ufl.inner(_stokes_traction(v, q, nu, n_gamma), u) * dx_gamma
    a += gamma_u * nu / h * ufl.inner(u, v) * dx_gamma
    if state["ghost_facets"].size > 0:
        a += (
            gamma_g
            * ufl.avg(h)
            * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
            * dS_ghost
        )
    a += (
        gamma_p
        * ufl.avg(h) ** 3
        * ufl.inner(ufl.jump(ufl.grad(p), n_facet), ufl.jump(ufl.grad(q), n_facet))
        * dS_pressure
    )
    L = ufl.inner(f, v) * dx_omega

    inflow_u = fem.Function(V)
    inflow_u.interpolate(lambda x: np.stack((1.0 - x[1] ** 2, np.zeros_like(x[0]))))
    walls_u = fem.Function(V)
    walls_u.x.array[:] = 0.0

    facet_dim = msh.topology.dim - 1
    inflow_facets = mesh.locate_entities_boundary(
        msh,
        facet_dim,
        lambda x: np.isclose(x[0], -3.0),
    )
    wall_facets = mesh.locate_entities_boundary(
        msh,
        facet_dim,
        lambda x: np.isclose(np.abs(x[1]), 1.0),
    )
    inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), facet_dim, inflow_facets)
    wall_dofs = fem.locate_dofs_topological((W.sub(0), V), facet_dim, wall_facets)
    bcs = [
        fem.dirichletbc(inflow_u, inflow_dofs, W.sub(0)),
        fem.dirichletbc(walls_u, wall_dofs, W.sub(0)),
    ]

    wh, _ = _solve_runtime_system(
        cutfemx.fem.form(a),
        cutfemx.fem.form(L),
        W,
        name="stokes_solution",
        bcs=bcs,
    )
    velocity = wh.sub(0).collapse()
    velocity.name = "u"
    V_out = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    velocity_out = fem.Function(V_out, name="u")
    velocity_out.interpolate(velocity)
    velocity_out.x.scatter_forward()
    state["velocity"] = velocity_out
    return velocity_out


def _stokes_velocity_grid(state: dict) -> pv.UnstructuredGrid:
    cut_mesh = cutfemx.create_cut_mesh(state["cut_data"], "phi>0", mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("Cannot plot an empty Stokes fluid mesh.")
    velocity_cut = cutfemx.fem.cut_function(_solve_stokes_velocity(state), cut_mesh)
    grid = _mesh_grid(cut_mesh.mesh)
    vectors = np.asarray(velocity_cut.x.array, dtype=np.float64).reshape((-1, 2))
    vectors_3d = np.column_stack((vectors, np.zeros(vectors.shape[0], dtype=np.float64)))
    _attach_vector_point_data(grid, vectors_3d, vector_name="u", magnitude_name="speed")
    return grid


def stokes_solution_scene() -> None:
    state = _stokes_state()
    grid = _stokes_velocity_grid(state)

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(grid, scalars="speed", cmap="Blues", show_edges=True, edge_color="#64748b", line_width=0.2)
        plotter.add_mesh(_circle_polyline(state["center"], state["radius"]), color=INTERFACE, line_width=4)

    _export_scene(OUTPUT_DIR / "stokes-solution-scene.png", "Velocity magnitude on the cut channel", draw, camera_distance=7.0)


def _elasticity_state():
    half_length = 1.5
    grip_radius = 0.42
    waist_radius = 0.22
    gauge_half_length = 0.45
    young_modulus = 1.0e3
    poisson_ratio = 0.3
    pull = 0.12
    gamma_ghost = 0.05
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([-half_length, -0.5, -0.5]), np.array([half_length, 0.5, 0.5])),
        (40, 10, 10),
        cell_type=mesh.CellType.tetrahedron,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(_dogbone_level_set(half_length, grip_radius, waist_radius, gauge_half_length))
    phi.x.scatter_forward()
    cut_data = cutfemx.cut(phi)
    solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
    solid_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", 4)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
    solid_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full").mesh
    return {
        "mesh": msh,
        "phi": phi,
        "cut_data": cut_data,
        "solid_cells": solid_cells,
        "solid_rules": solid_rules,
        "ghost_facets": ghost_facets,
        "solid_mesh": solid_mesh,
        "half_length": half_length,
        "young_modulus": young_modulus,
        "poisson_ratio": poisson_ratio,
        "pull": pull,
        "gamma_ghost": gamma_ghost,
    }


def _elasticity_epsilon(u):
    return ufl.sym(ufl.grad(u))


def _elasticity_sigma(u, mu: float, lmbda: float, gdim: int):
    eps = _elasticity_epsilon(u)
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)


def _elasticity_von_mises_3d(u, mu: float, lmbda: float):
    stress = _elasticity_sigma(u, mu, lmbda, 3)
    return ufl.sqrt(
        0.5
        * (
            (stress[0, 0] - stress[1, 1]) ** 2
            + (stress[1, 1] - stress[2, 2]) ** 2
            + (stress[2, 2] - stress[0, 0]) ** 2
        )
        + 3.0 * (stress[0, 1] ** 2 + stress[1, 2] ** 2 + stress[2, 0] ** 2)
    )


def _solve_elasticity(state: dict) -> tuple[fem.Function, fem.Function]:
    cached = state.get("elasticity_solution")
    if cached is not None:
        return cached

    msh = state["mesh"]
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    gdim = msh.geometry.dim
    young_modulus = state["young_modulus"]
    poisson_ratio = state["poisson_ratio"]
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )

    dx_solid = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[state["solid_cells"], state["solid_rules"]],
    )
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=1, subdomain_data=state["ghost_facets"])

    a = ufl.inner(_elasticity_sigma(u, mu, lmbda, gdim), _elasticity_epsilon(v)) * dx_solid
    if state["ghost_facets"].size > 0:
        n_facet = ufl.FacetNormal(msh)
        h_avg = ufl.avg(ufl.CellDiameter(msh))
        a += (
            state["gamma_ghost"]
            * (2.0 * mu + lmbda)
            * h_avg
            * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
            * dS_ghost
        )

    zero_force = fem.Constant(msh, default_scalar_type((0.0, 0.0, 0.0)))
    L = ufl.inner(zero_force, v) * dx_solid

    facet_dim = msh.topology.dim - 1
    half_length = state["half_length"]
    left_facets = mesh.locate_entities_boundary(
        msh,
        facet_dim,
        lambda x: np.isclose(x[0], -half_length),
    )
    right_facets = mesh.locate_entities_boundary(
        msh,
        facet_dim,
        lambda x: np.isclose(x[0], half_length),
    )
    u_left = fem.Function(V)
    left_dofs = fem.locate_dofs_topological(V, facet_dim, left_facets)
    bcs = [fem.dirichletbc(u_left, left_dofs)]

    Vx, _ = V.sub(0).collapse()
    u_right_x = fem.Function(Vx)
    u_right_x.interpolate(lambda x: state["pull"] * np.ones_like(x[0]))
    right_x_dofs = fem.locate_dofs_topological((V.sub(0), Vx), facet_dim, right_facets)
    bcs.append(fem.dirichletbc(u_right_x, right_x_dofs, V.sub(0)))

    displacement, _ = _solve_runtime_system(
        cutfemx.fem.form(a),
        cutfemx.fem.form(L),
        V,
        name="deformation",
        bcs=bcs,
    )

    Q = fem.functionspace(msh, ("Discontinuous Lagrange", 0))
    von_mises = fem.Function(Q, name="von_mises")
    interpolation_points = Q.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    von_mises.interpolate(
        fem.Expression(_elasticity_von_mises_3d(displacement, mu, lmbda), interpolation_points)
    )

    state["elasticity_solution"] = (displacement, von_mises)
    return displacement, von_mises


def _elasticity_solution_grid(state) -> pv.UnstructuredGrid:
    displacement, von_mises = _solve_elasticity(state)
    cut_mesh = cutfemx.create_cut_mesh(state["cut_data"], "phi<0", mode="full")
    if cut_mesh.mesh is None:
        raise RuntimeError("Cannot plot an empty elasticity solid mesh.")
    displacement_cut = cutfemx.fem.cut_function(displacement, cut_mesh)
    von_mises_cut = cutfemx.fem.cut_function(von_mises, cut_mesh)

    grid = _mesh_grid(cut_mesh.mesh)
    displacement_values = np.asarray(displacement_cut.x.array, dtype=np.float64).reshape((-1, 3))
    if displacement_values.shape[0] >= grid.n_points:
        grid.points[:, :3] += ELASTICITY_DEFORMATION_SCALE * displacement_values[: grid.n_points]

    stress_values = np.asarray(von_mises_cut.x.array, dtype=np.float64).reshape(-1)
    if stress_values.size == grid.n_cells:
        grid.cell_data["von_mises"] = stress_values
    elif stress_values.size >= grid.n_points:
        grid.point_data["von_mises"] = stress_values[: grid.n_points]
    else:
        raise RuntimeError(
            f"Cannot attach {stress_values.size} von Mises values to elasticity grid."
        )
    grid.set_active_scalars("von_mises")
    return grid


def elasticity_scene() -> None:
    state = _elasticity_state()
    _export_scene(
        OUTPUT_DIR / "elasticity-von-mises-scene.png",
        "Cut dogbone elasticity",
        lambda plotter: plotter.add_mesh(_surface(_elasticity_solution_grid(state)), scalars="von_mises", cmap="turbo", show_edges=False),
        flat=False,
        camera_zoom=1.28,
    )


def elasticity_geometry_scene() -> None:
    state = _elasticity_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_box_edges((-1.5, 1.5, -0.5, 0.5, -0.5, 0.5)), color=MESH_EDGE, line_width=2)
        plotter.add_mesh(_surface(_mesh_grid(state["solid_mesh"])), color="#bfdbfe", show_edges=True, edge_color="#475569", opacity=0.72)

    _export_scene(OUTPUT_DIR / "elasticity-geometry-scene.png", "Cut solid mesh in the tetrahedral box", draw, flat=False)


def elasticity_quadrature_scene() -> None:
    state = _elasticity_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_surface(_mesh_grid(state["solid_mesh"])), color="#bfdbfe", show_edges=True, edge_color="#64748b", opacity=0.35)
        _add_points(plotter, _quadrature_points(state["solid_rules"])[::8], color=QUADRATURE, point_size=4.2)

    _export_scene(OUTPUT_DIR / "elasticity-quadrature-scene.png", "Cut-volume quadrature in the solid", draw, flat=False)


def elasticity_stabilization_scene() -> None:
    state = _elasticity_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_surface(_mesh_grid(state["solid_mesh"])), color="#bfdbfe", show_edges=True, edge_color="#64748b", opacity=0.32)
        _add_facet_mesh(plotter, state["mesh"], state["ghost_facets"], color=GHOST, line_width=3, opacity=0.82)

    _export_scene(OUTPUT_DIR / "elasticity-stabilization-scene.png", "Ghost penalty facets around the cut solid", draw, flat=False)


def elasticity_solution_scene() -> None:
    state = _elasticity_state()
    _export_scene(
        OUTPUT_DIR / "elasticity-von-mises-solution-scene.png",
        "Deformed displacement and von Mises field",
        lambda plotter: plotter.add_mesh(_surface(_elasticity_solution_grid(state)), scalars="von_mises", cmap="turbo", show_edges=False),
        flat=False,
        camera_zoom=1.28,
    )


def _moving_base_state() -> dict:
    n = 16
    radius = 0.5
    order = 2
    center = (0.0, 0.0)
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        ((-1.0, -1.0), (4.0, 1.0)),
        (5 * n, n),
        cell_type=mesh.CellType.triangle,
    )
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(_circle_level_set(center, radius))
    phi.x.scatter_forward()
    cut_data = cutfemx.cut(phi)
    state = {
        "mesh": msh,
        "phi": phi,
        "cut_data": cut_data,
        "radius": radius,
        "order": order,
    }
    return _refresh_moving_state(state, center)


def _refresh_moving_state(state: dict, center: tuple[float, float]) -> dict:
    radius = state["radius"]
    phi = state["phi"]
    phi.interpolate(_circle_level_set(center, radius))
    phi.x.scatter_forward()
    cutfemx.update(state["cut_data"])
    cut_data = state["cut_data"]
    state.pop("moving_poisson_uh", None)
    state.update(
        {
            "inside": cutfemx.locate_entities(cut_data, "phi<0"),
            "cut": cutfemx.locate_entities(cut_data, "phi=0"),
            "active": cutfemx.locate_entities(cut_data, "phi<=0"),
            "volume_rules": cutfemx.runtime_quadrature(cut_data, "phi<0", state["order"]),
            "interface_rules": cutfemx.runtime_quadrature(cut_data, "phi=0", state["order"]),
            "ghost_facets": cutfemx.ghost_penalty_facets(cut_data, "phi<0"),
            "center": center,
            "interface_points": _circle_points(center, radius),
        }
    )
    return state


def _moving_state(center: tuple[float, float]) -> dict:
    return _refresh_moving_state(_moving_base_state(), center)


def moving_poisson_scene() -> None:
    states = [_moving_state((float(step), 0.0)) for step in range(4)]

    def draw(plotter: pv.Plotter) -> None:
        _add_background_mesh(plotter, states[0]["mesh"], fill_opacity=0.04, edge_opacity=0.92, line_width=0.78)
        colors = ["#0f766e", "#2563eb", "#7c3aed", "#dc2626"]
        for state, color in zip(states, colors):
            plotter.add_mesh(_mesh_grid(state["mesh"], state["active"]), color=color, show_edges=True, edge_color="#475569", opacity=0.22)
            plotter.add_mesh(_circle_polyline(state["center"], state["radius"]), color=color, line_width=4)

    _export_scene(OUTPUT_DIR / "moving-poisson-scene.png", "Moving cut domain on a fixed triangular mesh", draw, camera_distance=6.5)


def moving_poisson_step_scene() -> None:
    state = _moving_state((1.0, 0.0))
    _export_scene(
        OUTPUT_DIR / "moving-poisson-step-scene.png",
        "One rebuilt cut state",
        lambda plotter: (_add_2d_background(plotter, state, opacity=0.22), _add_cut_cell_classes(plotter, state)),
        camera_distance=6.5,
    )


def moving_poisson_quadrature_scene() -> None:
    state = _moving_state((1.0, 0.0))

    def draw(plotter: pv.Plotter) -> None:
        _add_2d_background(plotter, state, opacity=0.22)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.28)
        _add_points(plotter, _quadrature_points(state["volume_rules"]), color=QUADRATURE, point_size=3.6)
        _add_points(plotter, _quadrature_points(state["interface_rules"]), color=SURFACE_QUADRATURE, point_size=4.6)

    _export_scene(OUTPUT_DIR / "moving-poisson-quadrature-scene.png", "Rebuilt runtime quadrature", draw, camera_distance=6.5)


def _solve_moving_poisson(state: dict) -> fem.Function:
    uh = state.get("moving_poisson_uh")
    if uh is not None:
        return uh

    msh = state["mesh"]
    phi = state["phi"]
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    source = fem.Constant(msh, default_scalar_type(1.0))
    boundary_value = fem.Constant(msh, default_scalar_type(0.0))
    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)
    gamma = 100.0
    gamma_g = 0.1

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[state["inside"], state["volume_rules"]],
    )
    dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=state["interface_rules"])
    dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=state["ghost_facets"])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a += -ufl.dot(ufl.grad(u), n_gamma) * v * dx_gamma
    a += -ufl.dot(ufl.grad(v), n_gamma) * u * dx_gamma
    a += gamma / h * u * v * dx_gamma
    if state["ghost_facets"].size > 0:
        a += (
            gamma_g
            * h_avg
            * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
            * dS_ghost
        )

    L = source * v * dx_omega
    L += -ufl.dot(ufl.grad(v), n_gamma) * boundary_value * dx_gamma
    L += gamma / h * v * boundary_value * dx_gamma

    uh, _ = _solve_runtime_system(cutfemx.fem.form(a), cutfemx.fem.form(L), V, name="u_h")
    state["moving_poisson_uh"] = uh
    return uh


def _moving_poisson_solution_grids() -> tuple[mesh.Mesh, list[tuple[int, tuple[float, float], pv.UnstructuredGrid]]]:
    state = _moving_base_state()
    frames = []
    for step in range(4):
        center = (float(step), 0.0)
        _refresh_moving_state(state, center)
        grid = _cut_scalar_function_grid(state["cut_data"], "phi<0", _solve_moving_poisson(state))
        frames.append((step, center, grid.copy(deep=True)))
    return state["mesh"], frames


def _scalar_range(grids: list[pv.UnstructuredGrid], name: str) -> tuple[float, float]:
    values = []
    for grid in grids:
        if name in grid.point_data:
            values.append(np.asarray(grid.point_data[name], dtype=np.float64))
        elif name in grid.cell_data:
            values.append(np.asarray(grid.cell_data[name], dtype=np.float64))
    if not values:
        return (0.0, 1.0)
    data = np.concatenate(values)
    return (float(np.nanmin(data)), float(np.nanmax(data)))


def moving_poisson_solution_video() -> None:
    from PIL import Image

    background_mesh, step_grids = _moving_poisson_solution_grids()
    scalar_limits = _scalar_range([grid for _, _, grid in step_grids], "u")
    images: list[Image.Image] = []
    for step, center, grid in step_grids:
        title = f"Moving Poisson update step {step}"

        def draw(plotter: pv.Plotter, *, center=center, grid=grid) -> None:
            _add_background_mesh(
                plotter,
                background_mesh,
                fill_opacity=0.04,
                edge_opacity=0.82,
                line_width=0.70,
            )
            plotter.add_mesh(
                grid,
                scalars="u",
                cmap="viridis",
                clim=scalar_limits,
                show_edges=True,
                edge_color="#475569",
                opacity=0.82,
            )
            plotter.add_mesh(_circle_polyline(center, 0.5), color=INTERFACE, line_width=3)

        plotter = _prepare_plotter(
            OUTPUT_DIR / "moving-poisson-update-sequence",
            title,
            draw,
            camera_distance=6.5,
        )
        frame = plotter.screenshot(return_img=True)
        plotter.close()
        images.append(Image.fromarray(frame[:, :, :3]).convert("RGB"))

    poster_path = OUTPUT_DIR / "moving-poisson-update-sequence-poster.png"
    gif_path = OUTPUT_DIR / "moving-poisson-update-sequence.gif"
    mp4_path = OUTPUT_DIR / "moving-poisson-update-sequence.mp4"
    images[0].save(poster_path)
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=1200,
        loop=0,
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    with TemporaryDirectory(prefix="cutfemx-moving-poisson-") as tmp:
        frame_dir = Path(tmp)
        frame_index = 0
        for index, image in enumerate(images):
            for _ in range(6):
                image.save(frame_dir / f"frame_{frame_index:03d}.png")
                frame_index += 1
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "5",
                "-i",
                str(frame_dir / "frame_%03d.png"),
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-r",
                "5",
                "-g",
                "1",
                "-bf",
                "0",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _interface_state() -> dict:
    state = _poisson_state(n=32, center=(0.05, -0.03), radius=0.53, levelset_degree=1, order=4)
    msh = state["mesh"]
    facet_dim = msh.topology.dim - 1
    msh.topology.create_entities(facet_dim)
    msh.topology.create_connectivity(facet_dim, msh.topology.dim)
    state["outside"] = cutfemx.locate_entities(state["cut_data"], "phi>0")
    state["outside_rules"] = cutfemx.runtime_quadrature(state["cut_data"], "phi>0", 4)
    state["inside_ghost_facets"] = cutfemx.ghost_penalty_facets(state["cut_data"], "phi<0")
    state["outside_ghost_facets"] = cutfemx.ghost_penalty_facets(state["cut_data"], "phi>0")
    state["exterior_facets"] = mesh.exterior_facet_indices(msh.topology)
    state["kappa_1"] = 1.0
    state["kappa_2"] = 8.0
    return state


def _assemble_matrix_block(form) -> la.MatrixCSR:
    A = cutfemx.fem.assemble_matrix(form)
    A.scatter_reverse()
    return A


def _assemble_vector_block(form) -> la.Vector:
    b = cutfemx.fem.assemble_vector(form)
    b.scatter_reverse(la.InsertMode.add)
    return b


def _solve_interface_block_system(
    a_forms,
    L_forms,
    domain1: cutfemx.fem.ActiveDomain,
    domain2: cutfemx.fem.ActiveDomain,
    V1: fem.FunctionSpace,
    V2: fem.FunctionSpace,
) -> tuple[fem.Function, fem.Function]:
    A_matrix_blocks = [[_assemble_matrix_block(a_forms[i][j]) for j in range(2)] for i in range(2)]
    b_vector_blocks = [_assemble_vector_block(L_forms[i]) for i in range(2)]
    cutfemx.fem.deactivate_outside_blocks(A_matrix_blocks, [domain1, domain2], b_vector_blocks)
    A = bmat(
        [[A.to_scipy().tocsr() for A in row] for row in A_matrix_blocks],
        format="csr",
    )
    b = np.concatenate([block.array.copy() for block in b_vector_blocks])
    n1 = b_vector_blocks[0].array.size
    solution = spsolve(A, b)

    u1_h = fem.Function(V1, name="u1_h")
    u2_h = fem.Function(V2, name="u2_h")
    u1_h.x.array[:] = solution[:n1]
    u2_h.x.array[:] = solution[n1:]
    u1_h.x.scatter_forward()
    u2_h.x.scatter_forward()
    return u1_h, u2_h


def _solve_interface_poisson(state: dict) -> tuple[fem.Function, fem.Function]:
    cached = state.get("interface_solution")
    if cached is not None:
        return cached

    msh = state["mesh"]
    center = state["center"]
    radius = state["radius"]
    kappa_1 = state["kappa_1"]
    kappa_2 = state["kappa_2"]

    dx1 = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=[state["inside"], state["volume_rules"]],
    )
    dx2 = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=2,
        subdomain_data=[state["outside"], state["outside_rules"]],
    )
    dgamma = ufl.Measure("dx", domain=msh, subdomain_id=3, subdomain_data=state["interface_rules"])
    dS_ghost_1 = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=4,
        subdomain_data=state["inside_ghost_facets"],
    )
    dS_ghost_2 = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=5,
        subdomain_data=state["outside_ghost_facets"],
    )
    ds_outer = ufl.Measure("ds", domain=msh, subdomain_id=6, subdomain_data=state["exterior_facets"])

    V1 = fem.functionspace(msh, ("Lagrange", 1))
    V2 = fem.functionspace(msh, ("Lagrange", 1))
    W_block = ufl.MixedFunctionSpace(V1, V2)
    u1, u2 = ufl.TrialFunctions(W_block)
    v1, v2 = ufl.TestFunctions(W_block)

    x = ufl.SpatialCoordinate(msh)
    r2 = (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2
    ratio = kappa_1 / kappa_2
    u2_exact = ratio * r2 + radius**2 * (1.0 - ratio)
    f1 = fem.Constant(msh, default_scalar_type(-4.0 * kappa_1))
    f2 = fem.Constant(msh, default_scalar_type(-4.0 * kappa_1))

    n_gamma = cutfemx.normal(state["phi"])
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)
    gamma_interface = 40.0
    gamma_boundary = 40.0
    gamma_ghost = 0.1
    kappa_h = 2.0 * kappa_1 * kappa_2 / (kappa_1 + kappa_2)
    eta_interface = gamma_interface * kappa_h / h
    eta_boundary = gamma_boundary * kappa_2 / h
    w1 = kappa_2 / (kappa_1 + kappa_2)
    w2 = kappa_1 / (kappa_1 + kappa_2)

    def avg_flux_pair(side1, side2):
        return w1 * kappa_1 * ufl.dot(ufl.grad(side1), n_gamma) + w2 * kappa_2 * ufl.dot(
            ufl.grad(side2),
            n_gamma,
        )

    jump_u = u1 - u2
    jump_v = v1 - v2
    flux_u = avg_flux_pair(u1, u2)
    flux_v = avg_flux_pair(v1, v2)

    a = kappa_1 * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1
    a += kappa_2 * ufl.inner(ufl.grad(u2), ufl.grad(v2)) * dx2
    a += (-flux_u * jump_v - flux_v * jump_u + eta_interface * jump_u * jump_v) * dgamma

    if state["inside_ghost_facets"].size > 0:
        a += (
            gamma_ghost
            * kappa_1
            * h_avg
            * ufl.inner(ufl.jump(ufl.grad(u1), n_facet), ufl.jump(ufl.grad(v1), n_facet))
            * dS_ghost_1
        )
    if state["outside_ghost_facets"].size > 0:
        a += (
            gamma_ghost
            * kappa_2
            * h_avg
            * ufl.inner(ufl.jump(ufl.grad(u2), n_facet), ufl.jump(ufl.grad(v2), n_facet))
            * dS_ghost_2
        )

    a += (
        -kappa_2 * ufl.dot(ufl.grad(u2), n_facet) * v2
        - kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2
        + eta_boundary * u2 * v2
    ) * ds_outer

    L = f1 * v1 * dx1 + f2 * v2 * dx2
    L += (
        -kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2_exact + eta_boundary * u2_exact * v2
    ) * ds_outer

    a_blocks = ufl.extract_blocks(a)
    L_blocks = ufl.extract_blocks(L)
    a_forms = tuple(tuple(cutfemx.fem.form(a_blocks[i][j]) for j in range(2)) for i in range(2))
    L_forms = tuple(cutfemx.fem.form(L_blocks[i]) for i in range(2))
    domain1 = cutfemx.fem.active_domain(a_forms[0][0])
    domain2 = cutfemx.fem.active_domain(a_forms[1][1])
    solution = _solve_interface_block_system(a_forms, L_forms, domain1, domain2, V1, V2)
    state["interface_solution"] = solution
    return solution


def _interface_solution_grid(state: dict, selector: str, phase: int) -> pv.DataSet:
    u1_h, u2_h = _solve_interface_poisson(state)
    return _cut_scalar_function_grid(
        state["cut_data"],
        selector,
        u1_h if phase == 1 else u2_h,
    )


def interface_poisson_scene() -> None:
    state = _interface_state()
    inside_grid = _interface_solution_grid(state, "phi<0", phase=1)
    outside_grid = _interface_solution_grid(state, "phi>0", phase=2)
    values = np.concatenate((inside_grid["u"], outside_grid["u"]))
    clim = (float(values.min()), float(values.max()))
    warp_factor = INTERFACE_POISSON_SOLUTION_WARP_FACTOR
    view_data = _warped_surface_view_data([outside_grid, inside_grid], warp_factor=warp_factor)
    view_data["boundaryEdges"] = []
    view_data["shadeStrength"] = 0.0

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(
            outside_grid.warp_by_scalar("u", factor=warp_factor),
            scalars="u",
            cmap="viridis",
            clim=clim,
            show_edges=True,
            edge_color="#475569",
            opacity=1.0,
            lighting=False,
        )
        plotter.add_mesh(
            inside_grid.warp_by_scalar("u", factor=warp_factor),
            scalars="u",
            cmap="viridis",
            clim=clim,
            show_edges=True,
            edge_color="#334155",
            opacity=1.0,
            lighting=False,
        )

    _export_scene(OUTPUT_DIR / "interface-poisson-scene.png", "Two-domain interface solution", draw, flat=False)
    _write_interactive_warped_surface(
        OUTPUT_DIR / "interface-poisson-solution-view.html",
        "Interactive warped interface Poisson solution",
        view_data,
    )


def interface_poisson_geometry_scene() -> None:
    state = _interface_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_background_mesh(plotter, state["mesh"], fill_opacity=0.03, edge_opacity=0.88, line_width=0.78)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["outside"]), color="#93c5fd", show_edges=True, edge_color="#475569", opacity=0.42)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["inside"]), color=INTERIOR, show_edges=True, edge_color="#475569", opacity=0.62)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.84)
        plotter.add_mesh(_closed_polyline(state["interface_points"], z=0.08), color=INTERFACE, line_width=4)

    _export_scene(OUTPUT_DIR / "interface-poisson-geometry-scene.png", "Two material phases", draw)


def interface_poisson_quadrature_scene() -> None:
    state = _interface_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_background_mesh(plotter, state["mesh"], fill_opacity=0.02, edge_opacity=0.86, line_width=0.72)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.24)
        plotter.add_mesh(_closed_polyline(state["interface_points"], z=0.08), color=INTERFACE, line_width=3)
        _add_points(plotter, _quadrature_points(state["volume_rules"]), color=QUADRATURE, point_size=3.8)
        _add_points(plotter, _quadrature_points(state["outside_rules"]), color="#16a34a", point_size=3.8)
        _add_points(plotter, _quadrature_points(state["interface_rules"]), color=SURFACE_QUADRATURE, point_size=5.0)

    _export_scene(OUTPUT_DIR / "interface-poisson-quadrature-scene.png", "Phase and interface quadrature", draw)


def interface_poisson_coupling_scene() -> None:
    state = _interface_state()

    def draw(plotter: pv.Plotter) -> None:
        _add_background_mesh(plotter, state["mesh"], fill_opacity=0.03, edge_opacity=0.88, line_width=0.74)
        plotter.add_mesh(_mesh_grid(state["mesh"], state["cut"]), color=CUT, show_edges=True, edge_color="#7c2d12", opacity=0.46)
        _add_facet_bands(plotter, state["mesh"], state["inside_ghost_facets"], color="#dc2626", width=0.010, opacity=0.76)
        _add_facet_bands(plotter, state["mesh"], state["outside_ghost_facets"], color="#2563eb", width=0.008, opacity=0.54)
        plotter.add_mesh(_closed_polyline(state["interface_points"], z=0.09), color=INTERFACE, line_width=5)

    _export_scene(OUTPUT_DIR / "interface-poisson-coupling-scene.png", "Nitsche interface and ghost bands", draw)


def _boundary_perimeter_state() -> dict:
    centre = np.asarray((-0.11, 0.5, 0.5), dtype=np.float64)
    radius = 0.34
    msh = mesh.create_box(
        MPI.COMM_WORLD,
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (16, 16, 16),
        cell_type=mesh.CellType.tetrahedron,
    )
    fdim = msh.topology.dim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V, name="phi")
    phi.interpolate(
        lambda x: (x[0] - centre[0]) ** 2
        + (x[1] - centre[1]) ** 2
        + (x[2] - centre[2]) ** 2
        - radius**2
    )
    phi.x.scatter_forward()
    exterior_facets = mesh.exterior_facet_indices(msh.topology)
    cut_data = cutfemx.cut(phi, exterior_facets, fdim)
    circle_radius = float(np.sqrt(radius**2 - centre[0] ** 2))
    return {
        "mesh": msh,
        "cut_data": cut_data,
        "exterior_facets": exterior_facets,
        "negative_facets": cutfemx.locate_entities(cut_data, "phi<0"),
        "cut_facets": cutfemx.locate_entities(cut_data, "phi=0"),
        "patch_rules": cutfemx.runtime_quadrature(cut_data, "phi<0", 4),
        "curve_rules": cutfemx.runtime_quadrature(cut_data, "phi=0", 4),
        "patch_mesh": cutfemx.create_cut_mesh(cut_data, "phi<0", mode="cut_only").mesh,
        "curve_mesh": cutfemx.create_cut_mesh(cut_data, "phi=0", mode="cut_only").mesh,
        "circle_radius": circle_radius,
    }


def boundary_perimeter_scene() -> None:
    state = _boundary_perimeter_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_box_edges((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)), color=MESH_EDGE, line_width=2.0)
        plotter.add_mesh(_mesh_grid(state["patch_mesh"]), color=INTERIOR, show_edges=True, edge_color="#166534", opacity=0.72)
        plotter.add_mesh(_mesh_grid(state["curve_mesh"]), color=GHOST, line_width=6)

    _export_scene(OUTPUT_DIR / "boundary-perimeter-scene.png", "Sphere cut on exterior facets", draw, flat=False, camera_distance=4.5, camera_zoom=0.72)


def boundary_perimeter_facets_scene() -> None:
    state = _boundary_perimeter_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_box_edges((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)), color=MESH_EDGE, line_width=2.0)
        exterior = _facet_polydata(state["mesh"], state["exterior_facets"])
        cut = _facet_polydata(state["mesh"], state["cut_facets"])
        if exterior is not None:
            plotter.add_mesh(exterior, style="wireframe", color="#64748b", opacity=0.22, line_width=0.6)
        if cut is not None:
            plotter.add_mesh(cut, color=CUT, opacity=0.82, show_edges=True, edge_color="#7c2d12")
        plotter.add_mesh(_mesh_grid(state["curve_mesh"]), color=GHOST, line_width=6)

    _export_scene(OUTPUT_DIR / "boundary-perimeter-facets-scene.png", "Exterior facets cut by phi", draw, flat=False, camera_distance=4.5, camera_zoom=0.72)


def boundary_perimeter_quadrature_scene() -> None:
    state = _boundary_perimeter_state()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(_box_edges((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)), color=MESH_EDGE, line_width=2.0)
        plotter.add_mesh(_mesh_grid(state["patch_mesh"]), color=INTERIOR, show_edges=True, edge_color="#166534", opacity=0.34)
        plotter.add_mesh(_mesh_grid(state["curve_mesh"]), color=GHOST, line_width=5)
        _add_points(plotter, _quadrature_points(state["patch_rules"]), color=QUADRATURE, point_size=5.0)
        _add_points(plotter, _quadrature_points(state["curve_rules"]), color=SURFACE_QUADRATURE, point_size=7.0)

    _export_scene(OUTPUT_DIR / "boundary-perimeter-quadrature-scene.png", "Patch and curve quadrature", draw, flat=False, camera_distance=4.5, camera_zoom=0.72)


def _dino_surface() -> pv.PolyData:
    surface = pv.read((ROOT.parent / "python" / "demo" / "Dino.stl").as_posix())
    return surface.triangulate().clean()


def _padded_bounds(dataset: pv.DataSet, padding_fraction: float = 0.10) -> tuple[float, float, float, float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = dataset.bounds
    pad = padding_fraction * max(xmax - xmin, ymax - ymin, zmax - zmin)
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad, zmin - pad, zmax + pad)


def _stl_distance_state(*, compute_distance: bool = False) -> dict:
    global _STL_DISTANCE_STATE
    if _STL_DISTANCE_STATE is not None and (
        not compute_distance or "distance" in _STL_DISTANCE_STATE
    ):
        return _STL_DISTANCE_STATE

    from cutfemx.distance import adapt_mesh_to_stl, compute_stl_bbox

    stl_path = ROOT.parent / "python" / "demo" / "Dino.stl"
    surface = _dino_surface()
    comm = MPI.COMM_WORLD
    min_pt_stl = np.zeros(3, dtype=np.float64)
    max_pt_stl = np.zeros(3, dtype=np.float64)

    if comm.rank == 0:
        min_c, max_c = compute_stl_bbox(stl_path.as_posix())
        min_pt_stl[:] = min_c
        max_pt_stl[:] = max_c

    comm.Bcast(min_pt_stl, root=0)
    comm.Bcast(max_pt_stl, root=0)

    min_coord = np.min(min_pt_stl)
    max_coord = np.max(max_pt_stl)
    padding = 0.1 * (max_coord - min_coord)
    min_pt_mesh = min_pt_stl - padding
    max_pt_mesh = max_pt_stl + padding

    length_mesh = np.zeros(3, dtype=np.float64)
    if comm.rank == 0:
        length_mesh[:] = max_pt_mesh - min_pt_mesh
    comm.Bcast(length_mesh, root=0)

    n_min = 5
    min_length = np.min(length_mesh)
    n_mesh = np.round(length_mesh / min_length * n_min).astype(np.int32)
    background_mesh = mesh.create_box(
        comm,
        [min_pt_mesh, max_pt_mesh],
        n_mesh,
        cell_type=mesh.CellType.tetrahedron,
        ghost_mode=mesh.GhostMode.shared_facet,
    )
    refined_mesh = adapt_mesh_to_stl(
        background_mesh,
        stl_path.as_posix(),
        nlevels=3,
        k_ring=1,
        aabb_padding=0.0,
        ghost_mode=mesh.GhostMode.shared_facet,
    )

    state = {
        "surface": surface,
        "mesh": refined_mesh,
        "stl_path": stl_path,
    }

    if compute_distance:
        from cutfemx.distance import SignMode, from_stl

        distance = from_stl(
            refined_mesh,
            stl_path.as_posix(),
            sign_mode=SignMode.ComponentAnchor,
        )
        distance.name = "signed_distance"
        state["distance"] = distance

    _STL_DISTANCE_STATE = state
    return state


def stl_distance_scene() -> None:
    surface = _dino_surface()

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(surface, color="#94a3b8", show_edges=True, edge_color="#334155", line_width=0.15, opacity=0.95)

    _export_scene(OUTPUT_DIR / "stl-distance-scene.png", "Input STL surface", draw, flat=False, camera_distance=18.0)


def stl_distance_bbox_scene() -> None:
    surface = _dino_surface()
    bounds = _padded_bounds(surface)

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(surface, color="#94a3b8", show_edges=True, edge_color="#334155", line_width=0.12, opacity=0.72)
        plotter.add_mesh(_box_edges(bounds), color=CUT, line_width=3.0)

    _export_scene(OUTPUT_DIR / "stl-distance-bbox-scene.png", "Padded background box", draw, flat=False, camera_distance=20.0)


def stl_distance_refined_mesh_scene() -> None:
    state = _stl_distance_state()
    surface = state["surface"]
    refined_grid = _mesh_grid(state["mesh"])
    middle_slice = refined_grid.slice(normal=(1.0, 0.0, 0.0), origin=surface.center)

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(refined_grid, style="wireframe", color="#64748b", opacity=0.08, line_width=0.25)
        if middle_slice.n_points > 0:
            plotter.add_mesh(
                middle_slice,
                color="#dbeafe",
                show_edges=True,
                edge_color="#1e3a8a",
                line_width=0.55,
                opacity=0.52,
            )
        plotter.add_mesh(
            surface,
            color="#0f766e",
            show_edges=True,
            edge_color="#064e3b",
            line_width=0.18,
            opacity=0.90,
        )

    _export_scene(
        OUTPUT_DIR / "stl-distance-refined-mesh-scene.png",
        "Refined background mesh around STL",
        draw,
        flat=False,
        camera_distance=20.0,
        camera_zoom=1.02,
    )


def stl_distance_field_scene() -> None:
    state = _stl_distance_state(compute_distance=True)
    surface = state["surface"]
    distance_grid = _function_grid(state["distance"], "signed_distance")
    distance_slice = distance_grid.slice(normal=(1.0, 0.0, 0.0), origin=surface.center)
    values = np.asarray(distance_slice["signed_distance"], dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    distance_scale = float(np.percentile(np.abs(finite_values), 92)) if finite_values.size else 1.0
    distance_scale = max(distance_scale, 1.0e-10)
    clim = (-distance_scale, distance_scale)
    distance_slice["signed_distance_clipped"] = np.clip(values, clim[0], clim[1])
    contour_levels = np.linspace(clim[0], clim[1], 17)
    contour_levels = contour_levels[np.abs(contour_levels) > 1.0e-12]
    contours = distance_slice.contour(isosurfaces=contour_levels, scalars="signed_distance")
    zero_contour = distance_slice.contour(isosurfaces=[0.0], scalars="signed_distance")

    def draw(plotter: pv.Plotter) -> None:
        plotter.add_mesh(
            distance_slice,
            scalars="signed_distance_clipped",
            cmap="coolwarm",
            clim=clim,
            opacity=0.92,
            show_edges=False,
        )
        plotter.add_mesh(surface, color="#0f766e", opacity=0.08, show_edges=False)
        if contours.n_points > 0:
            plotter.add_mesh(contours, color="#0f172a", line_width=2.0)
        if zero_contour.n_points > 0:
            plotter.add_mesh(zero_contour, color="#facc15", line_width=5.0)

    _export_scene(
        OUTPUT_DIR / "stl-distance-yz-contours-scene.png",
        "Middle y-z signed-distance contours",
        draw,
        flat=False,
        camera_distance=20.0,
        camera_zoom=1.02,
        view="yz",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    poisson_scene()
    poisson_background_scene()
    poisson_interior_scene()
    poisson_cut_cells_scene()
    poisson_quadrature_scene()
    poisson_ghost_scene()
    poisson_solution_scene()
    dg_poisson_scene()
    dg_poisson_geometry_scene()
    dg_poisson_quadrature_scene()
    dg_poisson_skeleton_scene()
    dg_poisson_penalty_scene()
    dg_poisson_solution_scene()
    surface_poisson_scene()
    surface_poisson_geometry_scene()
    surface_poisson_quadrature_scene()
    surface_poisson_skeleton_scene()
    surface_poisson_solution_scene()
    stokes_scene()
    stokes_geometry_scene()
    stokes_cut_interface_scene()
    stokes_stabilization_scene()
    stokes_solution_scene()
    elasticity_scene()
    elasticity_geometry_scene()
    elasticity_quadrature_scene()
    elasticity_stabilization_scene()
    elasticity_solution_scene()
    moving_poisson_scene()
    moving_poisson_step_scene()
    moving_poisson_quadrature_scene()
    moving_poisson_solution_video()
    interface_poisson_scene()
    interface_poisson_geometry_scene()
    interface_poisson_quadrature_scene()
    interface_poisson_coupling_scene()
    boundary_perimeter_scene()
    boundary_perimeter_facets_scene()
    boundary_perimeter_quadrature_scene()
    stl_distance_scene()
    stl_distance_bbox_scene()
    stl_distance_refined_mesh_scene()
    stl_distance_field_scene()
    print(f"Wrote tutorial screenshots to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
