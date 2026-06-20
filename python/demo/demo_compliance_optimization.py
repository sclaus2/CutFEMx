# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

"""Level-set compliance optimization demo.

This is a deliberately compact CutFEM topology-optimization demonstrator. The
background mesh is fixed, the solid is ``phi < 0``, internal void boundaries are
updated by an H1/Riesz normal speed and a SUPG-stabilized level-set advection
step, and cut data is rebuilt from the current level set at every iteration.
The default benchmark uses the
OptiCut rectangular cantilever data: a ``[0, 2] x [0, 1]`` mesh, clamped left
edge, a centered partial traction on the right edge, and the striped OptiCut
cosine initial level set.  The initial level set is redistanced once with FIM
before the first elasticity solve.  Pass ``--initial-geometry holes`` for the
smaller circular-hole debugging initializer.

The demo is also a lightweight benchmark: it writes a per-iteration profile CSV
with phase timings and memory counters, and it periodically exports XDMF
snapshots for movie generation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import gc
from pathlib import Path
import resource
import sys
import time
import tracemalloc
from typing import Callable, Iterator
import warnings

import basix.ufl
from mpi4py import MPI
import numpy as np
import ufl

from cutfemx.distance import NormalExtensionResult, reinitialize
import cutfemx
from dolfinx import default_scalar_type, fem, io, la, mesh


PHASES = (
    "cut",
    "forms",
    "assembly",
    "solve",
    "sensitivity",
    "extension",
    "shape_check",
    "advection",
    "reinitialization",
    "output",
)

VELOCITY_NORMALIZATION_SCALE_FLOOR_FRACTION = 5.0e-2
VELOCITY_NORMALIZATION_ABS_FLOOR = 1.0e-14

PROFILE_FIELDS = (
    "iteration",
    "cells",
    "solid_cells",
    "cut_cells",
    "ghost_facets",
    "inactive_dofs",
    "compliance",
    "volume",
    "volume_fraction",
    "target_volume",
    "volume_constraint",
    "target_delta_volume",
    "lagrangian",
    "alm_lambda",
    "alm_penalty",
    "alm_multiplier",
    "volume_after_advection",
    "volume_after_raw_reinitialization",
    "volume_after_reinitialization",
    "reinit_volume_shift",
    "dt",
    "lambda_volume",
    "energy_scale",
    "elastic_half_energy",
    "raw_velocity_max",
    "velocity_h1_scale",
    "velocity_scale_floor",
    "velocity_scale",
    "velocity_scale_limited",
    "velocity_max",
    "speed_min",
    "speed_max",
    "phi_min",
    "phi_max",
    "phi_min_abs_dof",
    "supg_tau_scale",
    "volume_rate_target",
    "volume_rate_shape",
    "volume_rate_constraint",
    "volume_rate_predicted",
    "shape_check_step",
    "shape_check_predicted",
    "shape_check_predicted_half_energy",
    "shape_check_predicted_volume",
    "shape_check_predicted_discrete_volume",
    "shape_check_fd_half_energy",
    "shape_check_fd_compliance",
    "shape_check_fd_volume",
    "shape_check_rel_error",
    "shape_check_rel_error_half_energy",
    "shape_check_rel_error_volume",
    "shape_check_rel_error_discrete_volume",
    "solve_singular",
    "solve_nonfinite",
    "active_components",
    "floating_components",
    "floating_cells",
    "floating_loaded_components",
    "floating_loaded_cells",
    "anchored_loaded_components",
    "floating_island_components_removed",
    "floating_island_cells_removed",
    "floating_island_dofs_removed",
    "anchored_cells",
    "loaded_cells",
    "topology_invalid",
    "advection_backtracks",
    "min_component_cells",
    "max_component_cells",
    "time_iteration",
    "rss_bytes",
    "tracemalloc_current_bytes",
    "tracemalloc_peak_bytes",
    *(f"time_{phase}" for phase in PHASES),
)


@dataclass
class IterationResult:
    """State retained from one optimization iteration."""

    row: dict[str, float | int]
    displacement: fem.Function
    energy: fem.Function
    extension: NormalExtensionResult
    cut_data: cutfemx.CutData


@dataclass
class OptimizationResult:
    """Final state and scalar history from the demo run."""

    mesh: mesh.Mesh
    phi: fem.Function
    cut_data: cutfemx.CutData
    last_iteration: IterationResult
    history: list[dict[str, float | int]]
    profile_csv: Path


@dataclass
class RieszVelocitySolver:
    """Fixed-background H1 scalar normal-speed smoother for the shape update."""

    space: fem.FunctionSpace
    vector_space: fem.FunctionSpace
    scalar_space: fem.FunctionSpace
    bilinear_form: fem.Form
    matrix: object
    linear_solve: Callable[[np.ndarray], np.ndarray]
    bcs: list[fem.DirichletBC]


@dataclass
class LevelSetAdvectionSolver:
    """SUPG level-set advection solver on the fixed background mesh."""

    space: fem.FunctionSpace
    speed: fem.Function
    dt: fem.Constant
    bcs: list[fem.DirichletBC]
    bilinear_form: fem.Form
    rhs_form: fem.Form


@dataclass
class AugmentedLagrangianState:
    """OptiCut-style ALM state for the volume constraint."""

    lagrange_multiplier: float
    penalty: float
    penalty_limit: float
    penalty_multiplier: float
    slack: float = 0.0


@dataclass(frozen=True)
class ActiveSolidComponent:
    """Facet-connected active-solid component."""

    cells: tuple[int, ...]
    anchored: bool
    loaded: bool


@contextmanager
def _phase(row: dict[str, float | int], name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        key = f"time_{name}"
        row[key] = float(row.get(key, 0.0)) + time.perf_counter() - start


class ProfileWriter:
    """Streaming CSV writer for iteration timings and memory measurements."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=PROFILE_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        return self

    def write(self, row: dict[str, float | int]) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("ProfileWriter is not open.")
        self._writer.writerow({field: row.get(field, "") for field in PROFILE_FIELDS})
        self._file.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._file is not None:
            self._file.close()


def _write_profile_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PROFILE_FIELDS})


def epsilon(u):
    """Small-strain tensor."""
    return ufl.sym(ufl.grad(u))


def sigma(u, mu: float, lmbda: float, gdim: int):
    """Isotropic linear-elastic stress tensor."""
    eps = epsilon(u)
    return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)


def _interpolation_points(V) -> np.ndarray:
    points = V.element.interpolation_points
    return points() if callable(points) else points


def _vector_lagrange_space(msh: mesh.Mesh, degree: int):
    element = basix.ufl.element(
        "Lagrange",
        msh.basix_cell(),
        degree,
        shape=(msh.geometry.dim,),
        dtype=msh.geometry.x.dtype,
    )
    return fem.functionspace(msh, element)


def _make_opticut_level_set(args: argparse.Namespace):
    """OptiCut rectangle benchmark initial level set.

    The sign convention is negative in solid and positive in voids.
    """

    def level_set(x: np.ndarray) -> np.ndarray:
        return (
            -2.0
            * np.cos(6.0 * np.pi * x[0] / args.domain_length)
            * np.cos(4.0 * np.pi * x[1] / args.domain_height)
            - 0.6
        ) / 2.0

    return level_set


def _initial_void_holes(args: argparse.Namespace) -> tuple[tuple[float, float, float], ...]:
    """Return circular voids for the practical initial level set."""
    if args.hole_cols == 1:
        xs = np.array([0.5 * args.domain_length], dtype=np.float64)
    else:
        xs = np.linspace(
            0.15 * args.domain_length,
            0.85 * args.domain_length,
            args.hole_cols,
        )
    if args.hole_rows == 1:
        ys = np.array([0.5 * args.domain_height], dtype=np.float64)
    else:
        ys = np.linspace(
            0.25 * args.domain_height,
            0.75 * args.domain_height,
            args.hole_rows,
        )

    holes: list[tuple[float, float, float]] = []
    for row, y in enumerate(ys):
        if args.hole_cols > 1:
            dx = float(xs[1] - xs[0])
            x_shift = 0.15 * dx if row % 2 else 0.0
        else:
            x_shift = 0.0
        for x in xs:
            x_center = float(
                np.clip(
                    x + x_shift,
                    args.hole_radius,
                    args.domain_length - args.hole_radius,
                )
            )
            y_center = float(
                np.clip(y, args.hole_radius, args.domain_height - args.hole_radius)
            )
            holes.append((x_center, y_center, args.hole_radius))
    return tuple(holes)


def _make_hole_level_set(args: argparse.Namespace):
    """Analytic signed distance to the union of circular voids.

    The sign convention is negative in solid and positive inside holes.
    """
    holes = _initial_void_holes(args)

    def level_set(x: np.ndarray) -> np.ndarray:
        values = np.full_like(x[0], -np.inf, dtype=np.float64)
        for cx, cy, radius in holes:
            distance = np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2)
            values = np.maximum(values, radius - distance)
        return values

    return level_set


def _make_initial_level_set(args: argparse.Namespace):
    if args.initial_geometry == "holes":
        return _make_hole_level_set(args)
    if args.initial_geometry == "opticut":
        return _make_opticut_level_set(args)
    raise ValueError(f"Unknown initial geometry: {args.initial_geometry}")


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _as_3d_vectors(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 3:
        return values
    if values.shape[1] < 3:
        padding = np.zeros((values.shape[0], 3 - values.shape[1]), dtype=values.dtype)
        return np.hstack((values, padding))
    return values[:, :3]


def _function_grid(pyvista, plot, u: fem.Function, name: str):
    cells, cell_types, points = plot.vtk_mesh(u.function_space)
    grid = pyvista.UnstructuredGrid(cells, cell_types, points)
    values = np.asarray(u.x.array, dtype=float)
    block_size = int(getattr(u.function_space.dofmap, "index_map_bs", 1))
    if block_size > 1:
        values = _as_3d_vectors(values.reshape((-1, block_size)))
    if values.shape[0] != grid.n_points:
        raise RuntimeError(
            f"Cannot attach {name}: {values.shape[0]} values for {grid.n_points} points."
        )
    grid.point_data[name] = values
    return grid


def _add_interface_overlay(plotter, pyvista, plot, cut_data, *, color: str) -> None:
    interface_cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi=0", mode="cut_only")
    if interface_cut_mesh.mesh is None:
        return
    interface_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(interface_cut_mesh.mesh))
    plotter.add_mesh(interface_grid, color=color, line_width=5)


def _add_velocity_glyphs(
    plotter,
    pyvista,
    grid,
    vector_name: str,
    *,
    max_glyphs: int = 140,
    factor: float = 0.055,
) -> None:
    if vector_name not in grid.point_data:
        return
    stride = max(1, grid.n_points // max_glyphs)
    points = grid.points[::stride]
    vectors = grid.point_data[vector_name][::stride]
    magnitudes = np.linalg.norm(vectors, axis=1)
    mask = magnitudes > 1.0e-12
    if not np.any(mask):
        return

    glyph_points = pyvista.PolyData(points[mask])
    glyph_points[vector_name] = vectors[mask]
    glyphs = glyph_points.glyph(orient=vector_name, scale=False, factor=factor)
    plotter.add_mesh(glyphs, color="#111827", label=vector_name)


def _plot_iteration_diagnostics(
    msh: mesh.Mesh,
    step: IterationResult,
    *,
    screenshot: Path | None,
    show: bool,
) -> None:
    try:
        import pyvista
        from dolfinx import plot
    except ModuleNotFoundError:
        if msh.comm.rank == 0:
            print("pyvista and dolfinx.plot are required for plotting")
        return

    if msh.comm.rank != 0:
        return

    solid_cut_mesh = cutfemx.create_cut_mesh(step.cut_data, "phi<0", mode="full")
    if solid_cut_mesh.mesh is None:
        return

    displacement_cut = cutfemx.fem.cut_function(step.displacement, solid_cut_mesh)
    displacement_grid = _function_grid(
        pyvista, plot, displacement_cut, "displacement"
    )
    displacement = np.asarray(displacement_cut.x.array, dtype=float).reshape(
        (-1, msh.geometry.dim)
    )
    displacement_grid.point_data["displacement_magnitude"] = np.linalg.norm(
        displacement, axis=1
    )

    max_displacement = float(
        np.max(displacement_grid.point_data["displacement_magnitude"])
    )
    warp_factor = 0.15 / max(max_displacement, 1.0e-14)
    displacement_warped = displacement_grid.warp_by_vector(
        "displacement", factor=warp_factor
    )

    speed_grid = _function_grid(pyvista, plot, step.extension.speed, "update_speed")
    velocity_values = np.asarray(step.extension.velocity.x.array, dtype=float).reshape(
        (-1, msh.geometry.dim)
    )
    if velocity_values.shape[0] == speed_grid.n_points:
        speed_grid.point_data["update_velocity"] = _as_3d_vectors(velocity_values)
        speed_grid.point_data["update_velocity_magnitude"] = np.linalg.norm(
            velocity_values, axis=1
        )

    plotter = pyvista.Plotter(
        shape=(1, 2),
        off_screen=screenshot is not None and not show,
        window_size=(1450, 760),
    )

    row = step.row
    title = (
        f"iter={int(row['iteration'])}  "
        f"J={float(row['compliance']):.3e}  "
        f"V/V0={float(row['volume_fraction']):.3f}  "
        f"lambda={float(row['lambda_volume']):.3e}"
    )

    plotter.subplot(0, 0)
    plotter.add_text("Displacement", position="upper_left", font_size=12)
    plotter.add_mesh(
        displacement_warped,
        scalars="displacement_magnitude",
        cmap="magma",
        show_edges=True,
        edge_color="#cbd5e1",
        line_width=0.25,
        scalar_bar_args={"title": "|u|"},
    )
    _add_interface_overlay(plotter, pyvista, plot, step.cut_data, color="#f8fafc")
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.05)

    plotter.subplot(0, 1)
    plotter.add_text("Update velocity", position="upper_left", font_size=12)
    plotter.add_mesh(
        speed_grid,
        scalars="update_velocity_magnitude",
        cmap="coolwarm",
        show_edges=True,
        edge_color="#cbd5e1",
        line_width=0.25,
        scalar_bar_args={"title": "|V|"},
    )
    _add_interface_overlay(plotter, pyvista, plot, step.cut_data, color="#111827")
    _add_velocity_glyphs(
        plotter,
        pyvista,
        speed_grid,
        "update_velocity",
        factor=0.055,
    )
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.05)
    plotter.add_title(title, font_size=11)

    if screenshot is not None:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(screenshot))
        print(f"Wrote step screenshot to {screenshot}")
    if show:
        plotter.show()
    else:
        plotter.close()


def _solve_runtime_system_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
    bcs: list[fem.DirichletBC],
    row: dict[str, float | int],
) -> tuple[fem.Function, cutfemx.fem.ActiveDomain]:
    """Assemble, deactivate inactive dofs, apply boundary conditions, and solve."""
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    with _phase(row, "assembly"):
        A = cutfemx.fem.assemble_matrix(a, bcs=bcs)
        A.scatter_reverse()
        b = cutfemx.fem.assemble_vector(L)
        cutfemx.fem.apply_lifting(b.array, [a], [bcs])
        b.scatter_reverse(la.InsertMode.add)
        fem.set_bc(b.array, bcs)

        active = cutfemx.fem.active_domain(a)
        cutfemx.fem.deactivate_outside(A, b, active)

    with _phase(row, "solve"):
        uh = fem.Function(V, name="displacement")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MatrixRankWarning)
            uh.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
        row["solve_singular"] = int(
            any(issubclass(item.category, MatrixRankWarning) for item in caught)
        )
        row["solve_nonfinite"] = int(not np.all(np.isfinite(uh.x.array)))
        uh.x.scatter_forward()

    return uh, active


def _compute_energy_density(
    uh: fem.Function,
    mu: float,
    lmbda: float,
    out: fem.Function | None = None,
) -> fem.Function:
    msh = uh.function_space.mesh
    energy = out
    if energy is None:
        Q = fem.functionspace(msh, ("DG", 0))
        energy = fem.Function(Q, name="elastic_energy_density")
    expression = _elastic_energy_density_expr(uh, mu, lmbda)
    energy.interpolate(
        fem.Expression(expression, _interpolation_points(energy.function_space))
    )
    energy.x.scatter_forward()
    return energy


def _elastic_energy_density_expr(
    uh: fem.Function,
    mu: float,
    lmbda: float,
):
    msh = uh.function_space.mesh
    return ufl.inner(sigma(uh, mu, lmbda, msh.geometry.dim), epsilon(uh))


def _extension_max_norm(result: NormalExtensionResult) -> float:
    velocity_values = np.asarray(result.velocity.x.array, dtype=float)
    speed_values = np.asarray(result.speed.x.array, dtype=float)
    gdim = result.velocity.function_space.mesh.geometry.dim
    local_max = 0.0
    if velocity_values.size:
        local_max = float(np.max(np.linalg.norm(velocity_values.reshape((-1, gdim)), axis=1)))
    if speed_values.size:
        local_max = max(local_max, float(np.max(np.abs(speed_values))))
    return float(result.velocity.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX))


def _extension_h1_scale(
    speed: fem.Function,
    *,
    smoothing_length: float,
) -> float:
    msh = speed.function_space.mesh
    value = fem.assemble_scalar(
        fem.form(
            (
                smoothing_length**2
                * ufl.inner(ufl.grad(speed), ufl.grad(speed))
                + speed * speed
            )
            * ufl.dx(domain=msh)
        )
    )
    value = float(msh.comm.allreduce(value, op=MPI.SUM))
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(np.sqrt(value))


def _bounded_velocity_scale(
    h1_scale: float,
    reference_scale: float,
    *,
    floor_fraction: float = VELOCITY_NORMALIZATION_SCALE_FLOOR_FRACTION,
    absolute_floor: float = VELOCITY_NORMALIZATION_ABS_FLOOR,
) -> tuple[float, float, int]:
    """Return the denominator used for velocity normalization.

    The floor is relative to the largest reliable H1 velocity scale seen so
    far. This caps the amplification caused by normalizing a nearly zero
    residual velocity field.
    """
    if not np.isfinite(h1_scale) or h1_scale <= 0.0:
        return 1.0, 0.0, 0
    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        reference_scale = h1_scale
    floor = max(float(absolute_floor), float(floor_fraction) * reference_scale)
    scale = max(float(h1_scale), floor)
    limited = int(scale > h1_scale)
    return scale, floor, limited


def _normalise_extension_velocity(
    result: NormalExtensionResult,
    *,
    scale: float,
) -> tuple[float, float]:
    raw_max = _extension_max_norm(result)
    if np.isfinite(scale) and scale > 0.0 and scale != 1.0:
        result.velocity.x.array[:] /= scale
        result.speed.x.array[:] /= scale
        result.velocity.x.scatter_forward()
        result.speed.x.scatter_forward()
    return float(raw_max), _extension_max_norm(result)


def _function_min_max(function: fem.Function) -> tuple[float, float]:
    values = np.asarray(function.x.array, dtype=float)
    if values.size:
        local_min = float(np.min(values))
        local_max = float(np.max(values))
    else:
        local_min = np.inf
        local_max = -np.inf
    comm = function.function_space.mesh.comm
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    return float(global_min), float(global_max)


def _function_min_abs(function: fem.Function) -> float:
    values = np.asarray(function.x.array, dtype=float)
    local_min = float(np.min(np.abs(values))) if values.size else np.inf
    comm = function.function_space.mesh.comm
    return float(comm.allreduce(local_min, op=MPI.MIN))


def _solid_volume_from_cut_data(
    msh: mesh.Mesh,
    cut_data: cutfemx.CutData,
    order: int,
) -> float:
    solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
    solid_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    if solid_cells.size == 0 and solid_rules.parent_map.size == 0:
        return 0.0
    dx_solid = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[solid_cells, solid_rules],
    )
    value = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form(
            fem.Constant(msh, default_scalar_type(1.0)) * dx_solid
        )
    )
    return float(msh.comm.allreduce(value, op=MPI.SUM))


def _solid_volume(phi: fem.Function, args: argparse.Namespace) -> float:
    cut_data = _cut_level_set(phi, args)
    return _solid_volume_from_cut_data(phi.function_space.mesh, cut_data, args.order)


def _interface_measure_from_cut_data(
    msh: mesh.Mesh,
    cut_data: cutfemx.CutData,
    order: int,
) -> float:
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
    if interface_rules.parent_map.size == 0:
        return 0.0
    dx_interface = ufl.Measure("dx", domain=msh, subdomain_data=interface_rules)
    one = fem.Constant(msh, default_scalar_type(1.0))
    value = cutfemx.fem.assemble_scalar(cutfemx.fem.form(one * dx_interface))
    return float(msh.comm.allreduce(value, op=MPI.SUM))


def _apply_reinit_volume_correction(
    phi: fem.Function,
    args: argparse.Namespace,
    target_volume: float,
    current_volume: float,
) -> tuple[float, float]:
    """Shift signed distance by a constant so redistancing preserves volume."""
    cut_data = _cut_level_set(phi, args)
    interface_measure = _interface_measure_from_cut_data(
        phi.function_space.mesh, cut_data, args.order
    )
    if interface_measure <= 1.0e-14:
        return 0.0, current_volume

    # For phi < 0 in the solid and signed-distance phi, V(phi + c) changes as
    # dV/dc ~= -|Gamma|. A positive shift therefore removes solid volume.
    shift = (current_volume - target_volume) / interface_measure
    if args.reinit_volume_correction_limit > 0.0:
        limit = args.reinit_volume_correction_limit
        shift = float(np.clip(shift, -limit, limit))

    if shift == 0.0:
        return 0.0, current_volume

    phi.x.array[:] += shift
    phi.x.scatter_forward()
    corrected_volume = _solid_volume(phi, args)
    return float(shift), corrected_volume


def _build_riesz_velocity_solver(
    msh: mesh.Mesh,
    smoothing_length: float,
    zero_facets: np.ndarray | None = None,
) -> RieszVelocitySolver:
    from scipy.sparse.linalg import factorized

    Vs = fem.functionspace(msh, ("Lagrange", 1))
    Vv = _vector_lagrange_space(msh, 1)
    speed = ufl.TrialFunction(Vs)
    test = ufl.TestFunction(Vs)
    a = (
        smoothing_length**2 * ufl.inner(ufl.grad(speed), ufl.grad(test))
        + speed * test
    ) * ufl.dx(domain=msh)

    bcs: list[fem.DirichletBC] = []
    if zero_facets is not None and zero_facets.size > 0:
        fdim = msh.topology.dim - 1
        boundary_dofs = fem.locate_dofs_topological(Vs, fdim, zero_facets)
        zero = fem.Function(Vs)
        bcs = [fem.dirichletbc(zero, boundary_dofs)]

    a_form = fem.form(a)
    A = fem.assemble_matrix(a_form, bcs=bcs)
    A.scatter_reverse()
    matrix = A.to_scipy().tocsc()
    return RieszVelocitySolver(
        space=Vs,
        vector_space=Vv,
        scalar_space=Vs,
        bilinear_form=a_form,
        matrix=matrix,
        linear_solve=factorized(matrix),
        bcs=bcs,
    )


def _solve_riesz_rhs(
    solver: RieszVelocitySolver,
    L,
    *,
    name: str,
) -> fem.Function:
    b = cutfemx.fem.assemble_vector(L)
    fem.apply_lifting(b.array, [solver.bilinear_form], [solver.bcs])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, solver.bcs)

    velocity = fem.Function(solver.space, name=name)
    velocity.x.array[:] = solver.linear_solve(b.array)
    velocity.x.scatter_forward()
    return velocity


def _integrate_interface_speed(speed: fem.Function, dx_interface) -> float:
    msh = speed.function_space.mesh
    value = cutfemx.fem.assemble_scalar(cutfemx.fem.form(speed * dx_interface))
    return float(msh.comm.allreduce(value, op=MPI.SUM))


def _build_interface_riesz_forms(
    solver: RieszVelocitySolver,
    compliance_density,
    dx_interface,
) -> tuple[cutfemx.fem.CutForm, cutfemx.fem.CutForm]:
    test = ufl.TestFunction(solver.space)
    shape_rhs = cutfemx.fem.form((compliance_density * test) * dx_interface)
    volume_rhs = cutfemx.fem.form(-test * dx_interface)
    return shape_rhs, volume_rhs


def _build_level_set_advection_solver(
    phi: fem.Function,
    dt: float,
    fixed_facets: np.ndarray,
    tau_scale: float,
) -> LevelSetAdvectionSolver:
    V = phi.function_space
    msh = V.mesh
    phi_new = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    dx_background = ufl.dx(domain=msh)
    speed = fem.Function(V, name="advection_speed")
    dt_constant = fem.Constant(msh, default_scalar_type(dt))

    grad_norm = ufl.sqrt(ufl.inner(ufl.grad(phi), ufl.grad(phi)) + 1.0e-14)
    adv_velocity = speed * ufl.grad(phi) / grad_norm
    adv_norm = ufl.sqrt(ufl.inner(adv_velocity, adv_velocity) + 1.0e-14)
    h = ufl.CellDiameter(msh)
    tau = tau_scale / ufl.sqrt(
        (2.0 / dt_constant) ** 2 + (2.0 * adv_norm / h) ** 2 + 1.0e-30
    )
    streamline_test = ufl.dot(adv_velocity, ufl.grad(test))
    transport = ufl.dot(adv_velocity, ufl.grad(phi_new))

    a = (
        phi_new * test
        + dt_constant * transport * test
        + tau * (phi_new + dt_constant * transport) * streamline_test
    ) * dx_background
    rhs = (phi * test + tau * phi * streamline_test) * dx_background

    bcs: list[fem.DirichletBC] = []
    if fixed_facets.size > 0:
        fdim = msh.topology.dim - 1
        fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
        bcs = [fem.dirichletbc(phi, fixed_dofs)]

    a_form = fem.form(a)
    return LevelSetAdvectionSolver(
        space=V,
        speed=speed,
        dt=dt_constant,
        bcs=bcs,
        bilinear_form=a_form,
        rhs_form=fem.form(rhs),
    )


def _candidate_velocity_scale(
    solver: RieszVelocitySolver,
    values: np.ndarray,
) -> float:
    matrix_values = solver.matrix @ values
    value = float(np.dot(values, matrix_values))
    return float(np.sqrt(max(value, 1.0e-30)))


def _normalised_volume_rate(
    solver: RieszVelocitySolver,
    shape_values: np.ndarray,
    volume_values: np.ndarray,
    rate_shape: float,
    rate_constraint: float,
    lambda_volume: float,
) -> float:
    combined = shape_values + lambda_volume * volume_values
    scale = _candidate_velocity_scale(solver, combined)
    return (rate_shape + lambda_volume * rate_constraint) / scale


def _update_augmented_lagrangian(
    alm: AugmentedLagrangianState,
    constraint: float,
) -> None:
    alm.lagrange_multiplier += alm.penalty * (constraint + alm.slack)
    alm.penalty = min(alm.penalty_limit, alm.penalty_multiplier * alm.penalty)


def _alm_velocity_multiplier(
    alm: AugmentedLagrangianState,
    constraint: float,
) -> float:
    return alm.lagrange_multiplier + alm.penalty * (constraint + alm.slack)


def _lagrangian_value(
    objective: float,
    constraint: float,
    alm: AugmentedLagrangianState,
) -> float:
    constrained_value = constraint + alm.slack
    return (
        objective
        + alm.lagrange_multiplier * constrained_value
        + 0.5 * alm.penalty * constrained_value**2
    )


def _initialise_augmented_lagrangian_scale(
    alm: AugmentedLagrangianState,
    objective: float,
    constraint: float,
) -> None:
    """Scale ALM parameters from the initial objective and volume violation."""
    if (
        not np.isfinite(objective)
        or not np.isfinite(constraint)
        or objective <= 0.0
        or abs(constraint) <= 1.0e-14
    ):
        return

    alm.lagrange_multiplier = objective / constraint
    alm.penalty = objective / (constraint**2)
    alm.penalty_limit = max(alm.penalty, 10.0 * alm.penalty)


def _choose_volume_multiplier(
    solver: RieszVelocitySolver,
    shape_values: np.ndarray,
    volume_values: np.ndarray,
    rate_shape: float,
    rate_constraint: float,
    target_rate: float,
    lambda_limit: float,
) -> float:
    if not np.isfinite(rate_constraint) or abs(rate_constraint) <= 1.0e-14:
        return 0.0
    direct = (target_rate - rate_shape) / rate_constraint
    if not np.isfinite(direct):
        return 0.0

    def residual(lambda_value: float) -> float:
        return (
            _normalised_volume_rate(
                solver,
                shape_values,
                volume_values,
                rate_shape,
                rate_constraint,
                lambda_value,
            )
            - target_rate
        )

    candidates = [0.0, direct]
    if lambda_limit > 0.0:
        candidates.extend((-lambda_limit, lambda_limit))
        search_bounds = (-lambda_limit, lambda_limit)
    else:
        scale = max(1.0, abs(direct))
        candidates.extend((-scale, scale))
        search_bounds = None

    best_lambda = min(candidates, key=lambda value: abs(residual(value)))
    best_residual = abs(residual(best_lambda))
    bracket: tuple[float, float] | None = None

    if search_bounds is not None:
        lo, hi = search_bounds
        flo = residual(lo)
        fhi = residual(hi)
        if flo == 0.0:
            return float(lo)
        if fhi == 0.0:
            return float(hi)
        if flo * fhi < 0.0:
            bracket = (lo, hi)
    else:
        previous = candidates[0]
        previous_residual = residual(previous)
        scale = max(1.0, abs(direct))
        for _ in range(60):
            for current in (-scale, scale):
                current_residual = residual(current)
                if abs(current_residual) < best_residual:
                    best_lambda = current
                    best_residual = abs(current_residual)
                if previous_residual == 0.0:
                    return float(previous)
                if current_residual == 0.0:
                    return float(current)
                if previous_residual * current_residual < 0.0:
                    bracket = (previous, current)
                    break
                previous = current
                previous_residual = current_residual
            if bracket is not None:
                break
            scale *= 2.0

    if bracket is None:
        return float(best_lambda)

    lo, hi = bracket
    flo = residual(lo)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fmid = residual(mid)
        if abs(fmid) < 1.0e-10:
            return float(mid)
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return float(0.5 * (lo + hi))


def _solve_riesz_velocity_volume(
    solver: RieszVelocitySolver,
    phi: fem.Function,
    shape_rhs: fem.Form,
    volume_rhs: fem.Form,
    dx_interface,
    *,
    volume_multiplier: float | None,
    target_rate: float,
    lambda_limit: float,
) -> tuple[NormalExtensionResult, float, dict[str, float]]:
    """Cut-interface compliance derivative projected by an H1 Riesz solve."""
    speed_shape = _solve_riesz_rhs(solver, shape_rhs, name="riesz_shape_speed")
    speed_volume = _solve_riesz_rhs(solver, volume_rhs, name="riesz_volume_speed")
    # With phi < 0 in the solid and phi <- phi - dt*s, positive speed moves
    # the zero contour toward increasing phi and increases solid volume.
    rate_shape = _integrate_interface_speed(speed_shape, dx_interface)
    rate_constraint = _integrate_interface_speed(speed_volume, dx_interface)
    shape_values = np.asarray(speed_shape.x.array, dtype=float)
    volume_values = np.asarray(speed_volume.x.array, dtype=float)
    if volume_multiplier is None:
        lambda_volume = _choose_volume_multiplier(
            solver,
            shape_values,
            volume_values,
            rate_shape,
            rate_constraint,
            target_rate,
            lambda_limit,
        )
    else:
        lambda_volume = float(volume_multiplier)

    speed = fem.Function(solver.scalar_space, name="riesz_normal_speed")
    speed.x.array[:] = shape_values + lambda_volume * volume_values
    speed.x.scatter_forward()
    rate_predicted = rate_shape + lambda_volume * rate_constraint

    velocity = fem.Function(solver.vector_space, name="riesz_velocity")
    grad_phi = ufl.grad(phi)
    normal = grad_phi / ufl.sqrt(ufl.inner(grad_phi, grad_phi) + 1.0e-14)
    velocity.interpolate(
        fem.Expression(speed * normal, _interpolation_points(solver.vector_space))
    )
    velocity.x.scatter_forward()

    signed_distance = fem.Function(solver.scalar_space, name="riesz_signed_distance")
    signed_distance.interpolate(phi)
    signed_distance.x.scatter_forward()
    rates = {
        "volume_rate_shape": float(rate_shape),
        "volume_rate_constraint": float(rate_constraint),
        "volume_rate_predicted": float(rate_predicted),
    }
    return NormalExtensionResult(speed, velocity, signed_distance), float(lambda_volume), rates


def _evaluate_state_objectives(
    phi: fem.Function,
    args: argparse.Namespace,
    V: fem.FunctionSpace,
    u,
    v,
    bcs: list[fem.DirichletBC],
    ds_load,
    traction: fem.Constant,
    mu: float,
    lambda_lame: float,
) -> dict[str, float]:
    """Solve elasticity on a trial level set and return objective values."""
    msh = phi.function_space.mesh
    cut_data = _cut_level_set(phi, args)
    solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
    solid_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", args.order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
    dx_solid = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[solid_cells, solid_rules],
    )
    dS_ghost = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=1,
        subdomain_data=ghost_facets,
    )

    a_expr = ufl.inner(sigma(u, mu, lambda_lame, msh.geometry.dim), epsilon(v)) * dx_solid
    if ghost_facets.size > 0 and args.ghost_gamma != 0.0:
        n_facet = ufl.FacetNormal(msh)
        h_avg = ufl.avg(ufl.CellDiameter(msh))
        a_expr += (
            args.ghost_gamma
            * (2.0 * mu + lambda_lame)
            * h_avg**3
            * ufl.inner(
                ufl.jump(ufl.grad(u), n_facet),
                ufl.jump(ufl.grad(v), n_facet),
            )
            * dS_ghost
        )
    L_expr = ufl.inner(traction, v) * ds_load(1)

    probe_row: dict[str, float | int] = {
        "solve_singular": 0,
        "solve_nonfinite": 0,
    }
    for phase in PHASES:
        probe_row[f"time_{phase}"] = 0.0

    uh, _ = _solve_runtime_system_scipy(
        cutfemx.fem.form(a_expr),
        cutfemx.fem.form(L_expr),
        V,
        bcs,
        probe_row,
    )
    compliance = fem.assemble_scalar(fem.form(ufl.inner(traction, uh) * ds_load(1)))
    compliance_density = _elastic_energy_density_expr(uh, mu, lambda_lame)
    elastic_energy = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form(compliance_density * dx_solid)
    )
    compliance = float(msh.comm.allreduce(compliance, op=MPI.SUM))
    elastic_energy = float(msh.comm.allreduce(elastic_energy, op=MPI.SUM))
    return {
        "compliance": compliance,
        "half_energy": 0.5 * elastic_energy,
        "solve_singular": float(probe_row["solve_singular"]),
        "solve_nonfinite": float(probe_row["solve_nonfinite"]),
    }


def _shape_derivative_check(
    phi: fem.Function,
    args: argparse.Namespace,
    extension: NormalExtensionResult,
    compliance_density,
    dx_interface,
    V: fem.FunctionSpace,
    u,
    v,
    bcs: list[fem.DirichletBC],
    ds_load,
    traction: fem.Constant,
    mu: float,
    lambda_lame: float,
    current_half_energy: float,
    current_compliance: float,
    current_volume: float,
) -> dict[str, float]:
    """Finite-difference check of the current shape derivative direction."""
    msh = phi.function_space.mesh
    step = args.shape_check_step

    predicted_half_energy = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form((-0.5 * compliance_density) * extension.speed * dx_interface)
    )
    predicted_half_energy = float(
        msh.comm.allreduce(predicted_half_energy, op=MPI.SUM)
    )
    predicted_compliance = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form((-compliance_density) * extension.speed * dx_interface)
    )
    predicted_compliance = float(msh.comm.allreduce(predicted_compliance, op=MPI.SUM))
    predicted_volume = _integrate_interface_speed(extension.speed, dx_interface)

    trial_phi = fem.Function(phi.function_space, name="phi")
    grad_phi = ufl.grad(phi)
    grad_norm = ufl.sqrt(ufl.inner(grad_phi, grad_phi) + 1.0e-14)
    trial_phi.interpolate(
        fem.Expression(
            phi - step * extension.speed * grad_norm,
            _interpolation_points(phi.function_space),
        )
    )
    trial_phi.x.scatter_forward()
    predicted_discrete_volume = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form(
            ((phi - trial_phi) / (step * grad_norm)) * dx_interface
        )
    )
    predicted_discrete_volume = float(
        msh.comm.allreduce(predicted_discrete_volume, op=MPI.SUM)
    )

    trial = _evaluate_state_objectives(
        trial_phi,
        args,
        V,
        u,
        v,
        bcs,
        ds_load,
        traction,
        mu,
        lambda_lame,
    )
    fd_half_energy = (trial["half_energy"] - current_half_energy) / step
    fd_compliance = (trial["compliance"] - current_compliance) / step
    trial_volume = _solid_volume(trial_phi, args)
    fd_volume = (trial_volume - current_volume) / step

    denominator = max(abs(fd_compliance), abs(predicted_compliance), 1.0e-14)
    rel_error = abs(fd_compliance - predicted_compliance) / denominator
    denominator_half = max(abs(fd_half_energy), abs(predicted_half_energy), 1.0e-14)
    rel_error_half = (
        abs(fd_half_energy - predicted_half_energy) / denominator_half
    )
    denominator_volume = max(abs(fd_volume), abs(predicted_volume), 1.0e-14)
    rel_error_volume = abs(fd_volume - predicted_volume) / denominator_volume
    denominator_discrete_volume = max(
        abs(fd_volume), abs(predicted_discrete_volume), 1.0e-14
    )
    rel_error_discrete_volume = (
        abs(fd_volume - predicted_discrete_volume) / denominator_discrete_volume
    )
    if trial["solve_singular"] or trial["solve_nonfinite"]:
        rel_error = np.nan
        rel_error_half = np.nan

    return {
        "shape_check_step": float(step),
        "shape_check_predicted": predicted_compliance,
        "shape_check_predicted_half_energy": predicted_half_energy,
        "shape_check_predicted_volume": predicted_volume,
        "shape_check_predicted_discrete_volume": predicted_discrete_volume,
        "shape_check_fd_half_energy": float(fd_half_energy),
        "shape_check_fd_compliance": float(fd_compliance),
        "shape_check_fd_volume": float(fd_volume),
        "shape_check_rel_error": float(rel_error),
        "shape_check_rel_error_half_energy": float(rel_error_half),
        "shape_check_rel_error_volume": float(rel_error_volume),
        "shape_check_rel_error_discrete_volume": float(rel_error_discrete_volume),
    }


def _advect_level_set_supg(
    solver: LevelSetAdvectionSolver,
    phi: fem.Function,
    extension: NormalExtensionResult,
    dt: float,
) -> None:
    from scipy.sparse.linalg import spsolve

    solver.dt.value = default_scalar_type(dt)
    solver.speed.x.array[:] = extension.speed.x.array
    solver.speed.x.scatter_forward()

    A = fem.assemble_matrix(solver.bilinear_form, bcs=solver.bcs)
    A.scatter_reverse()
    b = fem.assemble_vector(solver.rhs_form)
    fem.apply_lifting(b.array, [solver.bilinear_form], [solver.bcs])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, solver.bcs)

    phi.x.array[:] = spsolve(A.to_scipy().tocsr(), b.array)
    phi.x.scatter_forward()


def _write_history_plot(path: Path, history: list[dict[str, float | int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [int(row["iteration"]) for row in history]
    compliance = [float(row["compliance"]) for row in history]
    volume_fraction = [float(row["volume_fraction"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].plot(iterations, compliance, marker="o")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("compliance")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(iterations, volume_fraction, marker="o", color="#2563eb")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("volume fraction")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_result(
    result: OptimizationResult,
    *,
    screenshot: Path | None,
    show: bool,
) -> None:
    _plot_iteration_diagnostics(
        result.mesh,
        result.last_iteration,
        screenshot=screenshot,
        show=show,
    )


def _write_xdmf_snapshot(
    msh: mesh.Mesh,
    phi: fem.Function,
    step: IterationResult,
    output_dir: Path,
    iteration: int,
    *,
    write_vtk: bool = False,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    """Write background and current solid cut-mesh fields for one iteration.

    XDMF is kept for compatibility with the benchmark output.  A matching VTK
    snapshot can also be written because ParaView handles these single-grid
    files more predictably when coloring by fields or applying warp filters.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    time_value = float(iteration)

    background_path = output_dir / f"background_{iteration:04d}.xdmf"
    with io.XDMFFile(msh.comm, background_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(phi, time_value)
        xdmf.write_function(step.extension.speed, time_value)
        xdmf.write_function(step.extension.velocity, time_value)
    background_vtk_path: Path | None = None
    if write_vtk:
        background_vtk_path = output_dir / f"background_{iteration:04d}.pvd"
        with io.VTKFile(msh.comm, background_vtk_path.as_posix(), "w") as vtk:
            vtk.write_mesh(msh, time_value)
            vtk.write_function(
                [phi, step.extension.speed, step.extension.velocity],
                time_value,
            )

    solid_cut_mesh = cutfemx.create_cut_mesh(step.cut_data, "phi<0", mode="full")
    if solid_cut_mesh.mesh is None:
        return background_path, None, background_vtk_path, None

    phi_cut = cutfemx.fem.cut_function(phi, solid_cut_mesh)
    phi_cut.name = "phi"
    displacement_cut = cutfemx.fem.cut_function(step.displacement, solid_cut_mesh)
    displacement_cut.name = "displacement"
    displacement_values = np.asarray(displacement_cut.x.array, dtype=float).reshape(
        (-1, msh.geometry.dim)
    )
    displacement_magnitude = fem.Function(
        fem.functionspace(solid_cut_mesh.mesh, ("Lagrange", 1)),
        name="displacement_magnitude",
    )
    displacement_magnitude.x.array[:] = np.linalg.norm(displacement_values, axis=1)
    displacement_magnitude.x.scatter_forward()
    energy_cut = cutfemx.fem.cut_function(step.energy, solid_cut_mesh)
    energy_cut.name = "elastic_energy_density"

    solid_path = output_dir / f"solid_{iteration:04d}.xdmf"
    with io.XDMFFile(solid_cut_mesh.mesh.comm, solid_path.as_posix(), "w") as xdmf:
        xdmf.write_mesh(solid_cut_mesh.mesh)
        xdmf.write_function(phi_cut, time_value)
        xdmf.write_function(displacement_cut, time_value)
        xdmf.write_function(displacement_magnitude, time_value)
        xdmf.write_function(energy_cut, time_value)

    solid_vtk_path: Path | None = None
    if write_vtk:
        solid_vtk_path = output_dir / f"solid_{iteration:04d}.pvd"
        with io.VTKFile(solid_cut_mesh.mesh.comm, solid_vtk_path.as_posix(), "w") as vtk:
            vtk.write_mesh(solid_cut_mesh.mesh, time_value)
            vtk.write_function(
                [phi_cut, displacement_cut, displacement_magnitude, energy_cut],
                time_value,
            )

    return background_path, solid_path, background_vtk_path, solid_vtk_path


def _write_tracemalloc_report(
    output_dir: Path,
    iteration: int,
    baseline,
    *,
    limit: int = 25,
) -> None:
    """Write current and baseline-delta Python allocation hot spots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = tracemalloc.take_snapshot()
    path = output_dir / f"tracemalloc_{iteration:04d}.txt"
    total = sum(stat.size for stat in snapshot.statistics("filename"))

    with path.open("w") as file:
        file.write(f"iteration: {iteration}\n")
        file.write(f"total_traced_bytes: {total}\n\n")
        file.write(f"top {limit} current allocations by line:\n")
        for stat in snapshot.statistics("lineno")[:limit]:
            file.write(f"{stat}\n")

        if baseline is not None:
            file.write(f"\ntop {limit} allocation growth since baseline:\n")
            for stat in snapshot.compare_to(baseline, "lineno")[:limit]:
                file.write(f"{stat}\n")


def _horizontal_cells(args: argparse.Namespace) -> int:
    return max(1, int(round(args.domain_length / args.domain_height * args.n)))


def _build_mesh(comm, args: argparse.Namespace) -> mesh.Mesh:
    nx = _horizontal_cells(args)
    cell_type = (
        mesh.CellType.triangle
        if args.cell_type == "triangle"
        else mesh.CellType.quadrilateral
    )
    return mesh.create_rectangle(
        comm,
        (
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([args.domain_length, args.domain_height], dtype=np.float64),
        ),
        (nx, args.n),
        cell_type=cell_type,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


def _build_boundary_data(msh: mesh.Mesh, args: argparse.Namespace):
    fdim = msh.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0)
    )
    right_boundary_facets = mesh.locate_entities_boundary(
        msh,
        fdim,
        lambda x: np.isclose(x[0], args.domain_length),
    )
    right_midpoints = mesh.compute_midpoints(msh, fdim, right_boundary_facets)
    in_load_patch = np.logical_and(
        right_midpoints[:, 1] > args.load_center - args.load_half_height,
        right_midpoints[:, 1] < args.load_center + args.load_half_height,
    )
    right_facets = right_boundary_facets[in_load_patch]
    order = np.argsort(right_facets)
    right_tags = mesh.meshtags(
        msh,
        fdim,
        right_facets[order],
        np.full(right_facets.size, 1, dtype=np.int32)[order],
    )
    return fdim, left_facets, right_tags


def _cut_level_set(phi: fem.Function, args: argparse.Namespace) -> cutfemx.CutData:
    return cutfemx.cut(
        phi,
        cut_approximation=args.cut_approximation,
        cut_approximation_order=args.cut_approximation_order,
        max_refinement_iterations=args.max_refinement_iterations,
        edge_max_depth=args.edge_max_depth,
    )


def _active_solid_components(
    msh: mesh.Mesh,
    solid_cells: np.ndarray,
    cut_parent_cells: np.ndarray,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
) -> list[ActiveSolidComponent]:
    """Return facet-connected active-solid components."""
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    cell_to_facets = msh.topology.connectivity(tdim, fdim)
    facet_to_cells = msh.topology.connectivity(fdim, tdim)
    if cell_to_facets is None or facet_to_cells is None:
        raise RuntimeError("Missing cell/facet connectivity for topology diagnostics.")

    if solid_cells.size + cut_parent_cells.size == 0:
        return []

    active_cells = np.unique(np.concatenate((solid_cells, cut_parent_cells))).astype(
        np.int32,
        copy=False,
    )
    active = {int(cell) for cell in active_cells}
    left = {int(facet) for facet in left_facets}
    load = {int(facet) for facet in load_facets}

    visited: set[int] = set()
    components: list[ActiveSolidComponent] = []

    for start in active_cells:
        start_cell = int(start)
        if start_cell in visited:
            continue

        stack = [start_cell]
        visited.add(start_cell)
        component_cells: list[int] = []
        anchored = False
        loaded = False

        while stack:
            cell = stack.pop()
            component_cells.append(cell)
            facets = cell_to_facets.links(cell)
            if any(int(facet) in left for facet in facets):
                anchored = True
            if any(int(facet) in load for facet in facets):
                loaded = True

            for facet in facets:
                for neighbor in facet_to_cells.links(int(facet)):
                    neighbor_cell = int(neighbor)
                    if neighbor_cell in active and neighbor_cell not in visited:
                        visited.add(neighbor_cell)
                        stack.append(neighbor_cell)

        components.append(
            ActiveSolidComponent(
                cells=tuple(sorted(component_cells)),
                anchored=anchored,
                loaded=loaded,
            )
        )

    return components


def _active_cell_diagnostics_from_components(
    components: list[ActiveSolidComponent],
) -> dict[str, int]:
    """Count anchored, loaded, and floating active-solid components."""
    if len(components) == 0:
        return {
            "active_components": 0,
            "floating_components": 0,
            "floating_cells": 0,
            "floating_loaded_components": 0,
            "floating_loaded_cells": 0,
            "anchored_loaded_components": 0,
            "anchored_cells": 0,
            "loaded_cells": 0,
            "min_component_cells": 0,
            "max_component_cells": 0,
        }

    component_sizes = [len(component.cells) for component in components]
    floating = [component for component in components if not component.anchored]
    floating_loaded = [
        component for component in floating if component.loaded
    ]
    anchored_loaded = [
        component for component in components if component.anchored and component.loaded
    ]

    return {
        "active_components": len(components),
        "floating_components": len(floating),
        "floating_cells": sum(len(component.cells) for component in floating),
        "floating_loaded_components": len(floating_loaded),
        "floating_loaded_cells": sum(
            len(component.cells) for component in floating_loaded
        ),
        "anchored_loaded_components": len(anchored_loaded),
        "anchored_cells": sum(
            len(component.cells) for component in components if component.anchored
        ),
        "loaded_cells": sum(
            len(component.cells) for component in components if component.loaded
        ),
        "min_component_cells": min(component_sizes),
        "max_component_cells": max(component_sizes),
    }


def _active_cell_diagnostics(
    msh: mesh.Mesh,
    solid_cells: np.ndarray,
    cut_parent_cells: np.ndarray,
    anchor_facets: np.ndarray,
    load_facets: np.ndarray,
) -> dict[str, int]:
    """Count anchored, loaded, and floating active-solid components."""
    return _active_cell_diagnostics_from_components(
        _active_solid_components(
            msh, solid_cells, cut_parent_cells, anchor_facets, load_facets
        )
    )


def _remove_unloaded_floating_components(
    phi: fem.Function,
    components: list[ActiveSolidComponent],
    clear_value: float,
) -> np.ndarray:
    """Remove unloaded floating islands by making their local vertices void."""
    msh = phi.function_space.mesh
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, 0)
    cell_to_vertices = msh.topology.connectivity(tdim, 0)
    if cell_to_vertices is None:
        raise RuntimeError("Missing cell/vertex connectivity.")

    floating = [
        component
        for component in components
        if not component.anchored and not component.loaded
    ]
    if len(floating) == 0:
        return np.empty(0, dtype=np.int32)

    protected_cells = {
        int(cell)
        for component in components
        if component.anchored or component.loaded
        for cell in component.cells
    }
    protected_vertices = {
        int(vertex)
        for cell in protected_cells
        for vertex in cell_to_vertices.links(cell)
    }
    remove_vertices: set[int] = set()

    for component in floating:
        component_cells = {int(cell) for cell in component.cells}
        component_vertices = {
            int(vertex)
            for cell in component_cells
            for vertex in cell_to_vertices.links(cell)
        }
        local_vertices = component_vertices - protected_vertices
        remove_vertices.update(local_vertices if local_vertices else component_vertices)

    vertices = np.asarray(sorted(remove_vertices), dtype=np.int32)
    if vertices.size == 0:
        return np.empty(0, dtype=np.int32)
    dofs = fem.locate_dofs_topological(phi.function_space, 0, vertices)
    if dofs.size > 0:
        phi.x.array[dofs] = np.maximum(phi.x.array[dofs], clear_value)
        phi.x.scatter_forward()
    return np.asarray(dofs, dtype=np.int32)


def _level_set_topology_diagnostics(
    phi: fem.Function,
    args: argparse.Namespace,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
) -> dict[str, int]:
    """Return active-component diagnostics for the current level set."""
    components = _level_set_components(phi, args, left_facets, load_facets)
    return _active_cell_diagnostics_from_components(components)


def _level_set_components(
    phi: fem.Function,
    args: argparse.Namespace,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
) -> list[ActiveSolidComponent]:
    """Return active-solid components for the current level set."""
    cut_data = _cut_level_set(phi, args)
    solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
    solid_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", args.order)
    return _active_solid_components(
        phi.function_space.mesh,
        np.asarray(solid_cells, dtype=np.int32),
        np.asarray(solid_rules.parent_map, dtype=np.int32),
        np.asarray(left_facets, dtype=np.int32),
        np.asarray(load_facets, dtype=np.int32),
    )


def _cleanup_unloaded_floating_islands(
    phi: fem.Function,
    args: argparse.Namespace,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
    clear_value: float,
    max_passes: int = 4,
) -> tuple[dict[str, int], int]:
    """Delete unloaded floating islands and return updated diagnostics."""
    total_dofs = 0
    diagnostics: dict[str, int] | None = None
    for _ in range(max_passes):
        components = _level_set_components(phi, args, left_facets, load_facets)
        diagnostics = _active_cell_diagnostics_from_components(components)
        if (
            diagnostics["floating_loaded_components"] > 0
            or diagnostics["floating_components"] == 0
        ):
            return diagnostics, total_dofs

        dofs = _remove_unloaded_floating_components(phi, components, clear_value)
        if dofs.size == 0:
            return diagnostics, total_dofs
        total_dofs += int(dofs.size)
        reinitialize(phi, max_iter=args.reinit_max_iter, tol=args.reinit_tol)
        phi.x.scatter_forward()

    components = _level_set_components(phi, args, left_facets, load_facets)
    diagnostics = _active_cell_diagnostics_from_components(components)
    return diagnostics, total_dofs


def run_optimization(args: argparse.Namespace) -> OptimizationResult:
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("demo_compliance_optimization.py currently uses a serial SciPy solve.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = args.profile_csv or (args.output_dir / "profile.csv")
    profile_csv = Path(profile_csv)
    xdmf_dir = args.xdmf_dir or (args.output_dir / "xdmf")
    tracemalloc_dir = args.tracemalloc_dir or (args.output_dir / "tracemalloc")

    msh = _build_mesh(comm, args)
    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi, name="phi")
    phi.interpolate(_make_initial_level_set(args))
    phi.x.scatter_forward()

    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    energy_field = fem.Function(
        fem.functionspace(msh, ("DG", 0)),
        name="elastic_energy_density",
    )
    fdim, left_facets, right_tags = _build_boundary_data(msh, args)
    left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
    zero_u = fem.Function(V)
    bcs = [fem.dirichletbc(zero_u, left_dofs)]

    ds_load = ufl.Measure("ds", domain=msh, subdomain_data=right_tags)
    traction = fem.Constant(msh, default_scalar_type((0.0, -args.traction)))

    young_modulus = args.young_modulus
    poisson_ratio = args.poisson_ratio
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lmbda = young_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )

    nx = _horizontal_cells(args)
    hmin = min(args.domain_length / nx, args.domain_height / args.n)
    island_clear_value = 0.25 * hmin
    if args.initial_reinit:
        reinitialize(phi, max_iter=args.reinit_max_iter, tol=args.reinit_tol)
        phi.x.scatter_forward()
    initial_diagnostics, initial_removed_dofs = _cleanup_unloaded_floating_islands(
        phi,
        args,
        np.asarray(left_facets, dtype=np.int32),
        np.asarray(right_tags.indices, dtype=np.int32),
        island_clear_value,
    )
    if initial_diagnostics["floating_loaded_components"] > 0:
        raise RuntimeError(
            "Initial level set has a loaded active component disconnected from "
            "the clamped boundary."
        )
    if comm.rank == 0 and initial_removed_dofs > 0:
        print(
            "Removed unloaded floating islands from the initial level set "
            f"({initial_removed_dofs} scalar dofs)."
        )
    dt = args.dt if args.dt > 0.0 else args.cfl * hmin
    smoothing_length = args.velocity_smoothing
    if smoothing_length is None:
        smoothing_length = float(np.sqrt(args.velocity_alpha))
    fixed_update_facets = np.unique(
        np.concatenate(
            (
                np.asarray(left_facets, dtype=np.int32),
                np.asarray(right_tags.indices, dtype=np.int32),
            )
        )
    )
    riesz_solver = _build_riesz_velocity_solver(
        msh,
        smoothing_length,
        zero_facets=fixed_update_facets,
    )
    advection_solver = _build_level_set_advection_solver(
        phi,
        dt,
        fixed_update_facets,
        args.supg_tau_scale,
    )
    reference_volume = args.domain_length * args.domain_height
    target_volume = args.volume_fraction * reference_volume
    alm = AugmentedLagrangianState(
        lagrange_multiplier=args.alm_lambda,
        penalty=args.alm_penalty,
        penalty_limit=args.alm_penalty_limit,
        penalty_multiplier=args.alm_penalty_multiplier,
    )
    history: list[dict[str, float | int]] = []
    last_iteration: IterationResult | None = None
    alm_auto_scaled = False
    velocity_scale_reference = 0.0

    tracemalloc.start()
    tracemalloc_baseline = (
        tracemalloc.take_snapshot() if args.tracemalloc_interval > 0 else None
    )
    with ProfileWriter(profile_csv) as profile:
        for iteration in range(args.iterations):
            iteration_start = time.perf_counter()
            row: dict[str, float | int] = {
                "iteration": iteration,
                "cells": msh.topology.index_map(msh.topology.dim).size_local,
                "target_volume": target_volume,
                "volume_constraint": 0.0,
                "dt": dt,
                "lagrangian": 0.0,
                "alm_lambda": alm.lagrange_multiplier,
                "alm_penalty": alm.penalty,
                "alm_multiplier": 0.0,
                "solve_singular": 0,
                "solve_nonfinite": 0,
                "active_components": 0,
                "floating_components": 0,
                "floating_cells": 0,
                "floating_loaded_components": 0,
                "floating_loaded_cells": 0,
                "anchored_loaded_components": 0,
                "floating_island_components_removed": 0,
                "floating_island_cells_removed": 0,
                "floating_island_dofs_removed": 0,
                "anchored_cells": 0,
                "loaded_cells": 0,
                "topology_invalid": 0,
                "advection_backtracks": 0,
                "min_component_cells": 0,
                "max_component_cells": 0,
                "volume_rate_target": 0.0,
                "volume_rate_shape": 0.0,
                "volume_rate_constraint": 0.0,
                "volume_rate_predicted": 0.0,
                "volume_after_advection": 0.0,
                "volume_after_raw_reinitialization": 0.0,
                "volume_after_reinitialization": 0.0,
                "reinit_volume_shift": 0.0,
                "raw_velocity_max": 0.0,
                "velocity_h1_scale": 0.0,
                "velocity_scale_floor": 0.0,
                "velocity_scale": 1.0,
                "velocity_scale_limited": 0,
                "velocity_max": 0.0,
                "speed_min": 0.0,
                "speed_max": 0.0,
                "phi_min": 0.0,
                "phi_max": 0.0,
                "phi_min_abs_dof": 0.0,
                "supg_tau_scale": args.supg_tau_scale,
                "time_iteration": 0.0,
            }
            for phase in PHASES:
                row[f"time_{phase}"] = 0.0

            with _phase(row, "cut"):
                for _cleanup_attempt in range(3):
                    cut_data = _cut_level_set(phi, args)
                    row["phi_min_abs_dof"] = _function_min_abs(phi)
                    solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
                    solid_rules = cutfemx.runtime_quadrature(
                        cut_data, "phi<0", args.order
                    )
                    interface_rules = cutfemx.runtime_quadrature(
                        cut_data, "phi=0", args.order
                    )
                    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
                    row["solid_cells"] = int(solid_cells.size)
                    row["cut_cells"] = int(interface_rules.parent_map.size)
                    row["ghost_facets"] = int(ghost_facets.size)
                    components = _active_solid_components(
                        msh,
                        np.asarray(solid_cells, dtype=np.int32),
                        np.asarray(solid_rules.parent_map, dtype=np.int32),
                        np.asarray(left_facets, dtype=np.int32),
                        np.asarray(right_tags.indices, dtype=np.int32),
                    )
                    component_diagnostics = (
                        _active_cell_diagnostics_from_components(components)
                    )
                    if args.topology_diagnostics:
                        row.update(component_diagnostics)
                    row["topology_invalid"] = int(
                        component_diagnostics["floating_components"] > 0
                    )
                    if (
                        component_diagnostics["floating_components"] == 0
                        or component_diagnostics["floating_loaded_components"] > 0
                    ):
                        break

                    row["floating_island_components_removed"] += int(
                        component_diagnostics["floating_components"]
                    )
                    row["floating_island_cells_removed"] += int(
                        component_diagnostics["floating_cells"]
                    )
                    removed_dofs = _remove_unloaded_floating_components(
                        phi, components, island_clear_value
                    )
                    row["floating_island_dofs_removed"] += int(removed_dofs.size)
                    if removed_dofs.size == 0:
                        break
                    reinitialize(phi, max_iter=args.reinit_max_iter, tol=args.reinit_tol)
                    phi.x.scatter_forward()
                elasticity_bcs = bcs

            with _phase(row, "forms"):
                dx_solid = ufl.Measure(
                    "dx",
                    domain=msh,
                    subdomain_id=0,
                    subdomain_data=[solid_cells, solid_rules],
                )
                dx_interface = ufl.Measure(
                    "dx",
                    domain=msh,
                    subdomain_id=1,
                    subdomain_data=interface_rules,
                )
                dS_ghost = ufl.Measure(
                    "dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets
                )
                a_expr = ufl.inner(sigma(u, mu, lmbda, msh.geometry.dim), epsilon(v)) * dx_solid
                if ghost_facets.size > 0 and args.ghost_gamma != 0.0:
                    n_facet = ufl.FacetNormal(msh)
                    h_avg = ufl.avg(ufl.CellDiameter(msh))
                    a_expr += (
                        args.ghost_gamma
                        * (2.0 * mu + lmbda)
                        * h_avg**3
                        * ufl.inner(
                            ufl.jump(ufl.grad(u), n_facet),
                            ufl.jump(ufl.grad(v), n_facet),
                        )
                        * dS_ghost
                    )
                L_expr = ufl.inner(traction, v) * ds_load(1)
                a_form = cutfemx.fem.form(a_expr)
                L_form = cutfemx.fem.form(L_expr)

            uh, active = _solve_runtime_system_scipy(
                a_form, L_form, V, elasticity_bcs, row
            )
            row["inactive_dofs"] = int(active.inactive_dofs.size)

            with _phase(row, "sensitivity"):
                compliance = fem.assemble_scalar(fem.form(ufl.inner(traction, uh) * ds_load(1)))
                volume = cutfemx.fem.assemble_scalar(
                    cutfemx.fem.form(fem.Constant(msh, default_scalar_type(1.0)) * dx_solid)
                )
                compliance = float(comm.allreduce(compliance, op=MPI.SUM))
                volume = float(comm.allreduce(volume, op=MPI.SUM))
                row["compliance"] = compliance
                row["volume"] = volume
                row["volume_fraction"] = volume / reference_volume
                volume_constraint = volume - target_volume
                row["volume_constraint"] = volume_constraint

                compliance_density = _elastic_energy_density_expr(uh, mu, lmbda)
                energy = _compute_energy_density(uh, mu, lmbda, out=energy_field)
                elastic_energy = cutfemx.fem.assemble_scalar(
                    cutfemx.fem.form(compliance_density * dx_solid)
                )
                elastic_energy = float(comm.allreduce(elastic_energy, op=MPI.SUM))
                half_elastic_energy = 0.5 * elastic_energy
                row["elastic_half_energy"] = half_elastic_energy
                energy_scale = abs(elastic_energy / max(volume, 1.0e-14))
                if energy_scale < 1.0e-14:
                    energy_scale = 1.0
                row["energy_scale"] = energy_scale

                target_delta = 0.0
                target_rate = 0.0
                volume_multiplier: float | None = None
                if (
                    args.alm_auto_scale
                    and not alm_auto_scaled
                    and args.volume_control == "alm"
                ):
                    _initialise_augmented_lagrangian_scale(
                        alm,
                        compliance,
                        volume_constraint,
                    )
                    alm_auto_scaled = True
                if args.volume_control == "rate" and volume > target_volume:
                    target_delta = -min(
                        volume - target_volume,
                        args.max_volume_step * reference_volume,
                    )
                    target_rate = target_delta / dt if dt > 0.0 else 0.0
                elif args.volume_control == "alm":
                    _update_augmented_lagrangian(alm, volume_constraint)
                    volume_multiplier = _alm_velocity_multiplier(
                        alm, volume_constraint
                    )
                row["target_delta_volume"] = target_delta
                row["volume_rate_target"] = target_rate
                row["alm_lambda"] = alm.lagrange_multiplier
                row["alm_penalty"] = alm.penalty
                row["alm_multiplier"] = (
                    0.0 if volume_multiplier is None else volume_multiplier
                )
                row["lagrangian"] = _lagrangian_value(
                    compliance, volume_constraint, alm
                )

                row["lambda_volume"] = 0.0

            with _phase(row, "extension"):
                riesz_shape_rhs, riesz_volume_rhs = _build_interface_riesz_forms(
                    riesz_solver,
                    compliance_density,
                    dx_interface,
                )
                extension, lambda_volume, rates = _solve_riesz_velocity_volume(
                    riesz_solver,
                    phi,
                    riesz_shape_rhs,
                    riesz_volume_rhs,
                    dx_interface,
                    volume_multiplier=volume_multiplier,
                    target_rate=target_rate,
                    lambda_limit=args.volume_lambda_limit,
                )
                row["lambda_volume"] = lambda_volume
                row.update(rates)
                velocity_h1_scale = _extension_h1_scale(
                    extension.speed,
                    smoothing_length=smoothing_length,
                )
                if np.isfinite(velocity_h1_scale) and velocity_h1_scale > 0.0:
                    velocity_scale_reference = max(
                        velocity_scale_reference, velocity_h1_scale
                    )
                velocity_scale, velocity_scale_floor, velocity_scale_limited = (
                    _bounded_velocity_scale(
                        velocity_h1_scale,
                        velocity_scale_reference,
                    )
                )
                raw_velocity_max, velocity_max = _normalise_extension_velocity(
                    extension,
                    scale=velocity_scale,
                )
                row["raw_velocity_max"] = raw_velocity_max
                row["velocity_h1_scale"] = velocity_h1_scale
                row["velocity_scale_floor"] = velocity_scale_floor
                row["velocity_scale"] = velocity_scale
                row["velocity_scale_limited"] = velocity_scale_limited
                row["velocity_max"] = velocity_max
                row["speed_min"], row["speed_max"] = _function_min_max(
                    extension.speed
                )
                if velocity_scale != 0.0:
                    row["volume_rate_predicted"] = (
                        float(row["volume_rate_predicted"]) / velocity_scale
                    )
                predicted_rate = float(row["volume_rate_predicted"])
                if volume > target_volume and predicted_rate < 0.0:
                    max_delta = args.max_volume_step * reference_volume
                    predicted_delta = dt * predicted_rate
                    if -predicted_delta > max_delta:
                        row["dt"] = max_delta / abs(predicted_rate)
                    row["target_delta_volume"] = float(row["dt"]) * predicted_rate

            if args.shape_check_interval > 0 and iteration % args.shape_check_interval == 0:
                with _phase(row, "shape_check"):
                    row.update(
                        _shape_derivative_check(
                            phi,
                            args,
                            extension,
                            compliance_density,
                            dx_interface,
                            V,
                            u,
                            v,
                            elasticity_bcs,
                            ds_load,
                            traction,
                            mu,
                            lmbda,
                            float(row["elastic_half_energy"]),
                            float(row["compliance"]),
                            float(row["volume"]),
                        )
                    )

            step_iteration = IterationResult(
                row=row,
                displacement=uh,
                energy=energy,
                extension=extension,
                cut_data=cut_data,
            )
            nonfinite_step = (
                bool(row["solve_singular"])
                or bool(row["solve_nonfinite"])
                or not np.isfinite(float(row["compliance"]))
                or not np.isfinite(float(row["lambda_volume"]))
            )
            stop_after_row = nonfinite_step
            write_periodic_snapshot = (
                args.xdmf_interval > 0 and iteration % args.xdmf_interval == 0
            )
            write_failure_snapshot = bool(args.failure_snapshot and nonfinite_step)
            if write_periodic_snapshot or write_failure_snapshot:
                with _phase(row, "output"):
                    (
                        background_path,
                        solid_path,
                        background_vtk_path,
                        solid_vtk_path,
                    ) = _write_xdmf_snapshot(
                        msh,
                        phi,
                        step_iteration,
                        xdmf_dir,
                        iteration,
                        write_vtk=args.write_vtk,
                    )
                    if comm.rank == 0:
                        reason = "failure " if write_failure_snapshot else ""
                        print(f"Wrote {reason}XDMF snapshot to {background_path}")
                        if background_vtk_path is not None:
                            print(
                                f"Wrote {reason}VTK snapshot to {background_vtk_path}"
                            )
                        if solid_path is not None:
                            print(
                                f"Wrote {reason}cut-mesh XDMF snapshot to {solid_path}"
                            )
                        if solid_vtk_path is not None:
                            print(
                                f"Wrote {reason}cut-mesh VTK snapshot to {solid_vtk_path}"
                            )

            if nonfinite_step:
                row["volume_after_advection"] = row["volume"]
                row["volume_after_raw_reinitialization"] = row["volume"]
                row["volume_after_reinitialization"] = row["volume"]
                row["phi_min"], row["phi_max"] = _function_min_max(phi)
            else:
                phi_before_update = phi.x.array.copy()
                base_dt = float(row["dt"])
                accepted_update = False
                max_backtracks = 8
                for attempt in range(max_backtracks + 1):
                    if attempt > 0:
                        phi.x.array[:] = phi_before_update
                        phi.x.scatter_forward()
                    trial_dt = base_dt / (2**attempt)
                    row["advection_backtracks"] = attempt
                    row["dt"] = trial_dt

                    with _phase(row, "advection"):
                        _advect_level_set_supg(advection_solver, phi, extension, trial_dt)
                        row["volume_after_advection"] = _solid_volume(phi, args)

                    with _phase(row, "reinitialization"):
                        redistanced = False
                        if args.reinit_interval > 0 and (
                            (iteration + 1) % args.reinit_interval == 0
                        ):
                            reinitialize(
                                phi,
                                max_iter=args.reinit_max_iter,
                                tol=args.reinit_tol,
                            )
                            redistanced = True
                        raw_reinit_volume = (
                            _solid_volume(phi, args)
                            if redistanced
                            else row["volume_after_advection"]
                        )
                        row["volume_after_raw_reinitialization"] = raw_reinit_volume
                        if redistanced and args.reinit_volume_correction:
                            shift, corrected_volume = _apply_reinit_volume_correction(
                                phi,
                                args,
                                float(row["volume_after_advection"]),
                                float(raw_reinit_volume),
                            )
                            row["reinit_volume_shift"] = shift
                            row["volume_after_reinitialization"] = corrected_volume
                        else:
                            row["volume_after_reinitialization"] = raw_reinit_volume
                        row["phi_min"], row["phi_max"] = _function_min_max(phi)

                    topology_after = _level_set_topology_diagnostics(
                        phi,
                        args,
                        np.asarray(left_facets, dtype=np.int32),
                        np.asarray(right_tags.indices, dtype=np.int32),
                    )
                    if (
                        topology_after["floating_components"] > 0
                        and topology_after["floating_loaded_components"] == 0
                    ):
                        row["floating_island_components_removed"] += int(
                            topology_after["floating_components"]
                        )
                        row["floating_island_cells_removed"] += int(
                            topology_after["floating_cells"]
                        )
                        topology_after, removed_dofs = (
                            _cleanup_unloaded_floating_islands(
                                phi,
                                args,
                                np.asarray(left_facets, dtype=np.int32),
                                np.asarray(right_tags.indices, dtype=np.int32),
                                island_clear_value,
                            )
                        )
                        row["floating_island_dofs_removed"] += removed_dofs
                        if removed_dofs > 0:
                            filtered_volume = _solid_volume(phi, args)
                            row["volume_after_raw_reinitialization"] = filtered_volume
                            row["volume_after_reinitialization"] = filtered_volume
                            row["phi_min"], row["phi_max"] = _function_min_max(phi)

                    if topology_after["floating_components"] == 0:
                        accepted_update = True
                        dt = trial_dt
                        break

                if not accepted_update:
                    phi.x.array[:] = phi_before_update
                    phi.x.scatter_forward()
                    row["topology_invalid"] = 1
                    row["volume_after_advection"] = row["volume"]
                    row["volume_after_raw_reinitialization"] = row["volume"]
                    row["volume_after_reinitialization"] = row["volume"]
                    row["phi_min"], row["phi_max"] = _function_min_max(phi)
                    stop_after_row = True

            current, peak = tracemalloc.get_traced_memory()
            row["time_iteration"] = time.perf_counter() - iteration_start
            row["rss_bytes"] = _rss_bytes()
            row["tracemalloc_current_bytes"] = int(current)
            row["tracemalloc_peak_bytes"] = int(peak)

            history.append(dict(row))
            profile.write(row)
            if args.tracemalloc_interval > 0 and (
                iteration % args.tracemalloc_interval == 0
                or iteration == args.iterations - 1
                or stop_after_row
            ):
                _write_tracemalloc_report(
                    tracemalloc_dir,
                    iteration,
                    tracemalloc_baseline,
                )
            last_iteration = IterationResult(
                row=dict(row),
                displacement=uh,
                energy=energy,
                extension=extension,
                cut_data=cut_data,
            )

            if comm.rank == 0:
                print(
                    f"iter {iteration:03d}: "
                    f"J={float(row['compliance']):.6e}, "
                    f"V/V0={float(row['volume_fraction']):.4f}"
                    f"->{float(row['volume_after_reinitialization']) / reference_volume:.4f}, "
                    f"lambda={float(row['lambda_volume']):.4e}, "
                    f"max|v|={float(row['velocity_max']):.3e}"
                )

            if args.gc_each_step:
                gc.collect()

            if stop_after_row:
                if comm.rank == 0:
                    if row["topology_invalid"]:
                        print(
                            "Stopping optimization after a floating active solid "
                            "component could not be removed by backtracking."
                        )
                    else:
                        print(
                            "Stopping optimization after nonfinite/singular step; "
                            "failure snapshot and profile row were written."
                        )
                break

    if last_iteration is None:
        raise RuntimeError("At least one optimization iteration is required.")

    final_cut_data = _cut_level_set(phi, args)
    return OptimizationResult(
        mesh=msh,
        phi=phi,
        cut_data=final_cut_data,
        last_iteration=last_iteration,
        history=history,
        profile_csv=profile_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32, help="Cells in the vertical direction.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--domain-length", type=float, default=2.0)
    parser.add_argument("--domain-height", type=float, default=1.0)
    parser.add_argument(
        "--cell-type",
        choices=("triangle", "quadrilateral"),
        default="quadrilateral",
        help=(
            "Background cell type. The demo default is quadrilateral for "
            "stable cut-mesh visualization; use triangle for the literal "
            "OptiCut background mesh."
        ),
    )
    parser.add_argument(
        "--initial-geometry",
        choices=("holes", "opticut"),
        default="opticut",
        help="Initial level set. 'opticut' is the striped OptiCut cosine initializer; 'holes' is a smaller circular-hole debugging case.",
    )
    parser.add_argument(
        "--hole-rows",
        type=int,
        default=2,
        help="Rows of initial circular holes for --initial-geometry holes.",
    )
    parser.add_argument(
        "--hole-cols",
        type=int,
        default=4,
        help="Columns of initial circular holes for --initial-geometry holes.",
    )
    parser.add_argument(
        "--hole-radius",
        type=float,
        default=0.107,
        help=(
            "Initial circular-hole radius for --initial-geometry holes. The "
            "default deliberately avoids placing the circular interface exactly "
            "on common background-mesh vertices."
        ),
    )
    parser.add_argument("--volume-fraction", type=float, default=0.70)
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-3,
        help="Fixed level-set advection pseudo-time step. Use --dt 0 to use --cfl*h.",
    )
    parser.add_argument("--cfl", type=float, default=0.1)
    parser.add_argument(
        "--volume-control",
        choices=("alm", "rate"),
        default="rate",
        help=(
            "Volume constraint handling. The demo default 'rate' enforces a "
            "bounded per-step volume decrease; 'alm' exposes the OptiCut-style "
            "augmented Lagrangian sequence."
        ),
    )
    parser.add_argument(
        "--max-volume-step",
        type=float,
        default=5.0e-4,
        help=(
            "Maximum reference-domain volume fraction changed per iteration. "
            "For ALM this caps the advection step size; for rate control it "
            "sets the target removal rate."
        ),
    )
    parser.add_argument(
        "--volume-lambda-limit",
        type=float,
        default=0.0,
        help=(
            "Absolute cap on the volume multiplier. Use 0 for no cap. This "
            "is mainly a diagnostic safeguard for overshooting volume terms."
        ),
    )
    parser.add_argument(
        "--alm-lambda",
        type=float,
        default=0.1,
        help=(
            "Initial augmented-Lagrangian multiplier for the volume constraint. "
            "Ignored on the first iteration when --alm-auto-scale is enabled."
        ),
    )
    parser.add_argument(
        "--alm-penalty",
        type=float,
        default=1.0,
        help=(
            "Initial augmented-Lagrangian penalty for the volume constraint. "
            "Ignored on the first iteration when --alm-auto-scale is enabled."
        ),
    )
    parser.add_argument(
        "--alm-penalty-limit",
        type=float,
        default=100.0,
        help="Upper bound for the augmented-Lagrangian penalty.",
    )
    parser.add_argument(
        "--alm-penalty-multiplier",
        type=float,
        default=1.1,
        help="Penalty growth factor used after each ALM update.",
    )
    parser.add_argument(
        "--alm-auto-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize ALM multiplier and penalty from the first objective and "
            "volume-constraint scales, following OptiCut's default posture."
        ),
    )
    parser.add_argument(
        "--velocity-smoothing",
        type=float,
        default=None,
        help=(
            "H1 smoothing length for the Riesz velocity solve. Defaults to "
            "sqrt(--velocity-alpha), matching OptiCut's alpha_reg_velocity."
        ),
    )
    parser.add_argument(
        "--velocity-alpha",
        type=float,
        default=1.0e-2,
        help="OptiCut-style H1 velocity regularization coefficient.",
    )
    parser.add_argument(
        "--supg-tau-scale",
        type=float,
        default=1.0,
        help="Multiplier for the residual-based SUPG level-set advection tau.",
    )
    parser.add_argument("--order", type=int, default=3, help="Cut quadrature order.")
    parser.add_argument("--young-modulus", type=float, default=210000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.33)
    parser.add_argument("--traction", type=float, default=10.0)
    parser.add_argument(
        "--load-center",
        type=float,
        default=0.5,
        help="Vertical center of the loaded right-boundary patch.",
    )
    parser.add_argument(
        "--load-half-height",
        type=float,
        default=0.05,
        help="Half-height of the loaded patch on the right boundary.",
    )
    parser.add_argument(
        "--ghost-gamma",
        type=float,
        default=1.0e-5,
        help="Elasticity ghost penalty factor multiplying (2*mu+lambda)*h^3.",
    )
    parser.add_argument(
        "--cut-approximation",
        choices=("auto", "linear", "iso_p1"),
        default="linear",
        help=(
            "Level-set approximation used by CutCells. The benchmark default "
            "forces the P1/linear lookup-table path."
        ),
    )
    parser.add_argument(
        "--cut-approximation-order",
        type=int,
        default=1,
        help="Order for iso_p1 cut approximation. Must be 1 with linear cuts.",
    )
    parser.add_argument(
        "--max-refinement-iterations",
        type=int,
        default=0,
        help=(
            "Maximum AdaptCell certification refinements. The benchmark default "
            "0 classifies the root cell but does not subdivide."
        ),
    )
    parser.add_argument(
        "--edge-max-depth",
        type=int,
        default=20,
        help="Maximum edge root subdivision depth inside CutCells.",
    )
    parser.add_argument("--reinit-max-iter", type=int, default=600)
    parser.add_argument("--reinit-tol", type=float, default=1.0e-10)
    parser.add_argument(
        "--initial-reinit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one FIM redistancing pass immediately after initial level-set interpolation.",
    )
    parser.add_argument(
        "--reinit-interval",
        type=int,
        default=1,
        help=(
            "Run FIM redistancing every N iterations. Use 0 to disable "
            "redistancing for diagnostics."
        ),
    )
    parser.add_argument(
        "--reinit-volume-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After redistancing, apply a constant signed-distance shift so the "
            "solid volume matches the pre-redistance projected volume."
        ),
    )
    parser.add_argument(
        "--reinit-volume-correction-limit",
        type=float,
        default=0.05,
        help=(
            "Absolute cap on the post-redistance constant level-set shift. "
            "Use 0 for no cap."
        ),
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show PyVista.")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("compliance_optimization_output"))
    parser.add_argument(
        "--xdmf-interval",
        type=int,
        default=10,
        help=(
            "Write background and cut-mesh XDMF snapshots every N iterations. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--xdmf-dir",
        type=Path,
        default=None,
        help="Directory for periodic XDMF snapshots. Defaults to output-dir/xdmf.",
    )
    parser.add_argument(
        "--write-vtk",
        action="store_true",
        help=(
            "Also write PVD/VTU snapshots next to XDMF. Useful for ParaView "
            "visualization checks, but disabled by default to reduce I/O."
        ),
    )
    parser.add_argument(
        "--failure-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an extra XDMF snapshot when a solve or sensitivity is nonfinite.",
    )
    parser.add_argument(
        "--topology-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record active-solid connected-component diagnostics in the profile.",
    )
    parser.add_argument(
        "--tracemalloc-interval",
        type=int,
        default=0,
        help="Write line-level Python allocation reports every N iterations. Use 0 to disable.",
    )
    parser.add_argument(
        "--tracemalloc-dir",
        type=Path,
        default=None,
        help="Directory for tracemalloc reports. Defaults to output-dir/tracemalloc.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Use benchmark posture: always profile and skip interactive plotting by default.",
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=None,
        help="Override the default profile CSV path in the output directory.",
    )
    parser.add_argument(
        "--gc-each-step",
        action="store_true",
        help="Force Python garbage collection after every iteration.",
    )
    parser.add_argument(
        "--shape-check-interval",
        type=int,
        default=0,
        help=(
            "Run a finite-difference shape derivative check every N iterations. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--shape-check-step",
        type=float,
        default=1.0e-4,
        help="Pseudo-time step used by the finite-difference shape check.",
    )
    args = parser.parse_args()
    if args.n < 4:
        parser.error("--n must be at least 4.")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1.")
    if args.domain_length <= 0.0:
        parser.error("--domain-length must be positive.")
    if args.domain_height <= 0.0:
        parser.error("--domain-height must be positive.")
    if not (0.0 < args.volume_fraction < 1.0):
        parser.error("--volume-fraction must lie in (0, 1).")
    if args.dt < 0.0:
        parser.error("--dt must be nonnegative.")
    if args.hole_rows < 1:
        parser.error("--hole-rows must be at least 1.")
    if args.hole_cols < 1:
        parser.error("--hole-cols must be at least 1.")
    if args.hole_radius <= 0.0:
        parser.error("--hole-radius must be positive.")
    if args.hole_radius >= 0.5 * min(args.domain_length, args.domain_height):
        parser.error("--hole-radius must be smaller than half the short domain side.")
    if args.cfl <= 0.0:
        parser.error("--cfl must be positive.")
    if args.max_volume_step <= 0.0:
        parser.error("--max-volume-step must be positive.")
    if args.volume_lambda_limit < 0.0:
        parser.error("--volume-lambda-limit must be nonnegative.")
    if args.alm_penalty <= 0.0:
        parser.error("--alm-penalty must be positive.")
    if args.alm_penalty_limit <= 0.0:
        parser.error("--alm-penalty-limit must be positive.")
    if args.alm_penalty_multiplier < 1.0:
        parser.error("--alm-penalty-multiplier must be at least 1.")
    if args.xdmf_interval < 0:
        parser.error("--xdmf-interval must be nonnegative.")
    if args.tracemalloc_interval < 0:
        parser.error("--tracemalloc-interval must be nonnegative.")
    if not (0.0 <= args.load_center <= args.domain_height):
        parser.error("--load-center must lie inside the vertical domain.")
    if args.load_half_height <= 0.0:
        parser.error("--load-half-height must be positive.")
    if (
        args.load_center - args.load_half_height < 0.0
        or args.load_center + args.load_half_height > args.domain_height
    ):
        parser.error("--load-center +/- --load-half-height must stay inside the domain.")
    if args.velocity_smoothing is not None and args.velocity_smoothing <= 0.0:
        parser.error("--velocity-smoothing must be positive.")
    if args.velocity_alpha <= 0.0:
        parser.error("--velocity-alpha must be positive.")
    if args.supg_tau_scale < 0.0:
        parser.error("--supg-tau-scale must be nonnegative.")
    if args.cut_approximation == "linear" and args.cut_approximation_order != 1:
        parser.error("--cut-approximation-order must be 1 for linear cuts.")
    if args.cut_approximation_order < 1:
        parser.error("--cut-approximation-order must be positive.")
    if args.max_refinement_iterations < 0:
        parser.error("--max-refinement-iterations must be nonnegative.")
    if args.edge_max_depth < 0:
        parser.error("--edge-max-depth must be nonnegative.")
    if args.reinit_interval < 0:
        parser.error("--reinit-interval must be nonnegative.")
    if args.reinit_volume_correction_limit < 0.0:
        parser.error("--reinit-volume-correction-limit must be nonnegative.")
    if args.shape_check_interval < 0:
        parser.error("--shape-check-interval must be nonnegative.")
    if args.shape_check_step <= 0.0:
        parser.error("--shape-check-step must be positive.")
    return args


def main() -> int:
    args = parse_args()
    result = run_optimization(args)

    show = not args.no_show and not args.benchmark
    has_output = args.history is not None or show or args.screenshot is not None
    if has_output:
        with _phase(result.last_iteration.row, "output"):
            if args.history is not None and MPI.COMM_WORLD.rank == 0:
                _write_history_plot(args.history, result.history)
                print(f"Wrote history plot to {args.history}")

            if show or args.screenshot is not None:
                _plot_result(result, screenshot=args.screenshot, show=show)

        if result.history:
            result.history[-1]["time_output"] = result.last_iteration.row["time_output"]
            if MPI.COMM_WORLD.rank == 0:
                _write_profile_rows(result.profile_csv, result.history)

    if MPI.COMM_WORLD.rank == 0:
        print(f"Wrote profile CSV to {result.profile_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
