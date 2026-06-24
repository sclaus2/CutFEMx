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
The demo also retains an adaptive Sobolev-gradient flow: a Barzilai-Borwein
step proposal can be clipped by a geometric interface-motion cap and checked by
Armijo backtracking after the complete CutFEM update.
The default benchmark uses the ``sotuto/base`` cantilever data: a
``[0, 2] x [0, 1]`` mesh, clamped left edge, a centered partial traction on the
right edge, the perforated circular-hole initial level set, and the original
null-space volume-constrained descent scaling.  Pass
``--initial-geometry opticut`` for the striped OptiCut cosine initializer or
``--initial-geometry holes`` for the smaller circular-hole debugging
initializer.

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
from typing import Callable, Iterator, Sequence
import warnings

import basix.ufl
from mpi4py import MPI
import numpy as np
import ufl

from cutfemx.distance import NormalExtensionResult, reinitialize
import cutfemx
from dolfinx import default_scalar_type, fem, geometry, io, la, mesh


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
    "line_search",
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
    "sotuto_penalty",
    "sotuto_alpha_objective",
    "sotuto_alpha_constraint",
    "sotuto_norm_xi_objective",
    "sotuto_norm_xi_constraint",
    "volume_after_advection",
    "volume_after_raw_reinitialization",
    "volume_after_reinitialization",
    "reinit_volume_shift",
    "dt",
    "step_dt_previous",
    "step_dt_bb",
    "step_dt_motion_cap",
    "step_dt_proposed",
    "step_dt_accepted",
    "step_bb_accepted",
    "lambda_volume",
    "energy_scale",
    "elastic_half_energy",
    "raw_velocity_max",
    "velocity_h1_scale",
    "velocity_scale_floor",
    "velocity_scale",
    "velocity_scale_limited",
    "velocity_max",
    "lbfgs_pairs",
    "lbfgs_curvature",
    "lbfgs_descent_dot",
    "lbfgs_reset",
    "lbfgs_update_accepted",
    "lbfgs_volume_correction",
    "lbfgs_fallback_to_gradient",
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
    "objective_rate_predicted",
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
    "line_search_failed",
    "line_search_trial_compliance",
    "line_search_trial_volume",
    "line_search_trial_merit",
    "line_search_armijo_rhs",
    "line_search_actual_delta",
    "min_component_cells",
    "max_component_cells",
    "time_iteration",
    "rss_bytes",
    "tracemalloc_current_bytes",
    "tracemalloc_peak_bytes",
    *(f"time_{phase}" for phase in PHASES),
)

CONVERGENCE_FIELDS = (
    "iteration",
    "compliance",
    "objective",
    "volume",
    "volume_fraction",
    "target_volume",
    "volume_constraint",
    "grad_J_h1",
    "grad_J_max",
    "grad_L_h1",
    "grad_L_max",
    "objective_rate_predicted",
    "volume_rate_predicted",
    "dt",
    "line_search_failed",
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
    convergence_csv: Path


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
class NodalGradientStencil:
    """Fixed-mesh least-squares stencil for scalar P1 nodal gradients."""

    neighbors: list[np.ndarray]
    weights: list[np.ndarray]


@dataclass
class RieszVelocityUpdate:
    """Projected normal-speed data before advection."""

    extension: NormalExtensionResult
    lambda_volume: float
    rates: dict[str, float]
    shape_speed_values: np.ndarray
    shape_rhs_values: np.ndarray
    volume_rhs_values: np.ndarray
    volume_speed_values: np.ndarray


@dataclass
class AdaptiveGradientStepState:
    """State for the Barzilai-Borwein Sobolev-gradient step length."""

    previous_phi: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    accepted_dt: float = 0.0


@dataclass
class LevelSetAdvectionSolver:
    """SUPG level-set advection solver on the fixed background mesh."""

    space: fem.FunctionSpace
    speed: fem.Function
    dt: fem.Constant
    fixed_dofs: np.ndarray
    nodal_gradient: NodalGradientStencil
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


@dataclass
class LBFGSState:
    """Limited-memory inverse-Hessian state in the scalar speed space."""

    s_vectors: list[np.ndarray]
    y_vectors: list[np.ndarray]
    rho_values: list[float]
    previous_x: np.ndarray | None = None
    previous_gradient: np.ndarray | None = None
    last_curvature: float = 0.0
    last_update_accepted: bool = False


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


class ConvergenceWriter:
    """Streaming CSV writer for scalar convergence monitoring."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CONVERGENCE_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        return self

    def write(self, row: dict[str, float | int]) -> None:
        if self._writer is None or self._file is None:
            raise RuntimeError("ConvergenceWriter is not open.")
        self._writer.writerow(
            {field: row.get(field, "") for field in CONVERGENCE_FIELDS}
        )
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


def _make_sotuto_level_set(args: argparse.Namespace):
    """Initial perforated cantilever level set from ``sotuto/base``.

    The sign convention is negative in solid and positive inside the circular
    voids, matching the negated FreeFem++ ``iniLS`` field.
    """
    rows = (
        (6, 0.285, 0.285, 1.00),
        (5, 0.3333, 0.3333, 0.75),
        (6, 0.285, 0.285, 0.50),
        (5, 0.3333, 0.3333, 0.25),
        (6, 0.285, 0.285, 0.00),
    )
    x_scale = args.domain_length / 2.0
    y_scale = args.domain_height
    radius = 0.05 * min(x_scale, y_scale)

    def level_set(x: np.ndarray) -> np.ndarray:
        distance = np.full_like(x[0], 10.0, dtype=np.float64)
        for count, start, spacing, cy_unit in rows:
            cy = cy_unit * y_scale
            for i in range(count):
                cx = (start + i * spacing) * x_scale
                distance = np.minimum(
                    distance,
                    np.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2) - radius,
                )
        return -distance

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
    if args.initial_geometry == "sotuto":
        return _make_sotuto_level_set(args)
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
    except ImportError as exc:
        if msh.comm.rank == 0:
            print(
                "Plotting skipped because pyvista/dolfinx.plot is unavailable: "
                f"{exc}"
            )
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


def _h1_scale_from_values(
    solver: RieszVelocitySolver,
    values: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    value = _h1_inner_from_values(solver, values, values)
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(np.sqrt(value))


def _h1_inner_from_values(
    solver: RieszVelocitySolver,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    value = float(left @ (solver.matrix @ right))
    return float(solver.space.mesh.comm.allreduce(value, op=MPI.SUM))


def _extension_h1_scale(
    solver: RieszVelocitySolver,
    speed: fem.Function,
) -> float:
    return _h1_scale_from_values(solver, np.asarray(speed.x.array, dtype=float))


def _bounded_velocity_scale(
    h1_scale: float,
    reference_scale: float,
    *,
    floor_fraction: float = VELOCITY_NORMALIZATION_SCALE_FLOOR_FRACTION,
    absolute_floor: float = VELOCITY_NORMALIZATION_ABS_FLOOR,
) -> tuple[float, float, int]:
    """Return the denominator used for velocity normalization.

    Normalize with the largest reliable H1 velocity scale seen so far instead
    of the current H1 scale. This keeps early updates O(1), but lets late
    weak-gradient velocities decay instead of amplifying residual noise back to
    unit size.
    """
    if not np.isfinite(h1_scale) or h1_scale <= 0.0:
        if np.isfinite(reference_scale) and reference_scale > 0.0:
            floor = max(float(absolute_floor), float(floor_fraction) * reference_scale)
            return float(reference_scale), floor, 1
        return 1.0, 0.0, 0

    if not np.isfinite(reference_scale) or reference_scale <= 0.0:
        reference_scale = h1_scale
    reference_scale = max(float(reference_scale), float(h1_scale))
    floor = max(float(absolute_floor), float(floor_fraction) * reference_scale)
    scale = max(reference_scale, floor)
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


def _normalise_extension_and_update_row(
    result: NormalExtensionResult,
    solver: RieszVelocitySolver,
    row: dict[str, float | int],
    velocity_scale_reference: float,
) -> tuple[float, float]:
    velocity_h1_scale = _extension_h1_scale(solver, result.speed)
    if np.isfinite(velocity_h1_scale) and velocity_h1_scale > 0.0:
        velocity_scale_reference = max(velocity_scale_reference, velocity_h1_scale)
    velocity_scale, velocity_scale_floor, velocity_scale_limited = (
        _bounded_velocity_scale(
            velocity_h1_scale,
            velocity_scale_reference,
        )
    )
    raw_velocity_max, velocity_max = _normalise_extension_velocity(
        result,
        scale=velocity_scale,
    )
    row["raw_velocity_max"] = raw_velocity_max
    row["velocity_h1_scale"] = velocity_h1_scale
    row["velocity_scale_floor"] = velocity_scale_floor
    row["velocity_scale"] = velocity_scale
    row["velocity_scale_limited"] = velocity_scale_limited
    row["velocity_max"] = velocity_max
    row["speed_min"], row["speed_max"] = _function_min_max(result.speed)
    if velocity_scale != 0.0:
        row["volume_rate_predicted"] = (
            float(row["volume_rate_predicted"]) / velocity_scale
        )
    return velocity_scale_reference, velocity_scale


def _global_dot(msh: mesh.Mesh, left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.dot(np.asarray(left, dtype=float), np.asarray(right, dtype=float)))
    return float(msh.comm.allreduce(value, op=MPI.SUM))


def _global_max_abs(msh: mesh.Mesh, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    local = float(np.max(np.abs(values))) if values.size else 0.0
    return float(msh.comm.allreduce(local, op=MPI.MAX))


def _predicted_compliance_rate(
    solver: RieszVelocitySolver,
    speed_values: np.ndarray,
    shape_rhs_values: np.ndarray,
) -> float:
    return -_global_dot(solver.space.mesh, speed_values, shape_rhs_values)


def _motion_dt_cap(hmin: float, velocity_max: float, motion_cfl: float) -> float:
    if motion_cfl <= 0.0:
        return np.inf
    if not np.isfinite(velocity_max) or velocity_max <= 0.0:
        return np.inf
    return float(motion_cfl * hmin / max(velocity_max, 1.0e-14))


def _adaptive_gradient_dt(
    state: AdaptiveGradientStepState,
    msh: mesh.Mesh,
    phi_values: np.ndarray,
    gradient_values: np.ndarray,
    previous_dt: float,
    hmin: float,
    velocity_max: float,
    motion_cfl: float,
    *,
    enabled: bool,
) -> dict[str, float | int]:
    """Return a BB step proposal clipped by growth and interface-motion caps."""
    previous_dt = float(previous_dt)
    bb_dt = previous_dt
    bb_accepted = 0
    if (
        enabled
        and state.previous_phi is not None
        and state.previous_gradient is not None
    ):
        s = np.asarray(phi_values, dtype=float) - state.previous_phi
        y = np.asarray(gradient_values, dtype=float) - state.previous_gradient
        sy = _global_dot(msh, s, y)
        ss = _global_dot(msh, s, s)
        if np.isfinite(sy) and sy > 1.0e-30 and np.isfinite(ss) and ss > 0.0:
            bb_dt = ss / sy
            bb_accepted = int(np.isfinite(bb_dt) and bb_dt > 0.0)
        if not bb_accepted:
            bb_dt = previous_dt

    if not np.isfinite(bb_dt) or bb_dt <= 0.0:
        bb_dt = previous_dt

    # Keep the BB estimate useful but conservative for a nonsmooth cut update.
    growth_limited = float(
        np.clip(
            bb_dt,
            0.25 * previous_dt,
            2.0 * previous_dt,
        )
    )
    motion_cap = _motion_dt_cap(hmin, velocity_max, motion_cfl)
    proposed_dt = min(growth_limited, motion_cap)
    if not np.isfinite(proposed_dt) or proposed_dt <= 0.0:
        proposed_dt = previous_dt
    return {
        "step_dt_previous": previous_dt,
        "step_dt_bb": float(bb_dt),
        "step_dt_motion_cap": float(motion_cap),
        "step_dt_proposed": float(proposed_dt),
        "step_bb_accepted": bb_accepted,
    }


def _accept_adaptive_gradient_step(
    state: AdaptiveGradientStepState,
    phi_values: np.ndarray,
    gradient_values: np.ndarray,
    accepted_dt: float,
) -> None:
    state.previous_phi = np.asarray(phi_values, dtype=float).copy()
    state.previous_gradient = np.asarray(gradient_values, dtype=float).copy()
    state.accepted_dt = float(accepted_dt)


def _armijo_rhs(
    current_objective: float,
    predicted_rate: float,
    dt: float,
    sufficient_decrease: float,
) -> float:
    if np.isfinite(predicted_rate) and predicted_rate < 0.0:
        return float(current_objective + sufficient_decrease * dt * predicted_rate)
    return float(current_objective * (1.0 + 1.0e-10))


def _target_volume_delta(
    volume: float,
    target_volume: float,
    reference_volume: float,
    max_volume_step: float,
) -> float:
    volume_error = target_volume - volume
    if abs(volume_error) <= 1.0e-12 * reference_volume:
        return 0.0
    max_delta = max_volume_step * reference_volume
    return float(np.clip(volume_error, -max_delta, max_delta))


def _clear_lbfgs_state(state: LBFGSState) -> None:
    state.s_vectors.clear()
    state.y_vectors.clear()
    state.rho_values.clear()
    state.previous_x = None
    state.previous_gradient = None
    state.last_curvature = 0.0
    state.last_update_accepted = False


def _set_extension_speed(
    result: NormalExtensionResult,
    phi: fem.Function,
    speed_values: np.ndarray,
) -> None:
    result.speed.x.array[:] = np.asarray(speed_values, dtype=float)
    result.speed.x.scatter_forward()
    grad_phi = ufl.grad(phi)
    normal = grad_phi / ufl.sqrt(ufl.inner(grad_phi, grad_phi) + 1.0e-14)
    result.velocity.interpolate(
        fem.Expression(
            result.speed * normal,
            _interpolation_points(result.velocity.function_space),
        )
    )
    result.velocity.x.scatter_forward()


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


def _lbfgs_update(
    state: LBFGSState,
    x: np.ndarray,
    gradient: np.ndarray,
    *,
    memory: int,
    curvature_tol: float,
) -> None:
    """Update L-BFGS history from the previous accepted design state."""
    state.last_curvature = 0.0
    state.last_update_accepted = False
    if state.previous_x is None or state.previous_gradient is None:
        state.previous_x = np.asarray(x, dtype=float).copy()
        state.previous_gradient = np.asarray(gradient, dtype=float).copy()
        return

    s = np.asarray(x, dtype=float) - state.previous_x
    y = np.asarray(gradient, dtype=float) - state.previous_gradient
    curvature = float(np.dot(s, y))
    state.last_curvature = curvature
    scale = max(float(np.linalg.norm(s) * np.linalg.norm(y)), 1.0e-30)
    if memory > 0 and np.isfinite(curvature) and curvature > curvature_tol * scale:
        state.s_vectors.append(s.copy())
        state.y_vectors.append(y.copy())
        state.rho_values.append(1.0 / curvature)
        if len(state.s_vectors) > memory:
            state.s_vectors.pop(0)
            state.y_vectors.pop(0)
            state.rho_values.pop(0)
        state.last_update_accepted = True

    state.previous_x = np.asarray(x, dtype=float).copy()
    state.previous_gradient = np.asarray(gradient, dtype=float).copy()


def _lbfgs_inverse_hessian_product(
    state: LBFGSState,
    gradient: np.ndarray,
) -> np.ndarray:
    """Apply the L-BFGS inverse Hessian approximation to a gradient vector."""
    q = np.asarray(gradient, dtype=float).copy()
    if len(state.s_vectors) == 0:
        return q

    alphas: list[float] = []
    for s, y, rho in zip(
        reversed(state.s_vectors),
        reversed(state.y_vectors),
        reversed(state.rho_values),
    ):
        alpha = rho * float(np.dot(s, q))
        alphas.append(alpha)
        q -= alpha * y

    s_last = state.s_vectors[-1]
    y_last = state.y_vectors[-1]
    yy = float(np.dot(y_last, y_last))
    sy = float(np.dot(s_last, y_last))
    gamma = sy / yy if yy > 1.0e-30 and sy > 0.0 else 1.0
    r = gamma * q

    for s, y, rho, alpha in zip(
        state.s_vectors,
        state.y_vectors,
        state.rho_values,
        reversed(alphas),
    ):
        beta = rho * float(np.dot(y, r))
        r += s * (alpha - beta)
    return r


def _lbfgs_descent_direction(
    state: LBFGSState,
    gradient: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Return a descent direction and whether history had to be discarded."""
    inverse_gradient = _lbfgs_inverse_hessian_product(state, gradient)
    direction = -inverse_gradient
    descent_dot = float(np.dot(gradient, direction))
    if np.isfinite(descent_dot) and descent_dot < 0.0:
        return direction, descent_dot, 0

    state.s_vectors.clear()
    state.y_vectors.clear()
    state.rho_values.clear()
    direction = -np.asarray(gradient, dtype=float)
    return direction, float(np.dot(gradient, direction)), 1


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


def _cut_level_set_and_solid_volume(
    phi: fem.Function,
    args: argparse.Namespace,
) -> tuple[cutfemx.CutData, float]:
    cut_data = _cut_level_set(phi, args)
    volume = _solid_volume_from_cut_data(
        phi.function_space.mesh,
        cut_data,
        args.order,
    )
    return cut_data, volume


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
    cut_data: cutfemx.CutData | None = None,
) -> tuple[float, float, cutfemx.CutData]:
    """Shift signed distance by a constant so redistancing preserves volume."""
    if cut_data is None:
        cut_data = _cut_level_set(phi, args)
    interface_measure = _interface_measure_from_cut_data(
        phi.function_space.mesh, cut_data, args.order
    )
    if interface_measure <= 1.0e-14:
        return 0.0, current_volume, cut_data

    # For phi < 0 in the solid and signed-distance phi, V(phi + c) changes as
    # dV/dc ~= -|Gamma|. A positive shift therefore removes solid volume.
    shift = (current_volume - target_volume) / interface_measure
    if args.reinit_volume_correction_limit > 0.0:
        limit = args.reinit_volume_correction_limit
        shift = float(np.clip(shift, -limit, limit))

    if shift == 0.0:
        return 0.0, current_volume, cut_data

    phi.x.array[:] += shift
    phi.x.scatter_forward()
    corrected_cut_data, corrected_volume = _cut_level_set_and_solid_volume(phi, args)
    return float(shift), corrected_volume, corrected_cut_data


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
) -> tuple[fem.Function, np.ndarray]:
    b = cutfemx.fem.assemble_vector(L)
    fem.apply_lifting(b.array, [solver.bilinear_form], [solver.bcs])
    b.scatter_reverse(la.InsertMode.add)
    fem.set_bc(b.array, solver.bcs)

    velocity = fem.Function(solver.space, name=name)
    velocity.x.array[:] = solver.linear_solve(b.array)
    velocity.x.scatter_forward()
    return velocity, np.asarray(b.array, dtype=float).copy()


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
    fixed_dofs = np.empty(0, dtype=np.int32)
    if fixed_facets.size > 0:
        fdim = msh.topology.dim - 1
        fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
        bcs = [fem.dirichletbc(phi, fixed_dofs)]

    a_form = fem.form(a)
    return LevelSetAdvectionSolver(
        space=V,
        speed=speed,
        dt=dt_constant,
        fixed_dofs=np.asarray(fixed_dofs, dtype=np.int32),
        nodal_gradient=_build_nodal_gradient_stencil(V),
        bcs=bcs,
        bilinear_form=a_form,
        rhs_form=fem.form(rhs),
    )


def _build_nodal_gradient_stencil(V: fem.FunctionSpace) -> NodalGradientStencil:
    """Precompute least-squares nodal-gradient weights for a scalar P1 space."""
    msh = V.mesh
    tdim = msh.topology.dim
    gdim = msh.geometry.dim
    num_cells = msh.topology.index_map(tdim).size_local
    dof_coords = np.asarray(V.tabulate_dof_coordinates(), dtype=float)[:, :gdim]
    neighbors: list[set[int]] = [set() for _ in range(dof_coords.shape[0])]

    for cell in range(num_cells):
        cell_dofs = np.asarray(V.dofmap.cell_dofs(cell), dtype=np.int32)
        for dof in cell_dofs:
            local_neighbors = neighbors[int(dof)]
            for neighbor in cell_dofs:
                if int(neighbor) != int(dof):
                    local_neighbors.add(int(neighbor))

    neighbor_arrays: list[np.ndarray] = []
    weight_arrays: list[np.ndarray] = []
    for dof, dof_neighbors in enumerate(neighbors):
        nbrs = np.asarray(sorted(dof_neighbors), dtype=np.int32)
        if nbrs.size < gdim:
            neighbor_arrays.append(nbrs)
            weight_arrays.append(np.zeros((gdim, nbrs.size), dtype=float))
            continue

        offsets = dof_coords[nbrs] - dof_coords[dof]
        weights = np.linalg.pinv(offsets)  # shape (gdim, n_neighbors)
        neighbor_arrays.append(nbrs)
        weight_arrays.append(np.asarray(weights, dtype=float))

    return NodalGradientStencil(neighbor_arrays, weight_arrays)


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


def _sotuto_merit(
    compliance: float,
    volume: float,
    target_volume: float,
    lagrange: float,
    penalty: float,
    alpha_objective: float,
    alpha_constraint: float,
) -> float:
    """Merit function used by ``sotuto/base`` for the null-space descent."""
    volume_error = volume - target_volume
    if not np.isfinite(penalty) or abs(penalty) <= 1.0e-30:
        return np.inf
    return float(
        alpha_objective * (compliance - lagrange * volume_error)
        + 0.5 * alpha_constraint / penalty * volume_error**2
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


def _sotuto_null_space_descent_values(
    solver: RieszVelocitySolver,
    shape_values: np.ndarray,
    volume_values: np.ndarray,
    *,
    volume: float,
    target_volume: float,
    mesh_size: float,
    max_norm_xi_objective: float,
    objective_weight: float,
    constraint_weight: float,
    eps: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return scalar speed values from ``sotuto/base`` null-space descent."""
    shape_values = np.asarray(shape_values, dtype=float)
    volume_values = np.asarray(volume_values, dtype=float)
    penalty = _h1_inner_from_values(solver, volume_values, volume_values)
    if not np.isfinite(penalty) or abs(penalty) <= eps:
        raise RuntimeError("Volume gradient has near-zero regularized norm.")

    lagrange = _h1_inner_from_values(solver, shape_values, volume_values) / penalty
    # shape_values = -theta_J and volume_values = -theta_G in the notation of
    # sotuto/base/sources/descent.edp, so these are the same xi fields up to sign.
    xi_objective = -shape_values + lagrange * volume_values
    xi_constraint = -((volume - target_volume) / penalty) * volume_values
    norm_xi_objective = _global_max_abs(solver.space.mesh, xi_objective)
    norm_xi_constraint = _global_max_abs(solver.space.mesh, xi_constraint)
    normalized_objective = max(norm_xi_objective, max_norm_xi_objective)
    alpha_objective = (
        objective_weight * mesh_size / (eps**2 + normalized_objective)
    )
    alpha_constraint = (
        constraint_weight * mesh_size / (eps**2 + norm_xi_constraint)
    )

    speed_values = -alpha_objective * xi_objective - alpha_constraint * xi_constraint
    return speed_values, {
        "lambda_volume": float(lagrange),
        "sotuto_penalty": float(penalty),
        "sotuto_alpha_objective": float(alpha_objective),
        "sotuto_alpha_constraint": float(alpha_constraint),
        "sotuto_norm_xi_objective": float(norm_xi_objective),
        "sotuto_norm_xi_constraint": float(norm_xi_constraint),
    }


def _apply_sotuto_null_space_descent(
    solver: RieszVelocitySolver,
    update: RieszVelocityUpdate,
    phi: fem.Function,
    *,
    volume: float,
    target_volume: float,
    mesh_size: float,
    max_norm_xi_objective: float,
    objective_weight: float,
    constraint_weight: float,
    eps: float,
) -> dict[str, float]:
    """Replace the raw Riesz speed by ``sotuto/base`` null-space descent."""
    speed_values, diagnostics = _sotuto_null_space_descent_values(
        solver,
        update.shape_speed_values,
        update.volume_speed_values,
        volume=volume,
        target_volume=target_volume,
        mesh_size=mesh_size,
        max_norm_xi_objective=max_norm_xi_objective,
        objective_weight=objective_weight,
        constraint_weight=constraint_weight,
        eps=eps,
    )
    _set_extension_speed(update.extension, phi, speed_values)
    volume_rate = -_global_dot(solver.space.mesh, speed_values, update.volume_rhs_values)
    update.lambda_volume = float(diagnostics["lambda_volume"])
    update.rates["volume_rate_predicted"] = float(volume_rate)
    diagnostics["volume_rate_predicted"] = float(volume_rate)
    return diagnostics


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
) -> RieszVelocityUpdate:
    """Cut-interface compliance derivative projected by an H1 Riesz solve."""
    speed_shape, shape_rhs_values = _solve_riesz_rhs(
        solver,
        shape_rhs,
        name="riesz_shape_speed",
    )
    speed_volume, volume_rhs_values = _solve_riesz_rhs(
        solver,
        volume_rhs,
        name="riesz_volume_speed",
    )
    # With phi < 0 in the solid and phi <- phi - dt*s, positive speed moves
    # the zero contour toward increasing phi and increases solid volume.
    shape_values = np.asarray(speed_shape.x.array, dtype=float)
    volume_values = np.asarray(speed_volume.x.array, dtype=float)
    comm = solver.space.mesh.comm
    rate_shape = -float(
        comm.allreduce(float(np.dot(shape_values, volume_rhs_values)), op=MPI.SUM)
    )
    rate_constraint = -float(
        comm.allreduce(float(np.dot(volume_values, volume_rhs_values)), op=MPI.SUM)
    )
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
    return RieszVelocityUpdate(
        extension=NormalExtensionResult(speed, velocity, signed_distance),
        lambda_volume=float(lambda_volume),
        rates=rates,
        shape_speed_values=shape_values.copy(),
        shape_rhs_values=shape_rhs_values,
        volume_rhs_values=volume_rhs_values,
        volume_speed_values=volume_values.copy(),
    )


def _apply_lbfgs_update(
    state: LBFGSState,
    solver: RieszVelocitySolver,
    update: RieszVelocityUpdate,
    phi: fem.Function,
    *,
    memory: int,
    curvature_tol: float,
    damping: float,
    volume_control: str,
    target_rate: float,
    lambda_limit: float,
) -> dict[str, float | int]:
    """Replace the gradient-descent speed by an L-BFGS speed."""
    current_x = np.asarray(phi.x.array, dtype=float)
    gradient = np.asarray(update.extension.speed.x.array, dtype=float)
    _lbfgs_update(
        state,
        current_x,
        gradient,
        memory=memory,
        curvature_tol=curvature_tol,
    )
    direction, descent_dot, reset = _lbfgs_descent_direction(state, gradient)
    speed_values = -direction
    speed_values = (1.0 - damping) * gradient + damping * speed_values

    volume_correction = 0.0
    comm = phi.function_space.mesh.comm
    predicted_rate = -float(
        comm.allreduce(
            float(np.dot(speed_values, update.volume_rhs_values)),
            op=MPI.SUM,
        )
    )
    if volume_control == "rate":
        rate_constraint = float(update.rates["volume_rate_constraint"])
        if np.isfinite(rate_constraint) and abs(rate_constraint) > 1.0e-14:
            volume_correction = _choose_volume_multiplier(
                solver,
                speed_values,
                update.volume_speed_values,
                predicted_rate,
                rate_constraint,
                target_rate,
                lambda_limit,
            )
            speed_values += volume_correction * update.volume_speed_values
            predicted_rate += volume_correction * rate_constraint

    _set_extension_speed(update.extension, phi, speed_values)
    update.lambda_volume += volume_correction
    update.rates["volume_rate_predicted"] = float(predicted_rate)
    return {
        "lbfgs_pairs": len(state.s_vectors),
        "lbfgs_curvature": float(state.last_curvature),
        "lbfgs_descent_dot": float(descent_dot),
        "lbfgs_reset": int(reset),
        "lbfgs_update_accepted": int(state.last_update_accepted),
        "lbfgs_volume_correction": float(volume_correction),
    }


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
    volume = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form(fem.Constant(msh, default_scalar_type(1.0)) * dx_solid)
    )
    compliance_density = _elastic_energy_density_expr(uh, mu, lambda_lame)
    elastic_energy = cutfemx.fem.assemble_scalar(
        cutfemx.fem.form(compliance_density * dx_solid)
    )
    compliance = float(msh.comm.allreduce(compliance, op=MPI.SUM))
    volume = float(msh.comm.allreduce(volume, op=MPI.SUM))
    elastic_energy = float(msh.comm.allreduce(elastic_energy, op=MPI.SUM))
    return {
        "compliance": compliance,
        "volume": volume,
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


def _advect_level_set_nodal(
    solver: LevelSetAdvectionSolver,
    phi: fem.Function,
    extension: NormalExtensionResult,
    dt: float,
) -> None:
    """Cheap diagnostic level-set update without a global transport solve."""
    old_values = np.asarray(phi.x.array, dtype=float).copy()
    grad_norm = np.zeros_like(old_values)
    for dof, (nbrs, weights) in enumerate(
        zip(
            solver.nodal_gradient.neighbors,
            solver.nodal_gradient.weights,
        )
    ):
        if nbrs.size == 0:
            continue
        grad = weights @ (old_values[nbrs] - old_values[dof])
        grad_norm[dof] = np.linalg.norm(grad)

    trial_values = (
        old_values
        - dt * np.asarray(extension.speed.x.array, dtype=float) * grad_norm
    )
    if solver.fixed_dofs.size > 0:
        trial_values[solver.fixed_dofs] = old_values[solver.fixed_dofs]
    phi.x.array[:] = trial_values
    phi.x.scatter_forward()


def _points_3d(points: np.ndarray, gdim: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    out = np.zeros((points.shape[0], 3), dtype=np.float64)
    out[:, :gdim] = points[:, :gdim]
    return out


def _locate_cells_for_points(msh: mesh.Mesh, points: np.ndarray) -> np.ndarray:
    """Locate cells for point evaluation, falling back to nearest cells."""
    tdim = msh.topology.dim
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    tree = geometry.bb_tree(msh, tdim, padding=1.0e-12)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(msh, candidates, points)
    for point in range(points.shape[0]):
        links = collisions.links(point)
        if len(links) > 0:
            cells[point] = links[0]

    missing = np.flatnonzero(cells < 0)
    if missing.size > 0:
        num_cells = msh.topology.index_map(tdim).size_local
        local_cells = np.arange(num_cells, dtype=np.int32)
        midpoint_tree = geometry.create_midpoint_tree(msh, tdim, local_cells)
        cells[missing] = geometry.compute_closest_entity(
            tree,
            midpoint_tree,
            msh,
            points[missing],
        )
    return cells


def _evaluate_function_at_points(
    function: fem.Function,
    points: np.ndarray,
) -> np.ndarray:
    msh = function.function_space.mesh
    points3 = _points_3d(points, msh.geometry.dim)
    cells = _locate_cells_for_points(msh, points3)
    return np.asarray(function.eval(points3, cells), dtype=float)


def _advect_level_set_characteristics(
    solver: LevelSetAdvectionSolver,
    phi: fem.Function,
    extension: NormalExtensionResult,
    dt: float,
) -> None:
    """Semi-Lagrangian characteristic update inspired by sotuto's advect step."""
    if phi.function_space.mesh.comm.size != 1:
        raise RuntimeError("Characteristic level-set advection is serial-only.")

    msh = phi.function_space.mesh
    gdim = msh.geometry.dim
    old_values = np.asarray(phi.x.array, dtype=float).copy()
    dof_points = phi.function_space.tabulate_dof_coordinates()[:, :gdim]
    velocity0 = _evaluate_function_at_points(extension.velocity, dof_points)[:, :gdim]
    half_points = dof_points - 0.5 * dt * velocity0
    velocity_mid = _evaluate_function_at_points(extension.velocity, half_points)[
        :, :gdim
    ]
    departure_points = dof_points - dt * velocity_mid
    new_values = _evaluate_function_at_points(phi, departure_points).reshape(-1)
    if solver.fixed_dofs.size > 0:
        new_values[solver.fixed_dofs] = old_values[solver.fixed_dofs]
    phi.x.array[:] = new_values
    phi.x.scatter_forward()


def _advect_level_set(
    solver: LevelSetAdvectionSolver,
    phi: fem.Function,
    extension: NormalExtensionResult,
    dt: float,
    method: str,
) -> None:
    if method == "supg":
        _advect_level_set_supg(solver, phi, extension, dt)
    elif method == "characteristics":
        _advect_level_set_characteristics(solver, phi, extension, dt)
    elif method == "nodal":
        _advect_level_set_nodal(solver, phi, extension, dt)
    else:
        raise ValueError(f"Unknown level-set advection method: {method!r}")


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


def _cut_data_topology_diagnostics(
    msh: mesh.Mesh,
    cut_data: cutfemx.CutData,
    order: int,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
) -> dict[str, int]:
    """Return active-component diagnostics from existing cut data."""
    components = _cut_data_components(
        msh,
        cut_data,
        order,
        left_facets,
        load_facets,
    )
    return _active_cell_diagnostics_from_components(components)


def _level_set_components(
    phi: fem.Function,
    args: argparse.Namespace,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
) -> list[ActiveSolidComponent]:
    """Return active-solid components for the current level set."""
    cut_data = _cut_level_set(phi, args)
    return _cut_data_components(
        phi.function_space.mesh,
        cut_data,
        args.order,
        left_facets,
        load_facets,
    )


def _cut_data_components(
    msh: mesh.Mesh,
    cut_data: cutfemx.CutData,
    order: int,
    left_facets: np.ndarray,
    load_facets: np.ndarray,
) -> list[ActiveSolidComponent]:
    """Return active-solid components from existing cut data."""
    solid_cells = cutfemx.locate_entities(cut_data, "phi<0")
    solid_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    return _active_solid_components(
        msh,
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
    convergence_csv = args.convergence_csv or (args.output_dir / "convergence.csv")
    convergence_csv = Path(convergence_csv)
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
    if args.volume_control == "sotuto":
        fixed_update_facets = np.asarray(right_tags.indices, dtype=np.int32)
    else:
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
    lbfgs_state = LBFGSState([], [], [])
    lbfgs_retry_countdown = 0
    adaptive_step_state = AdaptiveGradientStepState(accepted_dt=dt)
    sotuto_max_norm_xi_objective = 0.0

    tracemalloc.start()
    tracemalloc_baseline = (
        tracemalloc.take_snapshot() if args.tracemalloc_interval > 0 else None
    )
    with ProfileWriter(profile_csv) as profile, ConvergenceWriter(
        convergence_csv
    ) as convergence:
        for iteration in range(args.iterations):
            iteration_start = time.perf_counter()
            row: dict[str, float | int] = {
                "iteration": iteration,
                "cells": msh.topology.index_map(msh.topology.dim).size_local,
                "target_volume": target_volume,
                "objective": 0.0,
                "volume_constraint": 0.0,
                "dt": dt,
                "step_dt_previous": dt,
                "step_dt_bb": dt,
                "step_dt_motion_cap": np.inf,
                "step_dt_proposed": dt,
                "step_dt_accepted": dt,
                "step_bb_accepted": 0,
                "lagrangian": 0.0,
                "alm_lambda": alm.lagrange_multiplier,
                "alm_penalty": alm.penalty,
                "alm_multiplier": 0.0,
                "sotuto_penalty": 0.0,
                "sotuto_alpha_objective": 0.0,
                "sotuto_alpha_constraint": 0.0,
                "sotuto_norm_xi_objective": 0.0,
                "sotuto_norm_xi_constraint": 0.0,
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
                "line_search_failed": 0,
                "line_search_trial_compliance": 0.0,
                "line_search_trial_volume": 0.0,
                "line_search_trial_merit": 0.0,
                "line_search_armijo_rhs": 0.0,
                "line_search_actual_delta": 0.0,
                "min_component_cells": 0,
                "max_component_cells": 0,
                "volume_rate_target": 0.0,
                "volume_rate_shape": 0.0,
                "volume_rate_constraint": 0.0,
                "volume_rate_predicted": 0.0,
                "objective_rate_predicted": 0.0,
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
                "grad_J_h1": 0.0,
                "grad_J_max": 0.0,
                "grad_L_h1": 0.0,
                "grad_L_max": 0.0,
                "lbfgs_pairs": 0,
                "lbfgs_curvature": 0.0,
                "lbfgs_descent_dot": 0.0,
                "lbfgs_reset": 0,
                "lbfgs_update_accepted": 0,
                "lbfgs_volume_correction": 0.0,
                "lbfgs_fallback_to_gradient": 0,
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
                row["objective"] = compliance
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
                if args.volume_control == "rate":
                    target_delta = _target_volume_delta(
                        volume,
                        target_volume,
                        reference_volume,
                        args.max_volume_step,
                    )
                    target_rate = target_delta / dt if dt > 0.0 else 0.0
                elif args.volume_control == "alm":
                    _update_augmented_lagrangian(alm, volume_constraint)
                    volume_multiplier = _alm_velocity_multiplier(
                        alm, volume_constraint
                    )
                elif args.volume_control == "sotuto":
                    volume_multiplier = 0.0
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
                velocity_update = _solve_riesz_velocity_volume(
                    riesz_solver,
                    phi,
                    riesz_shape_rhs,
                    riesz_volume_rhs,
                    dx_interface,
                    volume_multiplier=volume_multiplier,
                    target_rate=target_rate,
                    lambda_limit=args.volume_lambda_limit,
                )
                extension = velocity_update.extension
                if args.volume_control == "sotuto":
                    row.update(
                        _apply_sotuto_null_space_descent(
                            riesz_solver,
                            velocity_update,
                            phi,
                            volume=volume,
                            target_volume=target_volume,
                            mesh_size=hmin,
                            max_norm_xi_objective=sotuto_max_norm_xi_objective,
                            objective_weight=args.sotuto_objective_weight,
                            constraint_weight=args.sotuto_constraint_weight,
                            eps=args.sotuto_eps,
                        )
                    )
                    if iteration == args.sotuto_lock_objective_norm_iteration:
                        sotuto_max_norm_xi_objective = float(
                            row["sotuto_norm_xi_objective"]
                        )
                    row["lagrangian"] = _sotuto_merit(
                        compliance,
                        volume,
                        target_volume,
                        float(row["lambda_volume"]),
                        float(row["sotuto_penalty"]),
                        float(row["sotuto_alpha_objective"]),
                        float(row["sotuto_alpha_constraint"]),
                    )
                gradient_speed_values = np.asarray(
                    extension.speed.x.array,
                    dtype=float,
                ).copy()
                gradient_lambda_volume = velocity_update.lambda_volume
                gradient_rates = dict(velocity_update.rates)
                use_lbfgs = (
                    args.optimizer == "lbfgs"
                    and args.volume_control != "sotuto"
                    and lbfgs_retry_countdown == 0
                )
                if args.optimizer == "lbfgs" and lbfgs_retry_countdown > 0:
                    lbfgs_retry_countdown -= 1
                if use_lbfgs:
                    row.update(
                        _apply_lbfgs_update(
                            lbfgs_state,
                            riesz_solver,
                            velocity_update,
                            phi,
                            memory=args.lbfgs_memory,
                            curvature_tol=args.lbfgs_curvature_tol,
                            damping=args.lbfgs_damping,
                            volume_control=args.volume_control,
                            target_rate=target_rate,
                            lambda_limit=args.volume_lambda_limit,
                        )
                    )
                row["lambda_volume"] = velocity_update.lambda_volume
                row.update(velocity_update.rates)
                raw_update_speed_values = np.asarray(
                    extension.speed.x.array,
                    dtype=float,
                ).copy()
                row["grad_J_h1"] = _h1_scale_from_values(
                    riesz_solver,
                    velocity_update.shape_speed_values,
                )
                row["grad_J_max"] = _global_max_abs(
                    msh,
                    velocity_update.shape_speed_values,
                )
                row["grad_L_h1"] = _h1_scale_from_values(
                    riesz_solver,
                    raw_update_speed_values,
                )
                row["grad_L_max"] = _global_max_abs(
                    msh,
                    raw_update_speed_values,
                )
                velocity_scale_reference_before_lbfgs = velocity_scale_reference
                if args.volume_control == "sotuto":
                    velocity_h1_scale = _extension_h1_scale(riesz_solver, extension.speed)
                    raw_velocity_max = _extension_max_norm(extension)
                    row["raw_velocity_max"] = raw_velocity_max
                    row["velocity_h1_scale"] = velocity_h1_scale
                    row["velocity_scale_floor"] = 0.0
                    row["velocity_scale"] = 1.0
                    row["velocity_scale_limited"] = 0
                    row["velocity_max"] = raw_velocity_max
                    row["speed_min"], row["speed_max"] = _function_min_max(
                        extension.speed
                    )
                else:
                    velocity_scale_reference, _ = _normalise_extension_and_update_row(
                        extension,
                        riesz_solver,
                        row,
                        velocity_scale_reference,
                    )
                row["objective_rate_predicted"] = _predicted_compliance_rate(
                    riesz_solver,
                    np.asarray(extension.speed.x.array, dtype=float),
                    velocity_update.shape_rhs_values,
                )
                if args.volume_control == "sotuto":
                    motion_cap = _motion_dt_cap(
                        hmin,
                        float(row["velocity_max"]),
                        args.sotuto_motion_cfl,
                    )
                    proposed_dt = min(float(dt), motion_cap)
                    if not np.isfinite(proposed_dt) or proposed_dt <= 0.0:
                        proposed_dt = float(dt)
                    row.update(
                        {
                            "step_dt_previous": dt,
                            "step_dt_bb": dt,
                            "step_dt_motion_cap": float(motion_cap),
                            "step_dt_proposed": float(proposed_dt),
                            "step_bb_accepted": 0,
                        }
                    )
                    row["dt"] = float(proposed_dt)
                else:
                    row.update(
                        _adaptive_gradient_dt(
                            adaptive_step_state,
                            msh,
                            np.asarray(phi.x.array, dtype=float),
                            np.asarray(extension.speed.x.array, dtype=float),
                            dt,
                            hmin,
                            float(row["velocity_max"]),
                            args.motion_cfl,
                            enabled=(
                                args.adaptive_step and args.optimizer == "gradient"
                            ),
                        )
                    )
                    row["dt"] = float(row["step_dt_proposed"])
                    predicted_rate = float(row["volume_rate_predicted"])
                    if target_delta != 0.0 and predicted_rate * target_delta > 0.0:
                        max_delta = min(
                            abs(target_delta),
                            args.max_volume_step * reference_volume,
                        )
                        predicted_delta = float(row["dt"]) * predicted_rate
                        if abs(predicted_delta) > max_delta:
                            row["dt"] = max_delta / abs(predicted_rate)
                        row["target_delta_volume"] = float(row["dt"]) * predicted_rate
                    row["step_dt_proposed"] = float(row["dt"])

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
                or not np.isfinite(float(row["lagrangian"]))
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
                initial_base_dt = float(row["dt"])
                base_dt = initial_base_dt
                accepted_update = False
                if args.volume_control == "sotuto":
                    max_backtracks = args.sotuto_max_line_search - 1
                else:
                    max_backtracks = args.armijo_backtracks
                fallback_to_gradient = False
                line_search_rejected = False
                while True:
                    for attempt in range(max_backtracks + 1):
                        if attempt > 0:
                            phi.x.array[:] = phi_before_update
                            phi.x.scatter_forward()
                        if args.volume_control == "sotuto":
                            raw_trial_dt = (
                                base_dt * args.sotuto_line_search_decay**attempt
                            )
                            if base_dt >= args.sotuto_min_coef:
                                trial_dt = max(args.sotuto_min_coef, raw_trial_dt)
                            else:
                                trial_dt = raw_trial_dt
                        else:
                            trial_dt = base_dt / (2**attempt)
                        row["advection_backtracks"] = attempt
                        row["dt"] = trial_dt

                        with _phase(row, "advection"):
                            _advect_level_set(
                                advection_solver,
                                phi,
                                extension,
                                trial_dt,
                                args.advection_method,
                            )
                            current_cut_data, volume_after_advection = (
                                _cut_level_set_and_solid_volume(phi, args)
                            )
                            row["volume_after_advection"] = volume_after_advection

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
                            if redistanced:
                                current_cut_data, raw_reinit_volume = (
                                    _cut_level_set_and_solid_volume(phi, args)
                                )
                            else:
                                raw_reinit_volume = row["volume_after_advection"]
                            row["volume_after_raw_reinitialization"] = raw_reinit_volume
                            if (
                                redistanced
                                and args.reinit_volume_correction
                                and args.volume_control != "sotuto"
                            ):
                                (
                                    shift,
                                    corrected_volume,
                                    current_cut_data,
                                ) = _apply_reinit_volume_correction(
                                    phi,
                                    args,
                                    float(row["volume_after_advection"]),
                                    float(raw_reinit_volume),
                                    current_cut_data,
                                )
                                row["reinit_volume_shift"] = shift
                                row["volume_after_reinitialization"] = corrected_volume
                            else:
                                row["volume_after_reinitialization"] = raw_reinit_volume
                            row["phi_min"], row["phi_max"] = _function_min_max(phi)

                        topology_after = _cut_data_topology_diagnostics(
                            msh,
                            current_cut_data,
                            args.order,
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
                                _, filtered_volume = _cut_level_set_and_solid_volume(
                                    phi,
                                    args,
                                )
                                row["volume_after_raw_reinitialization"] = filtered_volume
                                row["volume_after_reinitialization"] = filtered_volume
                                row["phi_min"], row["phi_max"] = _function_min_max(phi)

                        if topology_after["floating_components"] == 0:
                            needs_line_search = bool(
                                args.volume_control == "sotuto"
                                or args.armijo_line_search
                                and (
                                    args.optimizer == "gradient"
                                    or args.lbfgs_line_search
                                )
                            )
                            if needs_line_search:
                                with _phase(row, "line_search"):
                                    trial = _evaluate_state_objectives(
                                        phi,
                                        args,
                                        V,
                                        u,
                                        v,
                                        elasticity_bcs,
                                        ds_load,
                                        traction,
                                        mu,
                                        lmbda,
                                    )
                                row["line_search_trial_compliance"] = float(
                                    trial["compliance"]
                                )
                                row["line_search_trial_volume"] = float(
                                    trial["volume"]
                                )
                                trial_finite = (
                                    not trial["solve_singular"]
                                    and not trial["solve_nonfinite"]
                                    and np.isfinite(float(trial["compliance"]))
                                    and np.isfinite(float(trial["volume"]))
                                )
                                if args.volume_control == "sotuto":
                                    trial_merit = _sotuto_merit(
                                        float(trial["compliance"]),
                                        float(trial["volume"]),
                                        target_volume,
                                        float(row["lambda_volume"]),
                                        float(row["sotuto_penalty"]),
                                        float(row["sotuto_alpha_objective"]),
                                        float(row["sotuto_alpha_constraint"]),
                                    )
                                    merit_limit = float(row["lagrangian"]) + (
                                        args.sotuto_merit_tolerance
                                        * abs(float(row["lagrangian"]))
                                    )
                                    row["line_search_trial_merit"] = trial_merit
                                    row["line_search_armijo_rhs"] = merit_limit
                                    row["line_search_actual_delta"] = (
                                        trial_merit - float(row["lagrangian"])
                                    )
                                    line_search_ok = trial_finite and (
                                        trial_merit < merit_limit
                                        or attempt >= max_backtracks
                                        or (
                                            base_dt >= args.sotuto_min_coef
                                            and trial_dt <= args.sotuto_min_coef
                                        )
                                    )
                                else:
                                    armijo_rhs = _armijo_rhs(
                                        float(row["compliance"]),
                                        float(row["objective_rate_predicted"]),
                                        trial_dt,
                                        args.armijo_sufficient_decrease,
                                    )
                                    row["line_search_armijo_rhs"] = armijo_rhs
                                    row["line_search_actual_delta"] = (
                                        float(trial["compliance"])
                                        - float(row["compliance"])
                                    )
                                    line_search_ok = (
                                        trial_finite
                                        and float(trial["compliance"]) <= armijo_rhs
                                    )
                                if not line_search_ok:
                                    if attempt >= max_backtracks:
                                        line_search_rejected = True
                                        break
                                    continue
                            accepted_update = True
                            if args.volume_control == "sotuto":
                                dt = min(
                                    args.sotuto_max_coef,
                                    args.sotuto_line_search_growth * trial_dt,
                                )
                            else:
                                dt = trial_dt
                            row["step_dt_accepted"] = trial_dt
                            if (
                                args.volume_control != "sotuto"
                                and args.adaptive_step
                                and args.optimizer == "gradient"
                            ):
                                _accept_adaptive_gradient_step(
                                    adaptive_step_state,
                                    phi_before_update,
                                    np.asarray(extension.speed.x.array, dtype=float),
                                    trial_dt,
                                )
                            break

                    if accepted_update:
                        break
                    lbfgs_line_search_failed = (
                        line_search_rejected
                        and args.optimizer == "lbfgs"
                        and args.armijo_line_search
                        and args.lbfgs_line_search
                        and int(row["lbfgs_pairs"]) > 0
                        and not fallback_to_gradient
                    )
                    if not lbfgs_line_search_failed:
                        break

                    fallback_to_gradient = True
                    line_search_rejected = False
                    row["lbfgs_fallback_to_gradient"] = 1
                    row["lbfgs_reset"] = 1
                    row["lbfgs_pairs"] = 0
                    row["lbfgs_update_accepted"] = 0
                    row["lbfgs_curvature"] = 0.0
                    row["lbfgs_descent_dot"] = 0.0
                    row["lbfgs_volume_correction"] = 0.0
                    _clear_lbfgs_state(lbfgs_state)
                    lbfgs_retry_countdown = args.lbfgs_retry_interval
                    phi.x.array[:] = phi_before_update
                    phi.x.scatter_forward()
                    _set_extension_speed(extension, phi, gradient_speed_values)
                    row["lambda_volume"] = gradient_lambda_volume
                    row.update(gradient_rates)
                    velocity_scale_reference = velocity_scale_reference_before_lbfgs
                    velocity_scale_reference, _ = _normalise_extension_and_update_row(
                        extension,
                        riesz_solver,
                        row,
                        velocity_scale_reference,
                    )
                    row["objective_rate_predicted"] = _predicted_compliance_rate(
                        riesz_solver,
                        np.asarray(extension.speed.x.array, dtype=float),
                        velocity_update.shape_rhs_values,
                    )
                    predicted_rate = float(row["volume_rate_predicted"])
                    row["dt"] = initial_base_dt
                    row.update(
                        _adaptive_gradient_dt(
                            adaptive_step_state,
                            msh,
                            np.asarray(phi.x.array, dtype=float),
                            np.asarray(extension.speed.x.array, dtype=float),
                            dt,
                            hmin,
                            float(row["velocity_max"]),
                            args.motion_cfl,
                            enabled=False,
                        )
                    )
                    row["dt"] = min(initial_base_dt, float(row["step_dt_proposed"]))
                    if target_delta != 0.0 and predicted_rate * target_delta > 0.0:
                        max_delta = min(
                            abs(target_delta),
                            args.max_volume_step * reference_volume,
                        )
                        predicted_delta = float(row["dt"]) * predicted_rate
                        if abs(predicted_delta) > max_delta:
                            row["dt"] = max_delta / abs(predicted_rate)
                        row["target_delta_volume"] = float(row["dt"]) * predicted_rate
                    row["step_dt_proposed"] = float(row["dt"])
                    base_dt = float(row["dt"])

                if not accepted_update:
                    phi.x.array[:] = phi_before_update
                    phi.x.scatter_forward()
                    row["topology_invalid"] = 1
                    row["volume_after_advection"] = row["volume"]
                    row["volume_after_raw_reinitialization"] = row["volume"]
                    row["volume_after_reinitialization"] = row["volume"]
                    row["phi_min"], row["phi_max"] = _function_min_max(phi)
                    row["line_search_failed"] = int(line_search_rejected)
                    stop_after_row = True

            current, peak = tracemalloc.get_traced_memory()
            row["time_iteration"] = time.perf_counter() - iteration_start
            row["rss_bytes"] = _rss_bytes()
            row["tracemalloc_current_bytes"] = int(current)
            row["tracemalloc_peak_bytes"] = int(peak)

            history.append(dict(row))
            profile.write(row)
            convergence.write(row)
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
                    if row["line_search_failed"]:
                        if args.volume_control == "sotuto":
                            print(
                                "Stopping optimization after the sotuto merit "
                                "line search could not find an acceptable step."
                            )
                        else:
                            print(
                                "Stopping optimization after the Armijo line search "
                                "could not find an acceptable compliance step."
                            )
                    elif row["topology_invalid"]:
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
        convergence_csv=convergence_csv,
    )


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_output_dir: Path = Path("compliance_optimization_output"),
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description
        or (
            "CutFEM compliance topology optimization cantilever demo. "
            "Armijo line search is enabled by default."
        )
    )

    def hidden(*names, **kwargs) -> None:
        kwargs.setdefault("default", argparse.SUPPRESS)
        kwargs["help"] = argparse.SUPPRESS
        parser.add_argument(*names, **kwargs)

    parser.set_defaults(
        domain_length=2.0,
        domain_height=1.0,
        cell_type="quadrilateral",
        initial_geometry="sotuto",
        hole_rows=2,
        hole_cols=4,
        hole_radius=0.107,
        optimizer="gradient",
        lbfgs_memory=5,
        lbfgs_curvature_tol=1.0e-10,
        lbfgs_damping=0.25,
        lbfgs_line_search=True,
        lbfgs_line_search_backtracks=0,
        lbfgs_retry_interval=4,
        cfl=0.1,
        adaptive_step=True,
        armijo_line_search=True,
        armijo_sufficient_decrease=1.0e-4,
        armijo_backtracks=8,
        volume_control="sotuto",
        volume_lambda_limit=0.0,
        alm_lambda=0.1,
        alm_penalty=1.0,
        alm_penalty_limit=100.0,
        alm_penalty_multiplier=1.1,
        alm_auto_scale=True,
        velocity_smoothing=None,
        velocity_alpha=1.0e-2,
        supg_tau_scale=1.0,
        order=3,
        young_modulus=210000.0,
        poisson_ratio=0.33,
        traction=10.0,
        load_center=0.5,
        load_half_height=0.05,
        ghost_gamma=1.0e-5,
        cut_approximation="linear",
        cut_approximation_order=1,
        max_refinement_iterations=0,
        edge_max_depth=20,
        reinit_max_iter=600,
        reinit_tol=1.0e-10,
        initial_reinit=True,
        reinit_interval=1,
        reinit_volume_correction=True,
        reinit_volume_correction_limit=0.05,
        xdmf_dir=None,
        write_vtk=False,
        failure_snapshot=True,
        topology_diagnostics=True,
        tracemalloc_interval=0,
        tracemalloc_dir=None,
        profile_csv=None,
        convergence_csv=None,
        gc_each_step=False,
        shape_check_interval=0,
        shape_check_step=1.0e-4,
        sotuto_objective_weight=1.0,
        sotuto_constraint_weight=0.1,
        sotuto_eps=1.0e-6,
        sotuto_max_line_search=3,
        sotuto_max_coef=2.0,
        sotuto_min_coef=0.02,
        sotuto_motion_cfl=0.15,
        sotuto_line_search_decay=0.6,
        sotuto_line_search_growth=1.1,
        sotuto_merit_tolerance=0.02,
        sotuto_lock_objective_norm_iteration=50,
    )

    parser.add_argument("--n", type=int, default=32, help="Cells in the vertical direction.")
    parser.add_argument("--iterations", type=int, default=200, help="Optimization steps.")
    parser.add_argument("--volume-fraction", type=float, default=0.70)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help=(
            "Initial pseudo-time step/coefficient. Defaults to 1 for sotuto "
            "volume control and 1e-3 otherwise; use 0 to initialize from --cfl*h."
        ),
    )
    parser.add_argument(
        "--max-volume-step",
        type=float,
        default=5.0e-4,
        help="Maximum reference-domain volume fraction changed per step.",
    )
    parser.add_argument(
        "--motion-cfl",
        type=float,
        default=0.15,
        help="Maximum zero-contour motion per step in units of h.",
    )
    parser.add_argument(
        "--no-armijo",
        action="store_false",
        dest="armijo_line_search",
        help="Disable the post-update Armijo compliance line search.",
    )
    parser.add_argument(
        "--advection-method",
        choices=("supg", "characteristics", "nodal"),
        default="supg",
        help=(
            "Move phi with the SUPG solve, sotuto-inspired semi-Lagrangian "
            "characteristics, or cheap diagnostic nodal update."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--xdmf-interval",
        type=int,
        default=10,
        help="Write XDMF snapshots every N iterations. Use 0 to disable.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Skip interactive plotting and write the profile CSV.",
    )
    parser.add_argument(
        "--convergence-csv",
        type=Path,
        default=None,
        help="Override the scalar convergence CSV path.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show PyVista.")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--history", type=Path, default=None)

    hidden("--domain-length", type=float)
    hidden("--domain-height", type=float)
    hidden("--cell-type", choices=("triangle", "quadrilateral"))
    hidden("--initial-geometry", choices=("holes", "opticut", "sotuto"))
    hidden("--hole-rows", type=int)
    hidden("--hole-cols", type=int)
    hidden("--hole-radius", type=float)
    hidden("--optimizer", choices=("lbfgs", "gradient"))
    hidden("--lbfgs-memory", type=int)
    hidden("--lbfgs-curvature-tol", type=float)
    hidden("--lbfgs-damping", type=float)
    hidden("--lbfgs-line-search", action=argparse.BooleanOptionalAction)
    hidden("--lbfgs-line-search-backtracks", type=int)
    hidden("--lbfgs-retry-interval", type=int)
    hidden("--cfl", type=float)
    hidden("--adaptive-step", action=argparse.BooleanOptionalAction)
    hidden(
        "--armijo-line-search",
        action=argparse.BooleanOptionalAction,
        dest="armijo_line_search",
    )
    hidden("--armijo-sufficient-decrease", type=float)
    hidden("--armijo-backtracks", type=int)
    hidden("--volume-control", choices=("alm", "rate", "sotuto"))
    hidden("--volume-lambda-limit", type=float)
    hidden("--alm-lambda", type=float)
    hidden("--alm-penalty", type=float)
    hidden("--alm-penalty-limit", type=float)
    hidden("--alm-penalty-multiplier", type=float)
    hidden("--alm-auto-scale", action=argparse.BooleanOptionalAction)
    hidden("--velocity-smoothing", type=float)
    hidden("--velocity-alpha", type=float)
    hidden("--supg-tau-scale", type=float)
    hidden("--order", type=int)
    hidden("--young-modulus", type=float)
    hidden("--poisson-ratio", type=float)
    hidden("--traction", type=float)
    hidden("--load-center", type=float)
    hidden("--load-half-height", type=float)
    hidden("--ghost-gamma", type=float)
    hidden("--cut-approximation", choices=("auto", "linear", "iso_p1"))
    hidden("--cut-approximation-order", type=int)
    hidden("--max-refinement-iterations", type=int)
    hidden("--edge-max-depth", type=int)
    hidden("--reinit-max-iter", type=int)
    hidden("--reinit-tol", type=float)
    hidden("--initial-reinit", action=argparse.BooleanOptionalAction)
    hidden("--reinit-interval", type=int)
    hidden("--reinit-volume-correction", action=argparse.BooleanOptionalAction)
    hidden("--reinit-volume-correction-limit", type=float)
    hidden("--xdmf-dir", type=Path)
    hidden("--write-vtk", action="store_true")
    hidden("--failure-snapshot", action=argparse.BooleanOptionalAction)
    hidden("--topology-diagnostics", action=argparse.BooleanOptionalAction)
    hidden("--tracemalloc-interval", type=int)
    hidden("--tracemalloc-dir", type=Path)
    hidden("--profile-csv", type=Path)
    hidden("--gc-each-step", action="store_true")
    hidden("--shape-check-interval", type=int)
    hidden("--shape-check-step", type=float)
    hidden("--sotuto-objective-weight", type=float)
    hidden("--sotuto-constraint-weight", type=float)
    hidden("--sotuto-eps", type=float)
    hidden("--sotuto-max-line-search", type=int)
    hidden("--sotuto-max-coef", type=float)
    hidden("--sotuto-min-coef", type=float)
    hidden("--sotuto-motion-cfl", type=float)
    hidden("--sotuto-line-search-decay", type=float)
    hidden("--sotuto-line-search-growth", type=float)
    hidden("--sotuto-merit-tolerance", type=float)
    hidden("--sotuto-lock-objective-norm-iteration", type=int)

    args = parser.parse_args(argv)
    if args.dt is None:
        args.dt = 1.0 if args.volume_control == "sotuto" else 1.0e-3
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
    if args.lbfgs_memory < 0:
        parser.error("--lbfgs-memory must be nonnegative.")
    if args.lbfgs_curvature_tol < 0.0:
        parser.error("--lbfgs-curvature-tol must be nonnegative.")
    if not (0.0 <= args.lbfgs_damping <= 1.0):
        parser.error("--lbfgs-damping must lie in [0, 1].")
    if args.lbfgs_line_search_backtracks < 0:
        parser.error("--lbfgs-line-search-backtracks must be nonnegative.")
    if args.lbfgs_retry_interval < 0:
        parser.error("--lbfgs-retry-interval must be nonnegative.")
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
    if args.motion_cfl < 0.0:
        parser.error("--motion-cfl must be nonnegative.")
    if args.armijo_sufficient_decrease < 0.0:
        parser.error("--armijo-sufficient-decrease must be nonnegative.")
    if args.armijo_backtracks < 0:
        parser.error("--armijo-backtracks must be nonnegative.")
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
    if args.sotuto_objective_weight <= 0.0:
        parser.error("--sotuto-objective-weight must be positive.")
    if args.sotuto_constraint_weight <= 0.0:
        parser.error("--sotuto-constraint-weight must be positive.")
    if args.sotuto_eps <= 0.0:
        parser.error("--sotuto-eps must be positive.")
    if args.sotuto_max_line_search < 1:
        parser.error("--sotuto-max-line-search must be at least 1.")
    if args.sotuto_max_coef <= 0.0:
        parser.error("--sotuto-max-coef must be positive.")
    if args.sotuto_min_coef <= 0.0:
        parser.error("--sotuto-min-coef must be positive.")
    if args.sotuto_min_coef > args.sotuto_max_coef:
        parser.error("--sotuto-min-coef must not exceed --sotuto-max-coef.")
    if args.sotuto_motion_cfl < 0.0:
        parser.error("--sotuto-motion-cfl must be nonnegative.")
    if not (0.0 < args.sotuto_line_search_decay < 1.0):
        parser.error("--sotuto-line-search-decay must lie in (0, 1).")
    if args.sotuto_line_search_growth < 1.0:
        parser.error("--sotuto-line-search-growth must be at least 1.")
    if args.sotuto_merit_tolerance < 0.0:
        parser.error("--sotuto-merit-tolerance must be nonnegative.")
    if args.sotuto_lock_objective_norm_iteration < 0:
        parser.error("--sotuto-lock-objective-norm-iteration must be nonnegative.")
    return args


def main(
    argv: Sequence[str] | None = None,
    *,
    default_output_dir: Path = Path("compliance_optimization_output"),
    description: str | None = None,
) -> int:
    args = parse_args(
        argv,
        default_output_dir=default_output_dir,
        description=description,
    )
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
        print(f"Wrote convergence CSV to {result.convergence_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
