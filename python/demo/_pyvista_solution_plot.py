"""Shared PyVista solution plots for CutFEMx demos."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cutfemx
import numpy as np


def add_plot_arguments(parser) -> None:
    """Add common interactive PyVista controls to a demo argument parser."""
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the interactive PyVista solution plot.",
    )


def _plot_modules() -> tuple[Any | None, Any | None]:
    try:
        import pyvista

        from dolfinx import plot
    except (ImportError, ModuleNotFoundError):
        print("pyvista and dolfinx.plot are required for interactive solution plotting")
        return None, None
    return pyvista, plot


def _real_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.iscomplexobj(values):
        values = np.real_if_close(values)
    return np.asarray(values, dtype=float)


def _function_values(u) -> np.ndarray:
    values = _real_array(u.x.array)
    block_size = int(getattr(u.function_space.dofmap, "index_map_bs", 1))
    if block_size > 1:
        return values.reshape((-1, block_size))
    return values


def _as_3d_vectors(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("Expected vector data with shape (num_points, components).")
    if values.shape[1] == 3:
        return values
    if values.shape[1] < 3:
        padding = np.zeros((values.shape[0], 3 - values.shape[1]), dtype=values.dtype)
        return np.hstack((values, padding))
    return values[:, :3]


def _attach_function_data(grid, u, name: str) -> str:
    values = _function_values(u)
    if values.ndim == 2:
        values = _as_3d_vectors(values)

    if values.shape[0] == grid.n_points:
        grid.point_data[name] = values
        return "point"
    if values.shape[0] == grid.n_cells:
        grid.cell_data[name] = values
        return "cell"
    if values.ndim == 1 and values.size >= grid.n_points:
        grid.point_data[name] = values[: grid.n_points]
        return "point"
    raise ValueError(
        f"Cannot attach {name!r}: {values.shape[0]} values for "
        f"{grid.n_points} points and {grid.n_cells} cells."
    )


def _grid_from_cut_function(pyvista, plot, u_cut, field_name: str):
    cells, cell_types, points = plot.vtk_mesh(u_cut.function_space)
    grid = pyvista.UnstructuredGrid(cells, cell_types, points)
    if grid.n_points == 0 or grid.n_cells == 0:
        raise RuntimeError("Cannot plot an empty cut mesh.")
    _attach_function_data(grid, u_cut, field_name)
    return grid


def _cut_grid(pyvista, plot, cut_mesh: cutfemx.CutMesh, u, field_name: str):
    u_cut = cutfemx.fem.cut_function(u, cut_mesh)
    u_cut.name = field_name
    return _grid_from_cut_function(pyvista, plot, u_cut, field_name), u_cut


def _data_preference(grid, name: str) -> str:
    return "point" if name in grid.point_data else "cell"


def _data_array(grid, name: str) -> np.ndarray:
    if name in grid.point_data:
        return _real_array(grid.point_data[name])
    if name in grid.cell_data:
        return _real_array(grid.cell_data[name])
    raise KeyError(name)


def _scalar_range(grids: Sequence[object], name: str) -> tuple[float, float]:
    values = [_data_array(grid, name).reshape(-1) for grid in grids]
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    if finite.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        delta = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return (vmin - delta, vmax + delta)
    return (vmin, vmax)


def _domain_extent(grids: Sequence[object]) -> float:
    extent = 0.0
    for grid in grids:
        bounds = grid.bounds
        extent = max(
            extent,
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )
    return extent if np.isfinite(extent) and extent > 0.0 else 1.0


def _scalar_warp_factor(grids: Sequence[object], name: str) -> float:
    value_range = _scalar_range(grids, name)
    span = value_range[1] - value_range[0]
    if not np.isfinite(span) or span <= np.finfo(float).eps:
        return 1.0
    return 0.18 * _domain_extent(grids) / span


def _vector_warp_factor(grid, name: str) -> float:
    vectors = _data_array(grid, name)
    magnitudes = np.linalg.norm(vectors, axis=1)
    max_magnitude = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if not np.isfinite(max_magnitude) or max_magnitude <= np.finfo(float).eps:
        return 1.0
    return 0.18 * _domain_extent([grid]) / max_magnitude


def _apply_view(plotter, view: str) -> None:
    if view == "xy":
        plotter.view_xy()
    elif view == "yz":
        plotter.view_yz()
    elif view == "xz":
        plotter.view_xz()
    else:
        plotter.view_isometric()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(1.12)


def _finish_plot(plotter, show: bool) -> None:
    if show:
        plotter.show()
    else:
        plotter.close()


def _plotter(pyvista, **kwargs):
    return pyvista.Plotter(**kwargs)


def plot_scalar_cut_solution(
    cut_data: cutfemx.CutData,
    selector: str,
    u,
    *,
    title: str,
    field_name: str = "u_h",
    mode: str = "full",
    show: bool = True,
    view: str = "isometric",
    warp: bool = True,
    cmap: str = "viridis",
) -> None:
    """Plot one scalar background solution transferred to a cut mesh."""
    pyvista, plot = _plot_modules()
    if pyvista is None or plot is None:
        return

    cut_mesh = cutfemx.create_cut_mesh(cut_data, selector, mode=mode)
    if cut_mesh.mesh is None:
        raise RuntimeError(f"Cannot plot empty cut mesh for selector {selector!r}.")

    grid, _ = _cut_grid(pyvista, plot, cut_mesh, u, field_name)
    display_grid = grid
    if warp and field_name in grid.point_data:
        display_grid = grid.warp_by_scalar(
            field_name,
            factor=_scalar_warp_factor([grid], field_name),
        )

    plotter = _plotter(pyvista, window_size=(850, 700))
    plotter.add_title(title, font_size=12)
    plotter.add_mesh(
        display_grid,
        scalars=field_name,
        preference=_data_preference(display_grid, field_name),
        cmap=cmap,
        show_edges=True,
        edge_color="#334155",
        line_width=0.5,
        scalar_bar_args={"title": field_name},
    )
    plotter.add_axes()
    _apply_view(plotter, view)
    _finish_plot(plotter, show)


def plot_scalar_cut_domains(
    cut_data: cutfemx.CutData,
    domains: Sequence[tuple[str, object, str]],
    *,
    title: str,
    field_name: str = "u_h",
    mode: str = "full",
    show: bool = True,
    view: str = "isometric",
    warp: bool = True,
    cmap: str = "viridis",
) -> None:
    """Plot multiple scalar background solutions on their physical cut domains."""
    pyvista, plot = _plot_modules()
    if pyvista is None or plot is None:
        return

    prepared = []
    for selector, u, label in domains:
        cut_mesh = cutfemx.create_cut_mesh(cut_data, selector, mode=mode)
        if cut_mesh.mesh is None:
            raise RuntimeError(f"Cannot plot empty cut mesh for selector {selector!r}.")
        grid, _ = _cut_grid(pyvista, plot, cut_mesh, u, field_name)
        prepared.append((grid, label))

    clim = _scalar_range([grid for grid, _ in prepared], field_name)
    warp_factor = _scalar_warp_factor([grid for grid, _ in prepared], field_name)

    plotter = _plotter(pyvista, window_size=(850, 700))
    plotter.add_title(title, font_size=12)
    for i, (grid, label) in enumerate(prepared):
        display_grid = grid
        if warp and field_name in grid.point_data:
            display_grid = grid.warp_by_scalar(field_name, factor=warp_factor)
        plotter.add_mesh(
            display_grid,
            scalars=field_name,
            preference=_data_preference(display_grid, field_name),
            cmap=cmap,
            clim=clim,
            show_edges=True,
            edge_color="#334155",
            line_width=0.5,
            label=label,
            show_scalar_bar=i == len(prepared) - 1,
            scalar_bar_args={"title": field_name},
        )
    plotter.add_legend(background_opacity=0.0)
    plotter.add_axes()
    _apply_view(plotter, view)
    _finish_plot(plotter, show)


def plot_vector_scalar_cut_solution(
    cut_data: cutfemx.CutData,
    selector: str,
    vector,
    scalar,
    *,
    title: str,
    vector_name: str = "u",
    scalar_name: str = "p",
    mode: str = "full",
    show: bool = True,
) -> None:
    """Plot a vector magnitude and a scalar field on the same cut domain."""
    pyvista, plot = _plot_modules()
    if pyvista is None or plot is None:
        return

    cut_mesh = cutfemx.create_cut_mesh(cut_data, selector, mode=mode)
    if cut_mesh.mesh is None:
        raise RuntimeError(f"Cannot plot empty cut mesh for selector {selector!r}.")

    velocity_grid, _ = _cut_grid(pyvista, plot, cut_mesh, vector, vector_name)
    pressure_grid, _ = _cut_grid(pyvista, plot, cut_mesh, scalar, scalar_name)
    vectors = _data_array(velocity_grid, vector_name)
    speed_name = f"|{vector_name}|"
    velocity_grid.point_data[speed_name] = np.linalg.norm(vectors, axis=1)

    plotter = _plotter(
        pyvista,
        shape=(1, 2),
        window_size=(1200, 520),
    )
    plotter.subplot(0, 0)
    plotter.add_text("Velocity magnitude", position="upper_left", font_size=11)
    plotter.add_mesh(
        velocity_grid,
        scalars=speed_name,
        cmap="viridis",
        show_edges=True,
        edge_color="#334155",
        line_width=0.5,
        scalar_bar_args={"title": speed_name},
    )
    stride = max(1, int(np.ceil(velocity_grid.n_points / 250)))
    plotter.add_arrows(
        velocity_grid.points[::stride],
        vectors[::stride],
        mag=0.18,
        color="#111827",
    )
    plotter.add_axes()
    _apply_view(plotter, "xy")

    plotter.subplot(0, 1)
    plotter.add_text("Pressure", position="upper_left", font_size=11)
    plotter.add_mesh(
        pressure_grid,
        scalars=scalar_name,
        preference=_data_preference(pressure_grid, scalar_name),
        cmap="coolwarm",
        show_edges=True,
        edge_color="#334155",
        line_width=0.5,
        scalar_bar_args={"title": scalar_name},
    )
    plotter.add_axes()
    _apply_view(plotter, "xy")
    plotter.add_title(title, font_size=12)
    plotter.link_views()
    _finish_plot(plotter, show)


def plot_deformed_cut_solution(
    cut_data: cutfemx.CutData,
    selector: str,
    displacement,
    scalar,
    *,
    title: str,
    displacement_name: str = "u",
    scalar_name: str = "von_mises",
    mode: str = "full",
    show: bool = True,
) -> None:
    """Plot a vector displacement solution warped and colored by a scalar field."""
    pyvista, plot = _plot_modules()
    if pyvista is None or plot is None:
        return

    cut_mesh = cutfemx.create_cut_mesh(cut_data, selector, mode=mode)
    if cut_mesh.mesh is None:
        raise RuntimeError(f"Cannot plot empty cut mesh for selector {selector!r}.")

    grid, _ = _cut_grid(pyvista, plot, cut_mesh, displacement, displacement_name)
    scalar_cut = cutfemx.fem.cut_function(scalar, cut_mesh)
    scalar_cut.name = scalar_name
    _attach_function_data(grid, scalar_cut, scalar_name)

    deformed = grid.warp_by_vector(
        displacement_name,
        factor=_vector_warp_factor(grid, displacement_name),
    )

    plotter = _plotter(pyvista, window_size=(900, 700))
    plotter.add_title(title, font_size=12)
    plotter.add_mesh(grid, style="wireframe", color="#334155", opacity=0.18)
    plotter.add_mesh(
        deformed,
        scalars=scalar_name,
        preference=_data_preference(deformed, scalar_name),
        cmap="plasma",
        show_edges=False,
        scalar_bar_args={"title": scalar_name},
    )
    plotter.add_axes()
    _apply_view(plotter, "isometric")
    _finish_plot(plotter, show)
