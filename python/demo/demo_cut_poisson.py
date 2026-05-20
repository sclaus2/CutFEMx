# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

# # Poisson problem on an implicitly defined cut domain
#
# This demo illustrates how to solve a scalar Poisson problem on a domain
# described by a level-set function. The physical domain is
#
#     Omega = {x in [-1, 1]^2 : phi(x) < 0}.
#
# The computation is performed on a background mesh. CutFEMx is used to locate
# the active cells, construct quadrature rules on the cut cells, and create a
# visualisation mesh for the cut geometry.
#
# The weak problem solved in this demo is: find ``u`` such that
#
#     integral_Omega grad(u) . grad(v) dx
#   + gamma integral_Gamma u v ds
#   = integral_Omega f v dx
#
# for all test functions ``v``. Here ``Gamma`` is the embedded boundary
# ``phi = 0``.

from __future__ import annotations

import argparse

import numpy as np
import ufl
from mpi4py import MPI

import cutfemx
from dolfinx import fem, la, mesh


# ## Helper functions
#
# The following small utilities are only used for visualisation. They convert
# quadrature point arrays to PyVista coordinates and choose a sensible warp
# factor for plotting scalar fields.


def _points_as_pyvista_coordinates(points: np.ndarray) -> np.ndarray:
    """Return point coordinates in the three-component layout PyVista expects."""
    point_coordinates = np.asarray(points).T
    if point_coordinates.shape[1] == 3:
        return np.ascontiguousarray(point_coordinates)
    if point_coordinates.shape[1] == 2:
        coordinates_3d = np.zeros((point_coordinates.shape[0], 3), dtype=point_coordinates.dtype)
        coordinates_3d[:, :2] = point_coordinates
        return coordinates_3d
    raise ValueError("Can only plot quadrature points with geometry dimension 2 or 3.")


def _point_parent_index(rules) -> np.ndarray:
    """Return one parent cell index per quadrature point."""
    if rules.parent_map is None or rules.offsets is None:
        return np.empty(0, dtype=np.int32)
    counts = np.diff(rules.offsets).astype(np.int64, copy=False)
    return np.repeat(rules.parent_map, counts).astype(np.int32, copy=False)


def _scalar_warp_factor(points: np.ndarray, values: np.ndarray) -> float:
    """Return a geometry-scaled factor for PyVista scalar warping."""
    value_range = float(np.ptp(values)) if values.size > 0 else 0.0
    if not np.isfinite(value_range) or value_range <= np.finfo(float).eps:
        return 1.0

    extents = np.ptp(np.asarray(points)[:, :2], axis=0)
    mesh_width = float(np.max(extents)) if extents.size > 0 else 1.0
    if not np.isfinite(mesh_width) or mesh_width <= np.finfo(float).eps:
        mesh_width = 1.0
    return 0.25 * mesh_width / value_range


# ## Linear solvers
#
# The cut variational forms can be assembled into a DOLFINx ``MatrixCSR`` and
# solved with SciPy, or assembled directly into PETSc objects and solved with a
# PETSc Krylov solver. Both paths use the same active-domain information to
# deactivate degrees of freedom outside the physical domain.


def solve_runtime_system(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
    solver: str,
) -> fem.Function:
    """Solve the runtime system with the selected linear algebra backend."""
    match solver:
        case "scipy":
            return solve_runtime_system_scipy(a, L, V)
        case "petsc":
            return solve_runtime_system_petsc(a, L, V)
        case _:
            raise ValueError(f"Unknown solver '{solver}'.")


def solve_runtime_system_scipy(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
) -> fem.Function:
    """Assemble a MatrixCSR runtime system and solve it with SciPy."""
    from scipy.sparse.linalg import spsolve

    if V.mesh.comm.size != 1:
        raise RuntimeError("The MatrixCSR/SciPy solve path in this demo is serial-only.")

    # Assemble the CutFEMx runtime forms into DOLFINx MatrixCSR/vector objects.
    b = cutfemx.fem.assemble_vector(L)
    b.scatter_reverse(la.InsertMode.add)
    A = cutfemx.fem.assemble_matrix(a)
    A.scatter_reverse()

    # Replace inactive rows by identity rows before handing the matrix to SciPy.
    cutfemx.fem.deactivate_outside(A, b, cutfemx.fem.active_domain(a))

    # SciPy solves the local CSR system; the result is copied back into a Function.
    u_values = spsolve(A.to_scipy(), b.array)
    uh = fem.Function(V)
    uh.x.array[:] = u_values
    uh.x.scatter_forward()

    return uh


def solve_runtime_system_petsc(
    a: cutfemx.fem.CutForm,
    L: cutfemx.fem.CutForm,
    V: fem.FunctionSpace,
) -> fem.Function:
    """Assemble a PETSc runtime system and solve it with PETSc."""
    try:
        from petsc4py import PETSc
    except ModuleNotFoundError as exc:
        raise RuntimeError("The PETSc solver option requires petsc4py.") from exc

    import dolfinx

    if not dolfinx.has_petsc:
        raise RuntimeError("The PETSc solver option requires DOLFINx with PETSc enabled.")

    import cutfemx.petsc

    # Assemble the same CutFEMx runtime forms directly into PETSc objects.
    A = cutfemx.petsc.assemble_matrix(a)
    b = cutfemx.petsc.assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Apply the inactive-domain constraints before final PETSc assembly.
    cutfemx.petsc.deactivate_outside(A, b, cutfemx.fem.active_domain(a))
    A.assemble()
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Use a direct PETSc solve by default, while still allowing prefixed options.
    uh = fem.Function(V)
    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOptionsPrefix("cutfemx_demo_")
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, uh.x.petsc_vec)
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(f"PETSc KSP solve failed with reason={reason}")
    uh.x.scatter_forward()

    ksp.destroy()
    A.destroy()
    b.destroy()

    return uh


# ## Visualisation
#
# The plotting routine displays the cut mesh, the level-set function restricted
# to the cut mesh, the runtime quadrature points, and the computed solution.


def plot_cut_mesh(
    cut_mesh: cutfemx.CutMesh,
    phi_cut: fem.Function,
    quadrature_rules,
    u_cut: fem.Function | None = None,
) -> None:
    """Show the cut mesh, the cut level set, quadrature points, and solution."""
    try:
        import pyvista
        from dolfinx import plot
    except ImportError:
        if MPI.COMM_WORLD.rank == 0:
            print("pyvista is required for interactive plotting")
        return

    if cut_mesh.mesh is None:
        return

    plotter = pyvista.Plotter(shape=(2, 2))

    cells, types, x = plot.vtk_mesh(cut_mesh.mesh)
    mesh_grid = pyvista.UnstructuredGrid(cells, types, x)
    mesh_grid.cell_data["parent_index"] = np.asarray(cut_mesh.parent_index)
    mesh_grid.cell_data["is_cut_cell"] = np.asarray(cut_mesh.is_cut_cell)

    plotter.subplot(0, 0)
    plotter.add_text("cut mesh", font_size=12)
    plotter.add_mesh(
        mesh_grid,
        scalars="is_cut_cell",
        show_edges=True,
        cmap="coolwarm",
    )

    cells, types, x = plot.vtk_mesh(phi_cut.function_space)
    phi_grid = pyvista.UnstructuredGrid(cells, types, x)
    phi_values = np.asarray(phi_cut.x.array)
    if np.iscomplexobj(phi_values):
        phi_values = phi_values.real
    phi_grid.point_data["phi"] = phi_values[: phi_grid.n_points]

    plotter.subplot(0, 1)
    plotter.add_text("cut level set", font_size=12)
    plotter.add_mesh(
        phi_grid,
        scalars="phi",
        show_edges=True,
        cmap="viridis",
    )

    plotter.subplot(1, 0)
    plotter.add_text("runtime quadrature points", font_size=12)
    plotter.add_mesh(
        mesh_grid,
        style="wireframe",
        color="lightgray",
        line_width=1,
    )
    if quadrature_rules.physical_points is not None:
        quadrature_points = _points_as_pyvista_coordinates(quadrature_rules.physical_points)
        point_cloud = pyvista.PolyData(quadrature_points)
        point_cloud.point_data["weight"] = np.asarray(quadrature_rules.weights)
        point_cloud.point_data["parent_index"] = _point_parent_index(quadrature_rules)
        plotter.add_mesh(
            point_cloud,
            scalars="weight",
            cmap="magma",
            render_points_as_spheres=True,
            point_size=9,
        )

    plotter.subplot(1, 1)
    plotter.add_text("cut Poisson solution", font_size=12)
    if u_cut is not None:
        cells, types, x = plot.vtk_mesh(u_cut.function_space)
        solution_grid = pyvista.UnstructuredGrid(cells, types, x)
        solution_values = np.asarray(u_cut.x.array)
        if np.iscomplexobj(solution_values):
            solution_values = solution_values.real
        solution_values = solution_values[: solution_grid.n_points]
        if solution_values.size != solution_grid.n_points:
            raise RuntimeError("Cut solution has fewer values than PyVista points.")
        solution_grid.point_data["u"] = solution_values
        solution_grid.set_active_scalars("u")
        warped_solution = solution_grid.warp_by_scalar(
            "u",
            factor=_scalar_warp_factor(x, solution_values),
            normal=(0.0, 0.0, 1.0),
        )
        plotter.add_mesh(
            warped_solution,
            scalars="u",
            show_edges=True,
            cmap="plasma",
        )

    plotter.link_views()
    plotter.show()


# ## Implementation
#
# The ``main`` function follows the same order as the mathematical
# construction: create the background geometry, cut it with the level set,
# define the variational problem, solve it, and optionally visualise the result.


def main(show_plot: bool = True, solver: str = "scipy") -> None:
    comm = MPI.COMM_WORLD

    # The physical domain is the disk with radius ``r`` centered at the
    # origin, embedded in the square background mesh. The parameter ``gamma``
    # penalises the trace of the solution on the embedded boundary.
    r = 0.5
    gamma = 10.0
    N = 21

    def circle_level_set(x: np.ndarray) -> np.ndarray:
        """Negative inside a circle centered at the origin."""
        return np.sqrt(x[0] ** 2 + x[1] ** 2) - r

    # We first create the background mesh and interpolate the level-set
    # function into a continuous Lagrange space. The zero contour of this
    # function describes the cut boundary.
    msh = mesh.create_rectangle(
        comm=comm,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(N, N),
        cell_type=mesh.CellType.triangle,
    )

    V_phi = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V_phi)
    phi.interpolate(circle_level_set)

    # The level-set function is analysed once with ``cutfemx.cut``. The
    # resulting cut data can then be reused to locate active background cells,
    # create a cut mesh for visualisation
    cut_data = cutfemx.cut(phi)

    inside_cells = cutfemx.locate_entities(cut_data, "phi < 0")
    cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi < 0", mode="full")

    # CutFEMx constructs runtime quadrature rules for the physical volume
    # ``phi < 0`` and for the embedded interface ``phi = 0``. These quadrature
    # rules are attached to UFL measures below.
    cut_quadrature = cutfemx.runtime_quadrature(cut_data, "phi < 0", order=4)
    interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi = 0", order=4)

    # The unknown is represented in a standard background finite element
    # space. The custom measures restrict integration to the cut volume and
    # embedded boundary quadrature rules.
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, np.float64(1.0))

    dxq = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[inside_cells, cut_quadrature],
    )
    dsq = ufl.Measure("dx", domain=msh, subdomain_data=interface_quadrature)

    # The bilinear form contains the Poisson stiffness on the physical volume
    # and a boundary penalty on the embedded interface. The linear form uses a
    # constant source term on the cut volume.
    a_cut = ufl.inner(ufl.grad(u), ufl.grad(v)) * dxq
    a_cut += gamma * u * v * dsq

    L_cut = f * v * dxq

    # The UFL forms are compiled as CutFEMx forms, preserving the runtime
    # quadrature domains introduced above.
    a_cut = cutfemx.fem.form(a_cut)
    L_cut = cutfemx.fem.form(L_cut)

    # The system can be solved either through MatrixCSR and SciPy or directly
    # through PETSc. The background solution is then transferred to the cut
    # mesh for visualisation.
    uh = solve_runtime_system(a_cut, L_cut, V, solver)
    u_cut = cutfemx.fem.cut_function(uh, cut_mesh)

    # Finally, PyVista can be used to inspect the cut mesh, the level-set
    # function, runtime quadrature points, and the computed solution.
    if show_plot and comm.rank == 0:
        plot_cut_mesh(
            cut_mesh,
            phi_cut,
            cut_quadrature.with_physical_points(),
            u_cut,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Run the demo without opening the interactive PyVista window.",
    )
    parser.add_argument(
        "--solver",
        choices=("scipy", "petsc"),
        default="scipy",
        help="Linear solver backend to use. 'scipy' uses MatrixCSR and is serial-only.",
    )
    args = parser.parse_args()
    main(show_plot=not args.no_plot, solver=args.solver)
