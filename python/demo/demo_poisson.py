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
#   - integral_Gamma grad(u) . n v ds
#   - integral_Gamma grad(v) . n u ds
#   + gamma / h integral_Gamma u v ds
#   + gamma_g h integral_F jump(grad(u), n_F) jump(grad(v), n_F) ds
#   = integral_Omega f v dx
#   - integral_Gamma grad(v) . n g ds
#   + gamma / h integral_Gamma g v ds
#
# for all test functions ``v``. Here ``Gamma`` is the embedded boundary
# ``phi = 0`` and ``F`` is the ghost-penalty facet band around cut cells.

from __future__ import annotations

import argparse

from mpi4py import MPI

import cutfemx
import numpy as np

import ufl
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


def main(
    show_plot: bool = True,
    solver: str = "scipy",
    *,
    n: int = 21,
    order: int = 4,
    gamma: float = 40.0,
    gamma_g: float = 0.1,
) -> None:
    comm = MPI.COMM_WORLD

    # The physical domain is the disk with radius ``r`` centered at the
    # origin, embedded in the square background mesh. The manufactured
    # solution gives both the source term and the Nitsche boundary data.
    r = 0.5

    def circle_level_set(x: np.ndarray) -> np.ndarray:
        """Negative inside a circle centered at the origin."""
        return np.sqrt(x[0] ** 2 + x[1] ** 2) - r

    # We first create the background mesh and interpolate the level-set
    # function into a continuous Lagrange space. The zero contour of this
    # function describes the cut boundary.
    msh = mesh.create_rectangle(
        comm=comm,
        points=((-1.0, -1.0), (1.0, 1.0)),
        n=(n, n),
        cell_type=mesh.CellType.triangle,
    )

    V_phi = fem.functionspace(msh, ("Lagrange", 2))
    phi = fem.Function(V_phi)
    phi.interpolate(circle_level_set)

    # The level-set function is analysed once with ``cutfemx.cut``. The
    # resulting cut data can then be reused to locate active background cells,
    # create a cut mesh for visualisation
    cut_data = cutfemx.cut(phi)

    inside_cells = cutfemx.locate_entities(cut_data, "phi < 0")
    cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi < 0", mode="full")
    phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)

    # CutFEMx constructs runtime quadrature rules for the physical volume
    # ``phi < 0`` and for the embedded interface ``phi = 0``. These quadrature
    # rules are attached to UFL measures below.
    cut_quadrature = cutfemx.runtime_quadrature(cut_data, "phi < 0", order=order)
    interface_quadrature = cutfemx.runtime_quadrature(cut_data, "phi = 0", order=order)
    ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")

    # The unknown is represented in a standard background finite element
    # space. The measures restrict integration to the cut volume and embedded
    # boundary quadrature rules through subdomain_data.
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = 2.0 * np.pi**2 * u_exact
    n_gamma = cutfemx.normal(phi)
    n_facet = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = ufl.avg(h)

    dx_omega = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=0,
        subdomain_data=[inside_cells, cut_quadrature],
    )
    dx_gamma = ufl.Measure(
        "dx",
        domain=msh,
        subdomain_id=1,
        subdomain_data=interface_quadrature,
    )
    dS_ghost = ufl.Measure(
        "dS",
        domain=msh,
        subdomain_id=2,
        subdomain_data=ghost_facets,
    )

    # The bilinear form contains the Poisson stiffness, symmetric Nitsche
    # enforcement of the manufactured Dirichlet data on the embedded boundary,
    # and a standard ghost penalty on the cut-cell stabilization band.
    a_cut = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
    a_cut += (
        -ufl.dot(ufl.grad(u), n_gamma) * v - ufl.dot(ufl.grad(v), n_gamma) * u + gamma / h * u * v
    ) * dx_gamma
    if ghost_facets.size > 0:
        a_cut += (
            gamma_g
            * h_avg
            * ufl.inner(
                ufl.jump(ufl.grad(u), n_facet),
                ufl.jump(ufl.grad(v), n_facet),
            )
            * dS_ghost
        )

    L_cut = f * v * dx_omega
    L_cut += (
        -ufl.dot(ufl.grad(v), n_gamma) * u_exact + gamma / h * u_exact * v
    ) * dx_gamma

    # The UFL forms are compiled as CutFEMx forms, preserving the runtime
    # quadrature domains introduced above.
    a_cut = cutfemx.fem.form(a_cut)
    L_cut = cutfemx.fem.form(L_cut)

    # The system can be solved either through MatrixCSR and SciPy or directly
    # through PETSc. The background solution is then transferred to the cut
    # mesh for visualisation.
    uh = solve_runtime_system(a_cut, L_cut, V, solver)
    u_cut = cutfemx.fem.cut_function(uh, cut_mesh)

    error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_omega)
    error_sq = cutfemx.fem.assemble_scalar(error_form)
    error = np.sqrt(comm.allreduce(error_sq, op=MPI.SUM))
    if comm.rank == 0:
        print("CutFEMx Nitsche cut Poisson demo")
        print(f"  cells per side: {n}")
        print(f"  inside cells: {inside_cells.size}")
        print(f"  cut cells: {cut_quadrature.parent_map.size}")
        print(f"  ghost facets: {ghost_facets.size}")
        print(f"  L2 error: {error:.6e}")

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
    parser.add_argument("--n", type=int, default=21, help="Background cells per side.")
    parser.add_argument("--order", type=int, default=4, help="Runtime quadrature order.")
    parser.add_argument("--gamma", type=float, default=40.0, help="Nitsche penalty.")
    parser.add_argument("--gamma-g", type=float, default=0.1, help="Ghost penalty.")
    args = parser.parse_args()
    main(
        show_plot=not args.no_plot,
        solver=args.solver,
        n=args.n,
        order=args.order,
        gamma=args.gamma,
        gamma_g=args.gamma_g,
    )
