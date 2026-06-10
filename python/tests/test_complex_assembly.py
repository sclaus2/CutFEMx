import numpy as np
import pytest
from mpi4py import MPI
import ufl

from dolfinx import fem, mesh

import cutfemx
import cutfemx.fem as cf_fem
from quadrature_utils import runtime_quadrature_mesh


@pytest.mark.parametrize("geometry_dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "scalar_dtype",
    [np.float32, np.float64, np.complex64, np.complex128],
)
def test_runtime_assembly_supports_scalar_geometry_combinations(
    scalar_dtype, geometry_dtype
):
    scalar_dtype = np.dtype(scalar_dtype)
    geometry_dtype = np.dtype(geometry_dtype)
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=geometry_dtype.type)
    dx_runtime = ufl.Measure(
        "dx", subdomain_data=runtime_quadrature_mesh(msh, order=2), domain=msh
    )
    value = 2.0 + (0.5j if np.issubdtype(scalar_dtype, np.complexfloating) else 0.0)
    kappa = fem.Constant(msh, scalar_dtype.type(value))

    form = cf_fem.form(kappa * dx_runtime, dtype=scalar_dtype.type)
    assembled = cf_fem.assemble_scalar(form)
    assembled = msh.comm.allreduce(assembled, op=MPI.SUM)

    assert form.dtype == scalar_dtype
    assert form.geometry_dtype == geometry_dtype
    tol = 1.0e-5 if scalar_dtype.itemsize <= 8 else 1.0e-12
    np.testing.assert_allclose(
        assembled,
        scalar_dtype.type(value),
        rtol=tol,
        atol=tol,
    )


def test_complex_runtime_assembly_matches_standard_dolfinx():
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx_runtime = ufl.Measure(
        "dx", subdomain_data=runtime_quadrature_mesh(msh, order=2), domain=msh
    )
    kappa = fem.Constant(msh, np.complex128(2.0 + 3.0j))

    scalar_cut = cf_fem.assemble_scalar(
        cf_fem.form(kappa * dx_runtime, dtype=np.complex128)
    )
    scalar_ref = fem.assemble_scalar(
        fem.form(kappa * ufl.dx(domain=msh), dtype=np.complex128)
    )
    scalar_error = abs(msh.comm.allreduce(scalar_cut - scalar_ref, op=MPI.SUM))

    vector_cut = cf_fem.assemble_vector(
        cf_fem.form(ufl.inner(kappa, v) * dx_runtime, dtype=np.complex128)
    )
    vector_ref = fem.assemble_vector(
        fem.form(ufl.inner(kappa, v) * ufl.dx(domain=msh), dtype=np.complex128)
    )
    vector_error = np.linalg.norm(vector_cut.array - vector_ref.array)
    vector_error = msh.comm.allreduce(vector_error, op=MPI.MAX)

    matrix_cut = cf_fem.assemble_matrix(
        cf_fem.form(
            kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_runtime,
            dtype=np.complex128,
        )
    )
    matrix_ref = fem.assemble_matrix(
        fem.form(
            kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=msh),
            dtype=np.complex128,
        )
    )
    matrix_error = np.linalg.norm(matrix_cut.to_dense() - matrix_ref.to_dense())
    matrix_error = msh.comm.allreduce(matrix_error, op=MPI.MAX)

    assert scalar_error < 1.0e-12
    assert vector_error < 1.0e-12
    assert matrix_error < 1.0e-12


def test_complex_runtime_matrix_accepts_extension_terms():
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (1.0, 1.0)),
        n=(4, 4),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi = fem.Function(V)
    phi.interpolate(lambda x: x[0] - 0.51)
    cut_data = cutfemx.cut(phi)
    selector = "phi < 0"
    rules = cutfemx.runtime_quadrature(cut_data, selector, order=2)
    dx_runtime = ufl.Measure("dx", domain=msh, subdomain_data=rules)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = fem.Constant(msh, np.complex128(2.0 + 3.0j))
    a = cf_fem.form(kappa * ufl.inner(u, v) * dx_runtime, dtype=np.complex128)

    aggregation = cutfemx.extensions.create_cell_aggregation(
        cut_data,
        selector,
        1.0,
        root_policy="interior_only",
    )
    beta = np.complex128(2.5 + 0.75j)
    term = cutfemx.extensions.ExtensionPenaltyTerm(
        V, beta=beta, quadrature_degree=2, cut_data=cut_data, aggregation=aggregation
    )

    A_ref = cf_fem.assemble_matrix(a)
    A_ext = cutfemx.extensions.extension_penalty_matrix(
        V, cut_data, aggregation, beta=beta, quadrature_degree=2
    )
    A = cf_fem.assemble_matrix(a, extension_terms=[term])
    A_ref.scatter_reverse()
    A_ext.scatter_reverse()
    A.scatter_reverse()

    np.testing.assert_allclose(
        A.to_dense(), A_ref.to_dense() + A_ext.to_dense(), atol=1.0e-12
    )
