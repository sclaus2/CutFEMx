import numpy as np
import sys
import os
import time
import logging
from mpi4py import MPI
import dolfinx
from dolfinx.mesh import create_box, create_rectangle, CellType, GhostMode
from dolfinx.fem import functionspace, Function
from dolfinx.io import XDMFFile
from dolfinx import fem, log
import ufl

from cutfemx.level_set import reinitialize

def parabolic_circle(x, R=0.3):
    # f = x^2 + y^2 - R^2
    # Zero level set is a circle of radius R
    return x[0]**2 + x[1]**2 - R**2

def exact_circle_sdf(x, R=0.3):
    # d = sqrt(x^2 + y^2) - R
    return np.sqrt(x[0]**2 + x[1]**2) - R

def parabolic_sphere(x, R=0.3):
    # f = x^2 + y^2 + z^2 - R^2
    return x[0]**2 + x[1]**2 + x[2]**2 - R**2

def exact_sphere_sdf(x, R=0.3):
    # d = sqrt(x^2 + y^2 + z^2) - R
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - R

def demo_reinit_parabolic(dim=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Configure logging to see internal spdlog timings if any
    if rank == 0:
        log.set_log_level(log.LogLevel.INFO)
        print(f"=== Parabolic Reinitialization Demo ({dim}D) ===")

    # 1. Create mesh
    n = 64
    R = 0.3
    if dim == 3:
        mesh = create_box(
            comm,
            [np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])],
            [n, n, n],
            CellType.tetrahedron,
            ghost_mode=GhostMode.shared_facet
        )
    else:
        mesh = create_rectangle(
            comm,
            [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
            [n, n],
            CellType.triangle,
            ghost_mode=GhostMode.shared_facet
        )

    # 2. Setup functions
    V = functionspace(mesh, ("Lagrange", 1))
    phi = Function(V)
    phi_exact = Function(V)
    
    # 3. Interpolate initial state and exact SDF
    x = V.tabulate_dof_coordinates()
    
    if dim == 3:
        initial_vals = parabolic_sphere(x.T, R)
        exact_vals = exact_sphere_sdf(x.T, R)
    else:
        initial_vals = parabolic_circle(x.T, R)
        exact_vals = exact_circle_sdf(x.T, R)
        
    phi.x.array[:] = initial_vals
    phi_exact.x.array[:] = exact_vals
    
    phi.x.scatter_forward()
    phi_exact.x.scatter_forward()
    
    # 4. Compute error before
    # Only compare near the interface if desired, but L_inf global is fine
    err_local = np.abs(phi.x.array - phi_exact.x.array)
    l_inf_pre = comm.allreduce(np.max(err_local) if len(err_local) > 0 else 0, op=MPI.MAX)
    
    if rank == 0:
        print(f"L_inf error (parabolic): {l_inf_pre:.4e}")
        
    # Write initial state
    with XDMFFile(comm, "reinit_parabolic_initial.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi)

    # 5. Reinitialize
    # This modifies phi in-place using FMM
    t0 = time.time()
    reinitialize(phi, max_iter=500, tol=1e-10)
    t1 = time.time()
    
    if rank == 0:
        print(f"Reinitialization took: {t1 - t0:.4f}s")
    
    # 6. Compute error after
    err_local = np.abs(phi.x.array - phi_exact.x.array)
    l_inf_post = comm.allreduce(np.max(err_local) if len(err_local) > 0 else 0, op=MPI.MAX)
    
    if rank == 0:
        print(f"L_inf error (reinit):    {l_inf_post:.4e}")
        if l_inf_post > 0:
            print(f"Error reduction:         {l_inf_pre/l_inf_post:.1f}x")

    # Write final state
    with XDMFFile(comm, "reinit_parabolic_result.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi)
        
    # Write exact for comparison
    with XDMFFile(comm, "reinit_parabolic_exact.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi_exact)

if __name__ == "__main__":
    dim = 2
    if len(sys.argv) > 1:
        dim = int(sys.argv[1])
    demo_reinit_parabolic(dim)
