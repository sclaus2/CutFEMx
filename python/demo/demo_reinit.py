
import numpy as np
import sys
from mpi4py import MPI
import dolfinx
from dolfinx.mesh import create_box, create_rectangle, CellType, GhostMode
from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile
import ufl

from cutfemx.level_set import reinitialize

def exact_sphere(x):
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - 0.3

def exact_circle(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.3

def distortion(phi):
    # Strong exponential distortion preserving sign
    # phi' = sign(phi) * (exp(3*|phi|) - 1)
    return np.sign(phi) * (np.exp(3.0 * np.abs(phi)) - 1.0)

def demo_reinit(dim=3):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"=== Level Set Reinitialization Demo ({dim}D) ===")

    # 1. Create mesh
    n = 32
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
    V = FunctionSpace(mesh, ("Lagrange", 1))
    phi = Function(V)
    phi_exact = Function(V)
    
    # 3. Interpolate exact and distorted level sets
    x = V.tabulate_dof_coordinates()
    
    if dim == 3:
        vals = exact_sphere(x.T)
    else:
        vals = exact_circle(x.T)
        
    phi_exact.x.array[:] = vals
    phi.x.array[:] = distortion(vals)
    
    phi.x.scatter_forward()
    phi_exact.x.scatter_forward()
    
    # 4. Compute error before
    err_local = np.abs(phi.x.array - phi_exact.x.array)
    l_inf_pre = comm.allreduce(np.max(err_local), op=MPI.MAX)
    
    if rank == 0:
        print(f"L_inf error (distorted): {l_inf_pre:.4f}")
        
    # Write initial state
    with XDMFFile(comm, f"reinit_{dim}d_distorted.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi)

    # 5. Reinitialize
    # This modifies phi in-place
    reinitialize(phi, max_iter=200, tol=1e-10)
    
    # 6. Compute error after
    err_local = np.abs(phi.x.array - phi_exact.x.array)
    l_inf_post = comm.allreduce(np.max(err_local), op=MPI.MAX)
    
    if rank == 0:
        print(f"L_inf error (reinit):    {l_inf_post:.4f}")
        print(f"Error reduction:         {l_inf_pre/l_inf_post:.1f}x")

    # Write final state
    with XDMFFile(comm, f"reinit_{dim}d_result.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi)
        
    # Write exact for comparison
    with XDMFFile(comm, f"reinit_{dim}d_exact.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(phi_exact)

if __name__ == "__main__":
    dim = 3
    if len(sys.argv) > 1:
        dim = int(sys.argv[1])
    demo_reinit(dim)
