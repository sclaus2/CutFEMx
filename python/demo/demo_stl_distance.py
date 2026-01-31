
import numpy as np
import os
from mpi4py import MPI
import dolfinx
from dolfinx.mesh import create_box, CellType, GhostMode
from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile

from cutfemx.level_set import compute_distance_from_stl

def demo_stl_distance():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Path to STL file
    stl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sphere.stl"))
    
    if rank == 0:
        print(f"Computing distance to STL: {stl_path}")
        if not os.path.exists(stl_path):
            print("Error: sphere.stl not found in root directory!")
            return

    # 1. Get bbox of STL to define background mesh
    # Only need to do this on rank 0, then broadcast
    min_pt_stl = np.zeros(3, dtype=np.float64)
    max_pt_stl = np.zeros(3, dtype=np.float64)

    if rank == 0:
        from cutfemx.level_set import compute_stl_bbox
        min_c, max_c = compute_stl_bbox(stl_path)
        min_pt_stl[:] = min_c
        max_pt_stl[:] = max_c
        print(f"STL Bbox: {min_pt_stl} - {max_pt_stl}")

    comm.Bcast(min_pt_stl, root=0)
    comm.Bcast(max_pt_stl, root=0)

    # Add some padding
    padding = 0.2
    min_pt_mesh = min_pt_stl - padding
    max_pt_mesh = max_pt_stl + padding
    
    n = 20
    mesh = create_box(
        comm,
        [min_pt_mesh, max_pt_mesh],
        [n, n, n],
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet
    )

    import time
    
    # 2. Compute signed distance
    if rank == 0: print("Starting compute_distance_from_stl...", flush=True)
    t0 = time.time()
    
    from cutfemx.level_set import SignMode
    dist = compute_distance_from_stl(mesh, stl_path, sign_mode=SignMode.ComponentAnchor)
    
    t1 = time.time()
    if rank == 0: print(f"Computed distance in {t1-t0:.4f} s", flush=True)
    
    if rank == 0:
        print("Writing distance field to distance_from_stl.xdmf")

    # 5. Output
    with XDMFFile(comm, "distance_from_stl.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(dist)

if __name__ == "__main__":
    demo_stl_distance()
