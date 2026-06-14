# Copyright (c) 2026 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT

import numpy as np
import os
from mpi4py import MPI
import dolfinx
from dolfinx.mesh import create_box, CellType, GhostMode
from dolfinx.fem import FunctionSpace, Function
from dolfinx.io import XDMFFile

from cutfemx.distance import adapt_mesh_to_stl, from_stl

def demo_stl_distance():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Path to STL file
    stl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_surface.stl"))
    
    if rank == 0:
        print(f"Computing distance to STL: {stl_path}")
        if not os.path.exists(stl_path):
            print("Error: demo_surface.stl not found in the demo directory!")
            return

    # 1. Get bbox of STL to define background mesh
    # Only need to do this on rank 0, then broadcast
    min_pt_stl = np.zeros(3, dtype=np.float64)
    max_pt_stl = np.zeros(3, dtype=np.float64)

    if rank == 0:
        from cutfemx.distance import compute_stl_bbox
        min_c, max_c = compute_stl_bbox(stl_path)
        min_pt_stl[:] = min_c
        max_pt_stl[:] = max_c
        print(f"STL Bbox: {min_pt_stl} - {max_pt_stl}")

    comm.Bcast(min_pt_stl, root=0)
    comm.Bcast(max_pt_stl, root=0)

    # Add some padding
    min_coord = np.min(min_pt_stl)
    max_coord = np.max(max_pt_stl)

    length_diag = max_coord - min_coord
    padding = 0.1 * length_diag
    min_pt_mesh = min_pt_stl - padding
    max_pt_mesh = max_pt_stl + padding

    length_mesh = np.zeros(3, dtype=np.float64)

    if rank == 0:
        length_mesh[:] = max_pt_mesh - min_pt_mesh

    comm.Bcast(length_mesh, root=0)

    min_length = np.min(length_mesh)
    n_min = 5

    N_mesh = np.zeros(3, dtype=int)
    N_mesh[:] = np.round(length_mesh / min_length * n_min).astype(int)
    print(N_mesh)
    
    mesh = create_box(
        comm,
        [min_pt_mesh, max_pt_mesh],
        N_mesh,
        CellType.tetrahedron,
        ghost_mode=GhostMode.shared_facet
    )

    if comm.rank == 0:
        print(f"Initial mesh: {mesh.topology.index_map(mesh.topology.dim).size_global} cells")
    
    # Adaptive mesh refinement around STL:
    # nlevels: Number of refinement levels
    # k_ring: Number of neighbor rings to mark
    # aabb_padding: Padding for STL search
    
    refined_mesh = adapt_mesh_to_stl(mesh, stl_path, 
                                     nlevels=3, 
                                     k_ring=1, 
                                     aabb_padding=0.0,
                                     ghost_mode=GhostMode.shared_facet)
                                     
    if comm.rank == 0:
        print(f"Refined mesh: {refined_mesh.topology.index_map(refined_mesh.topology.dim).size_global} cells")

    import time
    
    # 2. Compute signed distance
    if rank == 0: print("Starting distance.from_stl...", flush=True)
    t0 = time.time()
    
    from cutfemx.distance import SignMode
    dist = from_stl(refined_mesh, stl_path, sign_mode=SignMode.ComponentAnchor)
    
    t1 = time.time()
    if rank == 0: print(f"Computed distance in {t1-t0:.4f} s", flush=True)

    local_values = dist.x.array
    local_min = np.min(local_values) if local_values.size else np.inf
    local_max = np.max(local_values) if local_values.size else -np.inf
    local_neg = np.count_nonzero(local_values < 0.0)
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_neg = comm.allreduce(local_neg, op=MPI.SUM)
    if rank == 0:
        print(
            f"Signed distance range: [{global_min:.6e}, {global_max:.6e}] "
            f"with {global_neg} negative dofs",
            flush=True,
        )
    
    if rank == 0:
        print("Writing distance field to distance_from_stl.xdmf")

    # 5. Output
    with XDMFFile(comm, "distance_from_stl.xdmf", "w") as xdmf:
        xdmf.write_mesh(refined_mesh)
        xdmf.write_function(dist)
        xdmf.close()
        
if __name__ == "__main__":
    demo_stl_distance()
