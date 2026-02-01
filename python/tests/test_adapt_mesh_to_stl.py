
import pytest
from mpi4py import MPI
import numpy as np
import os
from dolfinx.mesh import create_box, CellType, GhostMode
from cutfemx.mesh import adapt_mesh_to_stl

@pytest.fixture
def cube_stl(tmp_path):
    path = tmp_path / "cube.stl"
    # Create a simple cube [0.4, 0.4, 0.4] to [0.6, 0.6, 0.6] inside unit box
    # Just 12 triangles (2 per face)
    # Actually simpler: just one triangle? No, needs to be "feature".
    # A single triangle is fine for testing intersection.
    # Triangle: (0.5, 0.5, 0.5), (0.55, 0.5, 0.5), (0.5, 0.55, 0.5)
    
    with open(path, "w") as f:
        f.write("solid cube\n")
        f.write("facet normal 0 0 1\nouter loop\nvertex 0.5 0.5 0.5\nvertex 0.55 0.5 0.5\nvertex 0.5 0.55 0.5\nendloop\nendfacet\n")
        f.write("endsolid cube\n")
    return str(path)

def test_adapt_mesh_serial(cube_stl):
    comm = MPI.COMM_SELF
    mesh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])], 
                      [5, 5, 5], CellType.tetrahedron)
    
    n_cells_initial = mesh.topology.index_map(mesh.topology.dim).size_global
    
    mesh = adapt_mesh_to_stl(mesh, cube_stl, nlevels=1, k_ring=0)
    
    n_cells_final = mesh.topology.index_map(mesh.topology.dim).size_global
    
    assert n_cells_final > n_cells_initial
    
@pytest.mark.mpi
def test_adapt_mesh_mpi(cube_stl):
    comm = MPI.COMM_WORLD
    mesh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])], 
                      [5, 5, 5], CellType.tetrahedron, ghost_mode=GhostMode.shared_facet)
    
    n_cells_initial = mesh.topology.index_map(mesh.topology.dim).size_global
    
    # Run in MPI (if executed with mpirun)
    try:
        mesh = adapt_mesh_to_stl(mesh, cube_stl, nlevels=1, k_ring=1)
    except Exception as e:
        pytest.fail(f"MPI adaptation failed: {e}")
        
    n_cells_final = mesh.topology.index_map(mesh.topology.dim).size_global
    if comm.rank == 0:
        print(f"Cells: {n_cells_initial} -> {n_cells_final}")
        
    # We expect global increase (unless the STL is completely missed by partition, but with broadcast/search it should be found)
    # Total cells should increase
    n_cells_final_global = comm.allreduce(mesh.topology.index_map(mesh.topology.dim).size_local, op=MPI.SUM)
    n_cells_initial_global = comm.allreduce(n_cells_initial, op=MPI.SUM) # Wait, index_map size is local+ghosts? No, index_map size_global is global.
    
    # However, size_global is not always updated automatically in some dolfinx versions?
    # Safer to check local cells sum.
    
    assert n_cells_final > n_cells_initial # Local check? No, global check requires coordination.
    # In MPI test, just ensure it ran without error and returned a valid mesh.

def test_adapt_termination(cube_stl):
    comm = MPI.COMM_SELF
    mesh = create_box(comm, [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])], 
                      [2, 2, 2], CellType.tetrahedron)
                      
    # Refine until feature is resolved?
    # Or until max levels.
    # We test that it DOESN'T crash if we ask for many levels.
    mesh = adapt_mesh_to_stl(mesh, cube_stl, nlevels=5, k_ring=0)
    
    assert mesh.topology.index_map(mesh.topology.dim).size_global > 0
