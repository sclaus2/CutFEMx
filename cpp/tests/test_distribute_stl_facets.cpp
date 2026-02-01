
#include <catch2/catch_test_macros.hpp>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/common/MPI.h>
#include <cutfemx/mesh/distribute_stl.h>
#include <cutfemx/mesh/stl_surface.h>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace cutfemx::mesh;

// Helper to create a simple ASCII STL file
void create_test_stl(const std::string& filename)
{
    std::ofstream f(filename);
    f << "solid test_cube\n";
    
    // Triangle 1: (0,0,0)-(1,0,0)-(0,1,0) (in XY plane, z=0)
    // Normal (0,0,1)
    f << "facet normal 0 0 1\n";
    f << "  outer loop\n";
    f << "    vertex 0 0 0\n";
    f << "    vertex 1 0 0\n";
    f << "    vertex 0 1 0\n";
    f << "  endloop\n";
    f << "endfacet\n";

    // Triangle 2: (1,1,1)-(2,1,1)-(1,2,1) (in XY plane, z=1)
    // Normal (0,0,-1) - just some normal
    f << "facet normal 0 0 -1\n";
    f << "  outer loop\n";
    f << "    vertex 1 1 1\n";
    f << "    vertex 2 1 1\n";
    f << "    vertex 1 2 1\n";
    f << "  endloop\n";
    f << "endfacet\n";

    f << "endsolid test_cube\n";
}

TEST_CASE("distribute_stl_facets", "[mpi]")
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = dolfinx::MPI::rank(comm);
    int size = dolfinx::MPI::size(comm);

    // Create a temporary STL file
    std::string stl_path = "test_data.stl";
    if (rank == 0)
        create_test_stl(stl_path);
    MPI_Barrier(comm);

    // 1. Create a background mesh
    // Box from (0,0,0) to (2,2,2)
    // Partition it so different ranks own different parts.
    // If n=2, usually split along X or mesh partitioner decides.
    auto mesh = dolfinx::mesh::create_box<double>(
        comm, 
        {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, 
        {5, 5, 5},
        dolfinx::mesh::CellType::tetrahedron,
        dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::shared_facet));

    // 2. Distribute STL options
    DistributeSTLOptions opt;
    opt.aabb_padding = 0.1; 
    opt.normalize_normals = true;

    // 3. Call distribute
    TriSoup<double> soup = distribute_stl_facets(comm, mesh, stl_path, opt);
    
    // 3. Verify
    // Invariants
    CHECK(soup.tri.size() == static_cast<size_t>(3 * soup.nT()));
    CHECK(soup.tri_gid.size() == static_cast<size_t>(soup.nT()));
    CHECK(soup.N.size() == static_cast<size_t>(3 * soup.nT()));
    CHECK(soup.X.size() == static_cast<size_t>(3 * soup.nV())); // 9 * nT
    CHECK(soup.nV() == 3 * soup.nT()); // duplicated vertices
    
    // Check Triangle 1 (0,0,0)-(1,0,0)-(0,1,0). AABB [0,1]x[0,1]x[0,0]
    // Check Triangle 2 (1,1,1)-(2,1,1)-(1,2,1). AABB [1,2]x[1,2]x[1,1]
    
    // Gather all received GIDs to rank 0
    std::vector<int> all_gids;
    std::vector<int> counts(size);
    int my_count = static_cast<int>(soup.tri_gid.size());
    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    
    std::vector<int> displs(size + 1, 0);
    if (rank == 0)
    {
        for (int i=0; i<size; ++i) displs[i+1] = displs[i] + counts[i];
        all_gids.resize(displs[size]);
    }
    
    MPI_Gatherv(soup.tri_gid.data(), my_count, MPI_INT,
                all_gids.data(), counts.data(), displs.data(), MPI_INT, 0, comm);
                
    if (rank == 0)
    {
        // Check coverage: we should see both GID 0 and GID 1 at least once (globally).
        bool found_0 = false;
        bool found_1 = false;
        for (int gid : all_gids)
        {
            if (gid == 0) found_0 = true;
            if (gid == 1) found_1 = true;
        }
        
        CHECK(found_0);
        CHECK(found_1);
        
        // Cleanup
        std::remove(stl_path.c_str());
    }
}
