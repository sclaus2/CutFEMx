#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cutfemx/mesh/mp_predicates.h>
#include <cutfemx/mesh/tri_intersection.h>
#include <cutfemx/mesh/cell_triangle_map.h>
#include <cutfemx/mesh/stl_surface.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>

#include <mpi.h>
#include <array>

using namespace cutfemx::mesh;

TEST_CASE("orient3d predicate", "[predicates]")
{
    SECTION("Point above plane")
    {
        // Triangle in z=0 plane
        double a[3] = {0, 0, 0};
        double b[3] = {1, 0, 0};
        double c[3] = {0, 1, 0};
        
        // Point above
        double d[3] = {0.25, 0.25, 1.0};
        
        PredSign s = orient3d(a, b, c, d);
        REQUIRE(s == PredSign::POSITIVE);
    }
    
    SECTION("Point below plane")
    {
        double a[3] = {0, 0, 0};
        double b[3] = {1, 0, 0};
        double c[3] = {0, 1, 0};
        
        double d[3] = {0.25, 0.25, -1.0};
        
        PredSign s = orient3d(a, b, c, d);
        REQUIRE(s == PredSign::NEGATIVE);
    }
    
    SECTION("Point on plane")
    {
        double a[3] = {0, 0, 0};
        double b[3] = {1, 0, 0};
        double c[3] = {0, 1, 0};
        
        double d[3] = {0.5, 0.5, 0.0};
        
        PredSign s = orient3d(a, b, c, d);
        REQUIRE(s == PredSign::ZERO);
    }
}

TEST_CASE("point_in_tet", "[predicates]")
{
    // Unit tetrahedron with vertices at origin and along axes
    double v0[3] = {0, 0, 0};
    double v1[3] = {1, 0, 0};
    double v2[3] = {0, 1, 0};
    double v3[3] = {0, 0, 1};
    
    SECTION("Point inside")
    {
        double p[3] = {0.1, 0.1, 0.1};
        REQUIRE(point_in_tet(p, v0, v1, v2, v3) == true);
    }
    
    SECTION("Point outside")
    {
        double p[3] = {1, 1, 1};
        REQUIRE(point_in_tet(p, v0, v1, v2, v3) == false);
    }
    
    SECTION("Point on face")
    {
        double p[3] = {0.25, 0.25, 0};  // On bottom face
        REQUIRE(point_in_tet(p, v0, v1, v2, v3) == true);
    }
    
    SECTION("Point at vertex")
    {
        REQUIRE(point_in_tet(v0, v0, v1, v2, v3) == true);
    }
}

TEST_CASE("segment_triangle_intersect", "[intersection]")
{
    // Triangle in z=0.5 plane
    double t0[3] = {0, 0, 0.5};
    double t1[3] = {1, 0, 0.5};
    double t2[3] = {0.5, 1, 0.5};
    
    SECTION("Segment crosses triangle")
    {
        double p0[3] = {0.3, 0.3, 0};
        double p1[3] = {0.3, 0.3, 1};
        
        REQUIRE(segment_triangle_intersect(p0, p1, t0, t1, t2) == true);
    }
    
    SECTION("Segment parallel, above triangle")
    {
        double p0[3] = {0, 0, 1};
        double p1[3] = {1, 0, 1};
        
        REQUIRE(segment_triangle_intersect(p0, p1, t0, t1, t2) == false);
    }
    
    SECTION("Segment misses triangle (same plane height but outside)")
    {
        double p0[3] = {2, 2, 0};
        double p1[3] = {2, 2, 1};
        
        REQUIRE(segment_triangle_intersect(p0, p1, t0, t1, t2) == false);
    }
}

TEST_CASE("triangle_tet_intersect", "[intersection]")
{
    // Unit tetrahedron
    double v0[3] = {0, 0, 0};
    double v1[3] = {1, 0, 0};
    double v2[3] = {0, 1, 0};
    double v3[3] = {0, 0, 1};
    
    const double* tet[4] = {v0, v1, v2, v3};
    
    SECTION("Triangle inside tet")
    {
        double t0[3] = {0.1, 0.1, 0.1};
        double t1[3] = {0.2, 0.1, 0.1};
        double t2[3] = {0.1, 0.2, 0.1};
        
        const double* tri[3] = {t0, t1, t2};
        
        REQUIRE(triangle_tet_intersect(tri, tet) == true);
    }
    
    SECTION("Triangle outside tet")
    {
        double t0[3] = {2, 0, 0};
        double t1[3] = {3, 0, 0};
        double t2[3] = {2.5, 1, 0};
        
        const double* tri[3] = {t0, t1, t2};
        
        REQUIRE(triangle_tet_intersect(tri, tet) == false);
    }
    
    SECTION("Triangle crosses tet face")
    {
        // Triangle centered at tet, crossing the z=0 face
        double t0[3] = {0.2, 0.2, -0.5};
        double t1[3] = {0.3, 0.2, 0.5};
        double t2[3] = {0.2, 0.3, 0.5};
        
        const double* tri[3] = {t0, t1, t2};
        
        REQUIRE(triangle_tet_intersect(tri, tet) == true);
    }
}

TEST_CASE("build_cell_triangle_map", "[map]")
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // Create a simple 2x2x2 unit cube mesh
    auto mesh = dolfinx::mesh::create_box<double>(
        comm, 
        {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, 
        {2, 2, 2},
        dolfinx::mesh::CellType::tetrahedron,
        dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::shared_facet));
    
    // Create a simple TriSoup with a triangle that crosses the mesh
    // Triangle in plane z = 0.5, covering entire xy domain
    TriSoup<double> soup;
    
    // 3 vertices for 1 triangle
    soup.X = {
        -0.5, -0.5, 0.5,   // v0
         1.5, -0.5, 0.5,   // v1
         0.5,  1.5, 0.5    // v2
    };
    
    // Triangle connectivity (1 triangle, referencing vertices 0, 1, 2)
    soup.tri = {0, 1, 2};
    
    // Normal (pointing up in z)
    soup.N = {0, 0, 1};
    
    // Global triangle id
    soup.tri_gid = {0};
    
    // Build map
    CellTriangleMapOptions opt;
    auto map = build_cell_triangle_map(mesh, soup, opt);
    
    // Check that map has correct structure
    auto topology = mesh.topology();
    auto cell_map = topology->index_map(3);
    const std::int32_t num_cells = cell_map->size_local();
    
    REQUIRE(map.offsets.size() == static_cast<std::size_t>(num_cells + 1));
    
    // The triangle at z=0.5 should intersect cells that span z=0.5
    // In a 2x2x2 mesh with tets, roughly half the cells should intersect
    std::int32_t intersect_count = 0;
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
        auto tris = map.triangles(c);
        if (!tris.empty())
            intersect_count++;
    }
    
    // At least some cells should intersect the triangle
    REQUIRE(intersect_count > 0);
    
    // Print info for debugging
    if (rank == 0)
    {
        INFO("Cells intersecting triangle: " << intersect_count << " / " << num_cells);
    }
}

