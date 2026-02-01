
#include <catch2/catch_test_macros.hpp>
#include <cutfemx/mesh/orient_surface.h>
#include <cutfemx/mesh/stl_surface.h>
#include <cmath>

using namespace cutfemx::mesh;

TEST_CASE("orient_surface", "[geometry]")
{
    // Construct a simple non-manifold or flipped surface
    // Two triangles sharing an edge, but one is flipped relative to geometric normal
    
    // T1: (0,0,0) - (1,0,0) - (0,1,0). Normal (0,0,1).
    // T2: (1,0,0) - (1,1,0) - (0,1,0). Normal (0,0,1).
    // Let's flip T2 indices to be (1,0,0)-(0,1,0)-(1,1,0) -> Normal (0,0,-1)
    
    TriSoup<double> S;
    S.tri_gid = {0, 1};
    
    // Vertices - let's make them separate but coincident to test welding
    S.X = {
        // T1
        0.0,0.0,0.0, 
        1.0,0.0,0.0, 
        0.0,1.0,0.0,
        // T2 (flipped)
        1.0000001,0.0,0.0, // slight noise for weld test
        0.0,1.0,0.0, 
        1.0,1.0,0.0
    };
    
    S.tri = {0,1,2, 3,4,5};
    S.N = {
        0.0,0.0,1.0,
        0.0,0.0,-1.0
    };
    
    OrientOptions opt;
    opt.weld_tol_rel = 1e-4;
    opt.normalize_triangle_normals = true;
    opt.use_preferred_direction = true;
    opt.preferred_dir[0] = 0;
    opt.preferred_dir[1] = 0;
    opt.preferred_dir[2] = 1;

    OrientDiagnostics<double> diag = orient_surface(S, opt);
    
    // Check results
    CHECK(diag.component_count == 1);
    CHECK(diag.boundary_edge_count > 0); // Open surface
    CHECK(diag.non_manifold_edge_count == 0);
    CHECK(diag.constraint_conflict_count == 0);
    
    // Check if both normals are now positive Z
    // S.N after orientation
    CHECK(S.N[2] > 0.0);
    CHECK(S.N[5] > 0.0);
    
    // Check if vertices were flipped for T2.
    // Original T2: (1,0,0)->(0,1,0)->(1,1,0). Edge 3-4 is (1,0,0)->(0,1,0).
    // T1 edge 1-2 is (1,0,0)->(0,1,0).
    // Same direction -> incompatible. MUST flip one.
    // T1 is (0,0,1), aligned with preferred. T1 should stay.
    // T2 should flip.
}
