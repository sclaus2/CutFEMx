#pragma once

// Robust geometric predicates using geogram MultiPrecision
// Provides exact orient3d and point-in-tetrahedron tests

#ifdef CUTFEMX_USE_MULTIPRECISION_PSM
#include "MultiPrecision_psm.h"
#endif

#include <array>
#include <cmath>

namespace cutfemx::mesh {

/// Sign type for geometric predicates
enum class PredSign : int {
    NEGATIVE = -1,
    ZERO = 0,
    POSITIVE = 1
};

#ifdef CUTFEMX_USE_MULTIPRECISION_PSM

/// Robust 3D orientation predicate using MultiPrecision
/// Returns POSITIVE if d is above plane (a,b,c) (counter-clockwise normal),
/// NEGATIVE if below, ZERO if coplanar
inline PredSign orient3d(const double* a, const double* b, 
                         const double* c, const double* d)
{
    // The GEO macros (expansion_create, expansion_diff, etc.) require 
    // the GEO namespace to be available since they use unqualified names
    using namespace GEO;
    
    // Use expansion arithmetic for exact computation
    // Formula: det([b-a, c-a, d-a])
    
    const expansion& ax_ = expansion_create(a[0]);
    const expansion& ay_ = expansion_create(a[1]);
    const expansion& az_ = expansion_create(a[2]);
    
    const expansion& bx = expansion_diff(b[0], a[0]);
    const expansion& by = expansion_diff(b[1], a[1]);
    const expansion& bz = expansion_diff(b[2], a[2]);
    
    const expansion& cx = expansion_diff(c[0], a[0]);
    const expansion& cy = expansion_diff(c[1], a[1]);
    const expansion& cz = expansion_diff(c[2], a[2]);
    
    const expansion& dx = expansion_diff(d[0], a[0]);
    const expansion& dy = expansion_diff(d[1], a[1]);
    const expansion& dz = expansion_diff(d[2], a[2]);
    
    // Cross product c × d
    const expansion& cxdy = expansion_product(cx, dy);
    const expansion& cydx = expansion_product(cy, dx);
    const expansion& cxdz = expansion_product(cx, dz);
    const expansion& czdx = expansion_product(cz, dx);
    const expansion& cydz = expansion_product(cy, dz);
    const expansion& czdy = expansion_product(cz, dy);
    
    // Actually we need det3x3 of [b-a, c-a, d-a]
    // = bx*(cy*dz - cz*dy) - by*(cx*dz - cz*dx) + bz*(cx*dy - cy*dx)
    
    const expansion& t1 = expansion_diff(cydz, czdy);
    const expansion& t2 = expansion_diff(cxdz, czdx);
    const expansion& t3 = expansion_diff(cxdy, cydx);
    
    const expansion& p1 = expansion_product(bx, t1);
    const expansion& p2 = expansion_product(by, t2);
    const expansion& p3 = expansion_product(bz, t3);
    
    const expansion& s1 = expansion_diff(p1, p2);
    const expansion& det = expansion_sum(s1, p3);
    
    geo_argused(ax_);
    geo_argused(ay_);
    geo_argused(az_);
    
    Sign s = det.sign();
    return static_cast<PredSign>(static_cast<int>(s));
}

#else

/// Fallback: floating-point orient3d (not robust, but works without MultiPrecision)
inline PredSign orient3d(const double* a, const double* b, 
                         const double* c, const double* d)
{
    double ax = a[0], ay = a[1], az = a[2];
    double bx = b[0] - ax, by = b[1] - ay, bz = b[2] - az;
    double cx = c[0] - ax, cy = c[1] - ay, cz = c[2] - az;
    double dx = d[0] - ax, dy = d[1] - ay, dz = d[2] - az;
    
    double det = bx * (cy * dz - cz * dy)
               - by * (cx * dz - cz * dx)
               + bz * (cx * dy - cy * dx);
    
    constexpr double eps = 1e-14;
    if (det > eps) return PredSign::POSITIVE;
    if (det < -eps) return PredSign::NEGATIVE;
    return PredSign::ZERO;
}

#endif

/// Test if point p is strictly inside tetrahedron (v0, v1, v2, v3)
/// Uses orient3d to test against all 4 faces
/// @return true if p is inside (all orientations same sign as the opposite vertex)
inline bool point_in_tet(const double* p,
                         const double* v0, const double* v1,
                         const double* v2, const double* v3)
{
    // For each face, check that p is on the same side as the opposite vertex
    // Face 0: (v1, v2, v3), opposite vertex: v0
    // Face 1: (v0, v3, v2), opposite vertex: v1
    // Face 2: (v0, v1, v3), opposite vertex: v2
    // Face 3: (v0, v2, v1), opposite vertex: v3
    
    PredSign s0 = orient3d(v1, v2, v3, v0);
    PredSign sp0 = orient3d(v1, v2, v3, p);
    
    // If on boundary, consider inside
    if (sp0 == PredSign::ZERO) {
        // On face - still counts as "intersecting" for our purpose
        return true;
    }
    if (s0 != sp0) return false;
    
    PredSign s1 = orient3d(v0, v3, v2, v1);
    PredSign sp1 = orient3d(v0, v3, v2, p);
    if (sp1 == PredSign::ZERO) return true;
    if (s1 != sp1) return false;
    
    PredSign s2 = orient3d(v0, v1, v3, v2);
    PredSign sp2 = orient3d(v0, v1, v3, p);
    if (sp2 == PredSign::ZERO) return true;
    if (s2 != sp2) return false;
    
    PredSign s3 = orient3d(v0, v2, v1, v3);
    PredSign sp3 = orient3d(v0, v2, v1, p);
    if (sp3 == PredSign::ZERO) return true;
    if (s3 != sp3) return false;
    
    return true;
}

/// Test if point p is inside or on boundary of a convex polyhedron
/// defined by its triangulated faces
/// @param p Point to test
/// @param faces Array of face vertex pointers [f0v0, f0v1, f0v2, f1v0, ...]
/// @param num_faces Number of triangular faces
/// @param centroid A point known to be inside the polyhedron (for orientation reference)
/// @return true if p is inside or on boundary
inline bool point_in_convex_polyhedron(
    const double* p,
    const double* const* face_verts,
    int num_faces,
    const double* centroid)
{
    for (int f = 0; f < num_faces; ++f)
    {
        const double* v0 = face_verts[3*f + 0];
        const double* v1 = face_verts[3*f + 1];
        const double* v2 = face_verts[3*f + 2];
        
        PredSign s_ref = orient3d(v0, v1, v2, centroid);
        PredSign s_p = orient3d(v0, v1, v2, p);
        
        // On boundary is OK
        if (s_p == PredSign::ZERO) continue;
        
        // Different side from centroid means outside
        if (s_ref != s_p) return false;
    }
    return true;
}

} // namespace cutfemx::mesh
