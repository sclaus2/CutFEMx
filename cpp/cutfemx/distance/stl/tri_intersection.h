// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

// Robust triangle intersection tests using MultiPrecision predicates
// Provides segment-triangle and triangle-triangle intersection tests

#include "mp_predicates.h"
#include <array>
#include <algorithm>
#include <cmath>

namespace cutfemx::distance {

namespace detail {

inline double orient2d(const std::array<double, 2>& a,
                       const std::array<double, 2>& b,
                       const std::array<double, 2>& c)
{
    return (b[0] - a[0]) * (c[1] - a[1])
           - (b[1] - a[1]) * (c[0] - a[0]);
}

inline bool point_on_segment2d(const std::array<double, 2>& p,
                               const std::array<double, 2>& a,
                               const std::array<double, 2>& b,
                               double eps)
{
    if (std::abs(orient2d(a, b, p)) > eps)
        return false;
    return p[0] >= std::min(a[0], b[0]) - eps
           && p[0] <= std::max(a[0], b[0]) + eps
           && p[1] >= std::min(a[1], b[1]) - eps
           && p[1] <= std::max(a[1], b[1]) + eps;
}

inline bool segments_intersect2d(const std::array<double, 2>& a,
                                 const std::array<double, 2>& b,
                                 const std::array<double, 2>& c,
                                 const std::array<double, 2>& d,
                                 double eps)
{
    const double o1 = orient2d(a, b, c);
    const double o2 = orient2d(a, b, d);
    const double o3 = orient2d(c, d, a);
    const double o4 = orient2d(c, d, b);

    if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps))
        && ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps)))
        return true;

    return point_on_segment2d(c, a, b, eps)
           || point_on_segment2d(d, a, b, eps)
           || point_on_segment2d(a, c, d, eps)
           || point_on_segment2d(b, c, d, eps);
}

inline bool point_in_triangle2d(const std::array<double, 2>& p,
                                const std::array<double, 2>& a,
                                const std::array<double, 2>& b,
                                const std::array<double, 2>& c,
                                double eps)
{
    const double o0 = orient2d(a, b, p);
    const double o1 = orient2d(b, c, p);
    const double o2 = orient2d(c, a, p);

    const bool has_pos = o0 > eps || o1 > eps || o2 > eps;
    const bool has_neg = o0 < -eps || o1 < -eps || o2 < -eps;
    return !(has_pos && has_neg);
}

template <typename Real>
bool coplanar_segment_triangle_intersect(
    const Real* p0, const Real* p1,
    const Real* t0, const Real* t1, const Real* t2)
{
    const double u[3] = {
        static_cast<double>(t1[0] - t0[0]),
        static_cast<double>(t1[1] - t0[1]),
        static_cast<double>(t1[2] - t0[2])};
    const double v[3] = {
        static_cast<double>(t2[0] - t0[0]),
        static_cast<double>(t2[1] - t0[1]),
        static_cast<double>(t2[2] - t0[2])};
    const double n[3] = {
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]};

    int drop = 0;
    if (std::abs(n[1]) > std::abs(n[drop]))
        drop = 1;
    if (std::abs(n[2]) > std::abs(n[drop]))
        drop = 2;
    if (std::abs(n[drop]) <= 1.0e-14)
        return false;

    auto project = [drop](const Real* p)
    {
        std::array<double, 2> q{};
        int j = 0;
        for (int d = 0; d < 3; ++d)
        {
            if (d != drop)
                q[j++] = static_cast<double>(p[d]);
        }
        return q;
    };

    const auto a = project(t0);
    const auto b = project(t1);
    const auto c = project(t2);
    const auto s0 = project(p0);
    const auto s1 = project(p1);
    constexpr double eps = 1.0e-12;

    return point_in_triangle2d(s0, a, b, c, eps)
           || point_in_triangle2d(s1, a, b, c, eps)
           || segments_intersect2d(s0, s1, a, b, eps)
           || segments_intersect2d(s0, s1, b, c, eps)
           || segments_intersect2d(s0, s1, c, a, eps);
}

} // namespace detail

/// Test if segment (p0, p1) intersects triangle (t0, t1, t2)
/// Uses orient3d predicates for robustness
/// @return true if segment crosses or touches the triangle
template <typename Real>
bool segment_triangle_intersect(
    const Real* p0, const Real* p1,
    const Real* t0, const Real* t1, const Real* t2)
{
    // Convert to double for predicates
    double p0d[3] = {static_cast<double>(p0[0]), static_cast<double>(p0[1]), static_cast<double>(p0[2])};
    double p1d[3] = {static_cast<double>(p1[0]), static_cast<double>(p1[1]), static_cast<double>(p1[2])};
    double t0d[3] = {static_cast<double>(t0[0]), static_cast<double>(t0[1]), static_cast<double>(t0[2])};
    double t1d[3] = {static_cast<double>(t1[0]), static_cast<double>(t1[1]), static_cast<double>(t1[2])};
    double t2d[3] = {static_cast<double>(t2[0]), static_cast<double>(t2[1]), static_cast<double>(t2[2])};
    
    // Step 1: Check if segment endpoints are on opposite sides of triangle plane
    PredSign s0 = orient3d(t0d, t1d, t2d, p0d);
    PredSign s1 = orient3d(t0d, t1d, t2d, p1d);
    
    // If both on same side (and neither on plane), no intersection
    if (s0 == s1 && s0 != PredSign::ZERO)
        return false;
    
    // If both on plane, check 2D intersection (coplanar case)
    if (s0 == PredSign::ZERO && s1 == PredSign::ZERO)
    {
        return detail::coplanar_segment_triangle_intersect(
            p0, p1, t0, t1, t2);
    }
    
    // Step 2: Segment crosses plane. Find intersection point orientation.
    // The intersection point must be inside the triangle.
    // Test: orient3d(p0, p1, ti, tj) for each edge (ti, tj) of triangle
    // All orientations must be consistent.
    
    PredSign e0 = orient3d(p0d, p1d, t0d, t1d);
    PredSign e1 = orient3d(p0d, p1d, t1d, t2d);
    PredSign e2 = orient3d(p0d, p1d, t2d, t0d);
    
    // Intersection point is inside triangle if all edge orientations are same sign
    // (or zero, meaning on edge, which counts as inside)
    
    // Count positives and negatives
    int pos = (e0 == PredSign::POSITIVE) + (e1 == PredSign::POSITIVE) + (e2 == PredSign::POSITIVE);
    int neg = (e0 == PredSign::NEGATIVE) + (e1 == PredSign::NEGATIVE) + (e2 == PredSign::NEGATIVE);
    
    // If all same sign (or zero), intersection is inside triangle
    return (pos == 0 || neg == 0);
}

/// Test if triangle (a0, a1, a2) intersects triangle (b0, b1, b2)
/// Returns true if triangles share any region (including touching)
/// Algorithm:
/// 1. Test if any vertex of A is inside triangle B's 3D region
/// 2. Test all edges of A against face B
/// 3. Test all edges of B against face A
template <typename Real>
bool triangle_triangle_intersect(
    const Real* a0, const Real* a1, const Real* a2,
    const Real* b0, const Real* b1, const Real* b2)
{
    // Test edges of A against triangle B
    if (segment_triangle_intersect(a0, a1, b0, b1, b2)) return true;
    if (segment_triangle_intersect(a1, a2, b0, b1, b2)) return true;
    if (segment_triangle_intersect(a2, a0, b0, b1, b2)) return true;
    
    // Test edges of B against triangle A
    if (segment_triangle_intersect(b0, b1, a0, a1, a2)) return true;
    if (segment_triangle_intersect(b1, b2, a0, a1, a2)) return true;
    if (segment_triangle_intersect(b2, b0, a0, a1, a2)) return true;
    
    return false;
}

/// Test if surface triangle intersects a tetrahedral cell
/// @param tri Array of 3 vertex pointers for the surface triangle
/// @param tet Array of 4 vertex pointers for the tetrahedron
/// @return true if triangle intersects or is inside the tetrahedron
template <typename Real>
bool triangle_tet_intersect(
    const Real* const* tri,
    const Real* const* tet)
{
    // Convert to double for predicates
    double tv[3][3];
    for (int i = 0; i < 3; ++i)
    {
        tv[i][0] = static_cast<double>(tri[i][0]);
        tv[i][1] = static_cast<double>(tri[i][1]);
        tv[i][2] = static_cast<double>(tri[i][2]);
    }
    
    double tetv[4][3];
    for (int i = 0; i < 4; ++i)
    {
        tetv[i][0] = static_cast<double>(tet[i][0]);
        tetv[i][1] = static_cast<double>(tet[i][1]);
        tetv[i][2] = static_cast<double>(tet[i][2]);
    }
    
    // Step 1: Check if any triangle vertex is inside the tetrahedron
    for (int i = 0; i < 3; ++i)
    {
        if (point_in_tet(tv[i], tetv[0], tetv[1], tetv[2], tetv[3]))
            return true;
    }
    
    // Step 2: Test if triangle edges intersect tet faces
    // Tet faces (each as 3 vertices, oriented outward):
    // Face 0: (1, 2, 3)
    // Face 1: (0, 3, 2)
    // Face 2: (0, 1, 3)
    // Face 3: (0, 2, 1)
    
    const Real* tet_faces[4][3] = {
        {tet[1], tet[2], tet[3]},
        {tet[0], tet[3], tet[2]},
        {tet[0], tet[1], tet[3]},
        {tet[0], tet[2], tet[1]}
    };
    
    // Test each triangle edge against each tet face
    const Real* tri_edges[3][2] = {
        {tri[0], tri[1]},
        {tri[1], tri[2]},
        {tri[2], tri[0]}
    };
    
    for (int e = 0; e < 3; ++e)
    {
        for (int f = 0; f < 4; ++f)
        {
            if (segment_triangle_intersect(
                    tri_edges[e][0], tri_edges[e][1],
                    tet_faces[f][0], tet_faces[f][1], tet_faces[f][2]))
                return true;
        }
    }
    
    // Step 3: Test if tet edges intersect the triangle
    // Tet edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    const Real* tet_edges[6][2] = {
        {tet[0], tet[1]},
        {tet[0], tet[2]},
        {tet[0], tet[3]},
        {tet[1], tet[2]},
        {tet[1], tet[3]},
        {tet[2], tet[3]}
    };
    
    for (int e = 0; e < 6; ++e)
    {
        if (segment_triangle_intersect(
                tet_edges[e][0], tet_edges[e][1],
                tri[0], tri[1], tri[2]))
            return true;
    }
    
    return false;
}

/// Test if surface triangle intersects a hexahedral cell
/// The hex is defined by 8 vertices in standard VTK ordering
/// @param tri Array of 3 vertex pointers for the surface triangle
/// @param hex Array of 8 vertex pointers for the hexahedron
/// @return true if triangle intersects or is inside the hexahedron
template <typename Real>
bool triangle_hex_intersect(
    const Real* const* tri,
    const Real* const* hex)
{
    // Convert to double for predicates
    double tv[3][3];
    for (int i = 0; i < 3; ++i)
    {
        tv[i][0] = static_cast<double>(tri[i][0]);
        tv[i][1] = static_cast<double>(tri[i][1]);
        tv[i][2] = static_cast<double>(tri[i][2]);
    }
    
    // Compute centroid of hex for inside test
    double centroid[3] = {0, 0, 0};
    for (int i = 0; i < 8; ++i)
    {
        centroid[0] += static_cast<double>(hex[i][0]);
        centroid[1] += static_cast<double>(hex[i][1]);
        centroid[2] += static_cast<double>(hex[i][2]);
    }
    centroid[0] /= 8.0;
    centroid[1] /= 8.0;
    centroid[2] /= 8.0;
    
    // Hex faces (each quad split into 2 triangles)
    // VTK hex vertex ordering: 0-3 bottom, 4-7 top (counter-clockwise when viewed from outside)
    // Face 0 (bottom): 0,3,2,1 -> triangles (0,3,2), (0,2,1)
    // Face 1 (top): 4,5,6,7 -> triangles (4,5,6), (4,6,7)
    // Face 2 (front): 0,1,5,4 -> triangles (0,1,5), (0,5,4)
    // Face 3 (right): 1,2,6,5 -> triangles (1,2,6), (1,6,5)
    // Face 4 (back): 2,3,7,6 -> triangles (2,3,7), (2,7,6)
    // Face 5 (left): 3,0,4,7 -> triangles (3,0,4), (3,4,7)
    
    static const int hex_face_tris[12][3] = {
        {0, 3, 2}, {0, 2, 1},  // bottom
        {4, 5, 6}, {4, 6, 7},  // top
        {0, 1, 5}, {0, 5, 4},  // front
        {1, 2, 6}, {1, 6, 5},  // right
        {2, 3, 7}, {2, 7, 6},  // back
        {3, 0, 4}, {3, 4, 7}   // left
    };
    
    // Build face vertex pointers for point_in_convex_polyhedron
    const double* face_verts[36]; // 12 triangles * 3 vertices
    double hex_verts[8][3];
    for (int i = 0; i < 8; ++i)
    {
        hex_verts[i][0] = static_cast<double>(hex[i][0]);
        hex_verts[i][1] = static_cast<double>(hex[i][1]);
        hex_verts[i][2] = static_cast<double>(hex[i][2]);
    }
    
    for (int f = 0; f < 12; ++f)
    {
        face_verts[3*f + 0] = hex_verts[hex_face_tris[f][0]];
        face_verts[3*f + 1] = hex_verts[hex_face_tris[f][1]];
        face_verts[3*f + 2] = hex_verts[hex_face_tris[f][2]];
    }
    
    // Step 1: Check if any triangle vertex is inside the hex
    for (int i = 0; i < 3; ++i)
    {
        if (point_in_convex_polyhedron(tv[i], face_verts, 12, centroid))
            return true;
    }
    
    // Step 2: Test triangle edges against hex faces
    const Real* tri_edges[3][2] = {
        {tri[0], tri[1]},
        {tri[1], tri[2]},
        {tri[2], tri[0]}
    };
    
    for (int e = 0; e < 3; ++e)
    {
        for (int f = 0; f < 12; ++f)
        {
            const Real* f0 = hex[hex_face_tris[f][0]];
            const Real* f1 = hex[hex_face_tris[f][1]];
            const Real* f2 = hex[hex_face_tris[f][2]];
            
            if (segment_triangle_intersect(
                    tri_edges[e][0], tri_edges[e][1],
                    f0, f1, f2))
                return true;
        }
    }
    
    // Step 3: Test hex edges against surface triangle
    // Hex has 12 edges
    static const int hex_edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // bottom
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // top
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // verticals
    };
    
    for (int e = 0; e < 12; ++e)
    {
        if (segment_triangle_intersect(
                hex[hex_edges[e][0]], hex[hex_edges[e][1]],
                tri[0], tri[1], tri[2]))
            return true;
    }
    
    return false;
}

} // namespace cutfemx::distance
