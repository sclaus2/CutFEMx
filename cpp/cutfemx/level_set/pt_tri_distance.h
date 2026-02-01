#pragma once

// Point-Triangle Distance: exact unsigned distance from point to triangle
// Uses Voronoi region classification for robustness

#include <array>
#include <cmath>
#include <algorithm>

namespace cutfemx::level_set {

namespace detail {

/// Dot product of two 3D vectors
template <typename Real>
inline Real dot3(const Real* a, const Real* b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Squared norm of 3D vector
template <typename Real>
inline Real norm2_sq(const Real* v)
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

/// Subtract vectors: out = a - b
template <typename Real>
inline void sub3(const Real* a, const Real* b, Real* out)
{
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

/// Cross product: out = a × b
template <typename Real>
inline void cross3(const Real* a, const Real* b, Real* out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/// Clamp value to [0, 1]
template <typename Real>
inline Real clamp01(Real x)
{
    return std::max(Real(0), std::min(Real(1), x));
}

} // namespace detail

/// Compute squared distance from point P to line segment AB
/// Returns closest point parameter t in [0,1] where closest = A + t*(B-A)
template <typename Real>
Real point_segment_distance_sq(
    const Real* P, const Real* A, const Real* B,
    Real* closest = nullptr)
{
    Real AB[3], AP[3];
    detail::sub3(B, A, AB);
    detail::sub3(P, A, AP);
    
    Real ab_sq = detail::norm2_sq(AB);
    if (ab_sq < Real(1e-30))
    {
        // Degenerate segment (A == B)
        if (closest)
        {
            closest[0] = A[0];
            closest[1] = A[1];
            closest[2] = A[2];
        }
        return detail::norm2_sq(AP);
    }
    
    Real t = detail::clamp01(detail::dot3(AP, AB) / ab_sq);
    
    Real closest_pt[3] = {
        A[0] + t * AB[0],
        A[1] + t * AB[1],
        A[2] + t * AB[2]
    };
    
    if (closest)
    {
        closest[0] = closest_pt[0];
        closest[1] = closest_pt[1];
        closest[2] = closest_pt[2];
    }
    
    Real diff[3];
    detail::sub3(P, closest_pt, diff);
    return detail::norm2_sq(diff);
}

/// Compute squared unsigned distance from point P to triangle (A, B, C)
/// Uses Voronoi region classification
/// @param P Query point (3D)
/// @param A First triangle vertex
/// @param B Second triangle vertex  
/// @param C Third triangle vertex
/// @param closest Optional output: closest point on triangle
/// @return Squared distance
template <typename Real>
Real point_triangle_distance_sq(
    const Real* P, const Real* A, const Real* B, const Real* C,
    Real* closest = nullptr)
{
    // Edge vectors
    Real AB[3], AC[3], AP[3];
    detail::sub3(B, A, AB);
    detail::sub3(C, A, AC);
    detail::sub3(P, A, AP);
    
    // Compute dot products
    Real d1 = detail::dot3(AB, AP);
    Real d2 = detail::dot3(AC, AP);
    
    // Check if P is in vertex region outside A
    if (d1 <= 0 && d2 <= 0)
    {
        if (closest)
        {
            closest[0] = A[0];
            closest[1] = A[1];
            closest[2] = A[2];
        }
        return detail::norm2_sq(AP);
    }
    
    // Check if P is in vertex region outside B
    Real BP[3];
    detail::sub3(P, B, BP);
    Real d3 = detail::dot3(AB, BP);
    Real d4 = detail::dot3(AC, BP);
    
    if (d3 >= 0 && d4 <= d3)
    {
        if (closest)
        {
            closest[0] = B[0];
            closest[1] = B[1];
            closest[2] = B[2];
        }
        return detail::norm2_sq(BP);
    }
    
    // Check if P is in edge region of AB
    Real vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
    {
        Real v = d1 / (d1 - d3);
        Real pt[3] = {A[0] + v*AB[0], A[1] + v*AB[1], A[2] + v*AB[2]};
        if (closest)
        {
            closest[0] = pt[0];
            closest[1] = pt[1];
            closest[2] = pt[2];
        }
        Real diff[3];
        detail::sub3(P, pt, diff);
        return detail::norm2_sq(diff);
    }
    
    // Check if P is in vertex region outside C
    Real CP[3];
    detail::sub3(P, C, CP);
    Real d5 = detail::dot3(AB, CP);
    Real d6 = detail::dot3(AC, CP);
    
    if (d6 >= 0 && d5 <= d6)
    {
        if (closest)
        {
            closest[0] = C[0];
            closest[1] = C[1];
            closest[2] = C[2];
        }
        return detail::norm2_sq(CP);
    }
    
    // Check if P is in edge region of AC
    Real vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
    {
        Real w = d2 / (d2 - d6);
        Real pt[3] = {A[0] + w*AC[0], A[1] + w*AC[1], A[2] + w*AC[2]};
        if (closest)
        {
            closest[0] = pt[0];
            closest[1] = pt[1];
            closest[2] = pt[2];
        }
        Real diff[3];
        detail::sub3(P, pt, diff);
        return detail::norm2_sq(diff);
    }
    
    // Check if P is in edge region of BC
    Real va = d3*d6 - d5*d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
    {
        Real w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        Real pt[3] = {B[0] + w*(C[0] - B[0]), B[1] + w*(C[1] - B[1]), B[2] + w*(C[2] - B[2])};
        if (closest)
        {
            closest[0] = pt[0];
            closest[1] = pt[1];
            closest[2] = pt[2];
        }
        Real diff[3];
        detail::sub3(P, pt, diff);
        return detail::norm2_sq(diff);
    }
    
    // P is inside face region. Compute barycentric coordinates and project
    Real denom = Real(1) / (va + vb + vc);
    Real v = vb * denom;
    Real w = vc * denom;
    
    Real pt[3] = {
        A[0] + AB[0]*v + AC[0]*w,
        A[1] + AB[1]*v + AC[1]*w,
        A[2] + AB[2]*v + AC[2]*w
    };
    
    if (closest)
    {
        closest[0] = pt[0];
        closest[1] = pt[1];
        closest[2] = pt[2];
    }
    
    Real diff[3];
    detail::sub3(P, pt, diff);
    return detail::norm2_sq(diff);
}

/// Compute unsigned distance from point P to triangle (A, B, C)
/// @return Distance (not squared)
template <typename Real>
Real point_triangle_distance(
    const Real* P, const Real* A, const Real* B, const Real* C,
    Real* closest = nullptr)
{
    return std::sqrt(point_triangle_distance_sq(P, A, B, C, closest));
}

/// Compute squared distance from point P to line segment (A, B)
/// @return Distance (not squared)
template <typename Real>
Real point_segment_distance(
    const Real* P, const Real* A, const Real* B,
    Real* closest = nullptr)
{
    return std::sqrt(point_segment_distance_sq(P, A, B, closest));
}

} // namespace cutfemx::level_set
