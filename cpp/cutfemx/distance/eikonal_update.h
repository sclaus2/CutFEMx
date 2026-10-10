// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <cmath>
#include <algorithm>
#include <limits>
#include <array>
#include <span>

namespace cutfemx::distance {

namespace detail {

/// Squared norm
template <typename Real>
inline Real norm_sq(const Real* a) {
    return a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
}

/// Dot product
template <typename Real>
inline Real dot(const Real* a, const Real* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// Subtract
template <typename Real>
inline void sub(const Real* a, const Real* b, Real* out) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

}

/// Compute 1-point update (Edge Dijkstra)
/// d = d1 + dist(x1, x_target)
template <typename Real>
Real update_1pt(const Real* x_target, const Real* x1, Real d1)
{
    Real diff[3];
    detail::sub(x_target, x1, diff);
    return d1 + std::sqrt(detail::norm_sq(diff));
}

/// Compute 2-point update (Edge -> Vertex) in closed form.
/// Minimizes f(t) = (1-t)d1 + t*d2 + |x_target - (x1 + t(x2-x1))| over [0, 1].
///
/// f is convex. At an interior stationary point the unit vector from the
/// point to x_target makes the cosine c = (d2 - d1) / |x2 - x1| with the edge,
/// which fixes the point from the distance rho of x_target to the edge line.
template <typename Real>
Real update_2pt(const Real* x_target,
               const Real* x1, Real d1,
               const Real* x2, Real d2)
{
    if (d1 >= std::numeric_limits<Real>::max()) return d1;
    if (d2 >= std::numeric_limits<Real>::max()) return update_1pt(x_target, x1, d1);

    Real min_val = std::min(update_1pt(x_target, x1, d1),
                            update_1pt(x_target, x2, d2));

    Real e[3], r[3];
    detail::sub(x2, x1, e);
    detail::sub(x_target, x1, r);
    const Real length_sq = detail::norm_sq(e);
    if (length_sq <= Real(0))
        return min_val;
    const Real length = std::sqrt(length_sq);
    const Real c = (d2 - d1) / length;
    if (std::abs(c) >= Real(1))
        return min_val;

    // Position of the projection of x_target on the edge line, and the
    // distance rho of x_target to the line
    const Real along = detail::dot(r, e) / length;
    const Real rho = std::sqrt(std::max(detail::norm_sq(r) - along * along, Real(0)));

    // The stationary point lies c rho / sqrt(1 - c^2) before the projection
    const Real s = std::sqrt(Real(1) - c * c);
    const Real t = (along - c * rho / s) / length;
    if (t > Real(0) && t < Real(1))
        min_val = std::min(min_val, d1 + t * (d2 - d1) + rho / s);
    return min_val;
}

/// Interior part of the 3-point update (face -> vertex) in closed form.
///
/// Minimises F(p) = d(p) + |x_target - p| over the triangle (x1, x2, x3),
/// with d linear on the triangle. F is convex, so a stationary point inside
/// the triangle is the minimum over the whole triangle, including its edges
/// and vertices. At that point the unit vector g = (x_target - p) / |x_target - p|
/// satisfies g . (x2 - x1) = d2 - d1 and g . (x3 - x1) = d3 - d1, which
/// fixes its in-plane part; |g| = 1 fixes its normal part.
///
/// @return The minimum, or infinity if F has no stationary point inside the
/// triangle (then the minimum is on an edge and update_2pt gives it).
template <typename Real>
Real update_3pt_interior(const Real* x_target,
                         const Real* x1, Real d1,
                         const Real* x2, Real d2,
                         const Real* x3, Real d3)
{
    constexpr Real inf = std::numeric_limits<Real>::infinity();

    Real e1[3], e2[3], r[3];
    detail::sub(x2, x1, e1);
    detail::sub(x3, x1, e2);
    detail::sub(x_target, x1, r);

    const Real g11 = detail::dot(e1, e1);
    const Real g12 = detail::dot(e1, e2);
    const Real g22 = detail::dot(e2, e2);
    const Real det = g11 * g22 - g12 * g12;
    if (det <= Real(1e-12) * g11 * g22)
        return inf;

    // In-plane part of g: a e1 + b e2 with G (a, b) = (d2 - d1, d3 - d1)
    const Real dd1 = d2 - d1;
    const Real dd2 = d3 - d1;
    const Real a = (g22 * dd1 - g12 * dd2) / det;
    const Real b = (g11 * dd2 - g12 * dd1) / det;
    const Real gt_sq = a * dd1 + b * dd2;
    if (gt_sq >= Real(1))
        return inf;

    Real n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                 e1[2] * e2[0] - e1[0] * e2[2],
                 e1[0] * e2[1] - e1[1] * e2[0]};
    const Real n_norm = std::sqrt(detail::norm_sq(n));
    for (Real& c : n)
        c /= n_norm;

    // Height of the target over the face plane; the normal part of g points
    // from the face to the target.
    const Real h = detail::dot(r, n);
    const Real alpha = std::sqrt(Real(1) - gt_sq);
    if (std::abs(h) <= Real(1e-12) * std::sqrt(g11) || alpha <= Real(1e-12))
        return inf;

    // Foot point p = x_target - lambda g in the face plane
    const Real lambda = std::abs(h) / alpha;
    const Real sign = (h > 0) ? Real(1) : Real(-1);
    Real q[3];
    for (int k = 0; k < 3; ++k)
        q[k] = r[k] - lambda * (a * e1[k] + b * e2[k] + sign * alpha * n[k]);

    // Barycentric coordinates of p: G (s, t) = (e1 . q, e2 . q)
    const Real q1 = detail::dot(e1, q);
    const Real q2 = detail::dot(e2, q);
    const Real s = (g22 * q1 - g12 * q2) / det;
    const Real t = (g11 * q2 - g12 * q1) / det;
    constexpr Real eps = Real(1e-12);
    if (s < -eps || t < -eps || s + t > Real(1) + eps)
        return inf;

    return d1 + s * dd1 + t * dd2 + lambda;
}

/// Compute 3-point update (Face -> Vertex): the minimum of d(p) + |x_target - p|
/// over the triangle (x1, x2, x3), including its edges and vertices.
template <typename Real>
Real update_3pt(const Real* x_target,
               const Real* x1, Real d1,
               const Real* x2, Real d2,
               const Real* x3, Real d3)
{
    const Real interior = update_3pt_interior(x_target, x1, d1, x2, d2, x3, d3);
    if (interior < std::numeric_limits<Real>::infinity())
        return interior;

    Real val = update_2pt(x_target, x1, d1, x2, d2);
    val = std::min(val, update_2pt(x_target, x2, d2, x3, d3));
    val = std::min(val, update_2pt(x_target, x3, d3, x1, d1));
    return val;
}

} // namespace cutfemx::distance
