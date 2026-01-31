#pragma once

#include <cmath>
#include <algorithm>
#include <limits>
#include <array>
#include <span>

namespace cutfemx::level_set {

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

/// Compute 2-point update (Edge -> Vertex) using Analytic Solution
/// Minimizes f(t) = (1-t)d1 + t*d2 + |x_target - (x1 + t(x2-x1))|
template <typename Real>
Real update_2pt(const Real* x_target, 
               const Real* x1, Real d1,
               const Real* x2, Real d2)
{
    if (d1 >= std::numeric_limits<Real>::max()) return d1;
    if (d2 >= std::numeric_limits<Real>::max()) return update_1pt(x_target, x1, d1);
    
    Real u[3], v[3];
    detail::sub(x2, x1, u);        // u = x2 - x1 (Edge)
    detail::sub(x_target, x1, v);  // v = target - x1
    
    Real u_sq = detail::norm_sq(u);
    Real v_sq = detail::norm_sq(v);
    Real uv = detail::dot(u, v);
    
    Real delta = d2 - d1;
    Real delta_sq = delta * delta;
    
    // Gradient condition: |d2-d1| must be <= distance(x1,x2)
    if (delta_sq > u_sq) {
        return (d1 < d2) ? update_1pt(x_target, x1, d1) : update_1pt(x_target, x2, d2);
    }
    
    // Quadratic: At^2 + Bt + C = 0
    Real A = u_sq * (delta_sq - u_sq);
    Real B = 2 * uv * (u_sq - delta_sq);
    Real C = delta_sq * v_sq - uv * uv;
    
    Real min_val = std::min(update_1pt(x_target, x1, d1), update_1pt(x_target, x2, d2));
    
    // Solve Ax^2 + Bx + C = 0
    if (std::abs(A) < 1e-13 * u_sq) {
        if (std::abs(B) > 1e-13) {
            Real t = -C / B;
            if (t > 0.0 && t < 1.0) {
                 Real dx = v[0] - t*u[0];
                 Real dy = v[1] - t*u[1];
                 Real dz = v[2] - t*u[2];
                 min_val = std::min(min_val, d1 + t*delta + std::sqrt(dx*dx + dy*dy + dz*dz));
            }
        }
    } else {
        Real disc = B*B - 4*A*C;
        if (disc >= 0) {
            Real sqrt_disc = std::sqrt(disc);
            Real t1 = (-B - sqrt_disc) / (2*A);
            Real t2 = (-B + sqrt_disc) / (2*A);
            
            if (t1 > 0.0 && t1 < 1.0) {
                 Real t = t1;
                 Real dx = v[0] - t*u[0];
                 Real dy = v[1] - t*u[1];
                 Real dz = v[2] - t*u[2];
                 min_val = std::min(min_val, d1 + t*delta + std::sqrt(dx*dx + dy*dy + dz*dz));
            }
            if (t2 > 0.0 && t2 < 1.0) {
                 Real t = t2;
                 Real dx = v[0] - t*u[0];
                 Real dy = v[1] - t*u[1];
                 Real dz = v[2] - t*u[2];
                 min_val = std::min(min_val, d1 + t*delta + std::sqrt(dx*dx + dy*dy + dz*dz));
            }
        }
    }
    return min_val;
}

/// Compute 3-point update (Face -> Vertex) using Projected Gradient / Sampling
template <typename Real>
Real update_3pt(const Real* x_target, 
               const Real* x1, Real d1,
               const Real* x2, Real d2,
               const Real* x3, Real d3)
{
    // 1. Recursive check of edges first
    Real val = update_2pt(x_target, x1, d1, x2, d2);
    val = std::min(val, update_2pt(x_target, x2, d2, x3, d3));
    val = std::min(val, update_2pt(x_target, x3, d3, x1, d1));
    
    // 2. Interior Face Update via Projected Gradient Descent
    Real e1[3], e2[3], v_target[3];
    detail::sub(x2, x1, e1);
    detail::sub(x3, x1, e2);
    detail::sub(x_target, x1, v_target);
    
    // Check if triangle is degenerate
    Real cross[3];
    cross[0] = e1[1]*e2[2] - e1[2]*e2[1];
    cross[1] = e1[2]*e2[0] - e1[0]*e2[2];
    cross[2] = e1[0]*e2[1] - e1[1]*e2[0];
    if (detail::norm_sq(cross) < 1e-14) return val;

    Real dd1 = d2 - d1;
    Real dd2 = d3 - d1;
    
    // Initial guess: centroid (u=1/3, v=1/3)
    Real u = 1.0/3.0, v = 1.0/3.0;
    
    const int ITER = 5; 
    const Real step = 0.5;
    
    for(int i=0; i<ITER; ++i) {
        // x_curr = u*e1 + v*e2
        Real xc[3] = {u*e1[0] + v*e2[0], u*e1[1] + v*e2[1], u*e1[2] + v*e2[2]};
        // dist_geom = |target - (x1+xc)|
        Real diff[3] = {v_target[0]-xc[0], v_target[1]-xc[1], v_target[2]-xc[2]};
        Real dist_geom = std::sqrt(detail::norm_sq(diff));
        
        if (dist_geom < 1e-14) break;
        
        Real diff_dot_e1 = detail::dot(diff, e1);
        Real diff_dot_e2 = detail::dot(diff, e2);
        
        // Grad J w.r.t u,v
        Real grad_u = dd1 - diff_dot_e1 / dist_geom;
        Real grad_v = dd2 - diff_dot_e2 / dist_geom;
        
        u -= step * grad_u;
        v -= step * grad_v;
        
        // Project to triangle u>=0, v>=0, u+v<=1
        if (u < 0) u = 0;
        if (v < 0) v = 0;
        if (u + v > 1) {
            Real d = u + v - 1;
            u -= d/2;
            v -= d/2;
            if (u < 0) { u = 0; v = 1; }
            if (v < 0) { v = 0; u = 1; }
        }
        
        Real d_bary = d1 + u*dd1 + v*dd2;
        Real xc_new[3] = {u*e1[0] + v*e2[0], u*e1[1] + v*e2[1], u*e1[2] + v*e2[2]};
        Real diff_new[3] = {v_target[0]-xc_new[0], v_target[1]-xc_new[1], v_target[2]-xc_new[2]};
        Real final_dist = d_bary + std::sqrt(detail::norm_sq(diff_new));
        if (final_dist < val) val = final_dist;
    }
    
    return val;
}

} // namespace cutfemx::level_set
