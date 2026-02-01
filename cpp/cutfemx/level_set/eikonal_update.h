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
    // Use stable quadratic formula or robust check
    // If A ~ 0, linear
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
            
            auto check_t = [&](Real t) {
                if (t > 0.0 && t < 1.0) {
                     Real dx = v[0] - t*u[0];
                     Real dy = v[1] - t*u[1];
                     Real dz = v[2] - t*u[2];
                     min_val = std::min(min_val, d1 + t*delta + std::sqrt(dx*dx + dy*dy + dz*dz));
                }
            };
            check_t(t1);
            check_t(t2);
        }
    }
    return min_val;
}

/// Compute 3-point update (Face -> Vertex) using Projected Newton's Method
/// Objective: F(u,v) = d1 + u(d2-d1) + v(d3-d1) + |target - (x1 + u*e1 + v*e2)|
/// This is convex. We use Newton-Raphson with projection to the triangle (u>=0, v>=0, u+v<=1).
template <typename Real>
Real update_3pt(const Real* x_target, 
               const Real* x1, Real d1,
               const Real* x2, Real d2,
               const Real* x3, Real d3)
{
    // 1. Edges/Vertices check (Boundary of simplex)
    Real val = update_2pt(x_target, x1, d1, x2, d2);
    val = std::min(val, update_2pt(x_target, x2, d2, x3, d3));
    val = std::min(val, update_2pt(x_target, x3, d3, x1, d1));
    
    // 2. Interior check
    Real e1[3], e2[3], v_target[3];
    detail::sub(x2, x1, e1);
    detail::sub(x3, x1, e2);
    detail::sub(x_target, x1, v_target);
    
    // Degeneracy check
    Real cross[3];
    cross[0] = e1[1]*e2[2] - e1[2]*e2[1];
    cross[1] = e1[2]*e2[0] - e1[0]*e2[2];
    cross[2] = e1[0]*e2[1] - e1[1]*e2[0];
    if (detail::norm_sq(cross) < 1e-14) return val;

    Real dd1 = d2 - d1;
    Real dd2 = d3 - d1;
    
    // Dot products needed for Hessian
    Real e11 = detail::dot(e1,e1);
    Real e12 = detail::dot(e1,e2);
    Real e22 = detail::dot(e2,e2);
    
    // Initial guess: centroid (u=1/3, v=1/3)
    Real u = 1.0/3.0, v = 1.0/3.0;
    
    const int MAX_ITER = 20;
    
    for(int iter=0; iter<MAX_ITER; ++iter) {
        // Current position xc = u*e1 + v*e2
        // r = xc - v_target (Note: F uses |v_target - xc|, so let r = v_target - xc)
        // No, let's keep r = v_target - xc for gradient
        Real r[3];
        for(int k=0; k<3; ++k) r[k] = v_target[k] - (u*e1[k] + v*e2[k]);
        
        Real dist_sq = detail::norm_sq(r);
        Real dist_geom = std::sqrt(dist_sq);
        
        if (dist_geom < 1e-14) break; // Should not happen if target is outside face
        
        Real r_dot_e1 = detail::dot(r, e1);
        Real r_dot_e2 = detail::dot(r, e2);
        
        // Gradient of F:
        // dF/du = dd1 - (r . e1) / |r|   (Note sign flip: d(-|r|)/du = - - (r . -e1)/|r| ... wait)
        // Let phi(u,v) = |r(u,v)|. r = v_target - (u*e1 + v*e2).
        // d(phi)/du = (1/2|r|) * 2 * r . dr/du = (r . -e1) / |r|.
        // So dF/du = dd1 - (r . e1) / |r|. Correct.
        Real inv_dist = 1.0 / dist_geom;
        Real g1 = dd1 - r_dot_e1 * inv_dist;
        Real g2 = dd2 - r_dot_e2 * inv_dist;
        
        // Convergence check
        if (std::abs(g1) < 1e-6 && std::abs(g2) < 1e-6) break;
        
        // Hessian of F:
        // H = (1/|r|) * ( J^T J - (J^T r)(J^T r)^T / |r|^2 )
        // J = [e1, e2]. J^T J = [[e11, e12], [e12, e22]]
        // J^T r = [r.e1, r.e2]
        // Term2_ij = (r.ei * r.ej) / |r|^2
        
        Real t1 = r_dot_e1 * inv_dist; // (r.e1)/|r|
        Real t2 = r_dot_e2 * inv_dist; // (r.e2)/|r|
        
        Real H11 = (e11 - t1*t1) * inv_dist;
        Real H12 = (e12 - t1*t2) * inv_dist;
        Real H22 = (e22 - t2*t2) * inv_dist;
        
        // Invert Hessian (2x2)
        Real detH = H11*H22 - H12*H12;
        Real du = 0, dv = 0;
        
        if (detH > 1e-14) {
             // Newton Step: -H^-1 g
             du = -(H22*g1 - H12*g2) / detH;
             dv = -(-H12*g1 + H11*g2) / detH;
        } else {
             // Fallback to Gradient Descent if Hessian degenerate (e.g. flat)
             du = -0.5 * g1;
             dv = -0.5 * g2;
        }
        
        // Line Search / Damped Newton
        Real alpha = 1.0;
        Real u_new = u, v_new = v;
        bool valid = false;
        
        Real current_val = d1 + u*dd1 + v*dd2 + dist_geom;
        
        for(int k=0; k<5; ++k) {
             u_new = u + alpha * du;
             v_new = v + alpha * dv;
             
             // Projection to constraints
             if (u_new < 0) u_new = 0;
             if (v_new < 0) v_new = 0;
             if (u_new + v_new > 1) {
                 Real shift = u_new + v_new - 1.0;
                 u_new -= shift/2;
                 v_new -= shift/2;
                 // Corner fix
                 if (u_new < 0) { u_new=0; v_new=1; }
                 if (v_new < 0) { v_new=0; u_new=1; }
             }
             
             // Evaluate
             Real r_new[3];
             for(int j=0; j<3; ++j) r_new[j] = v_target[j] - (u_new*e1[j] + v_new*e2[j]);
             Real val_new = d1 + u_new*dd1 + v_new*dd2 + std::sqrt(detail::norm_sq(r_new));
             
             if (val_new < current_val) {
                 valid = true;
                 break; // Decrease found
             }
             alpha *= 0.5;
        }
        
        if (valid) {
            u = u_new;
            v = v_new;
        } else {
            // Cannot decrease, probably at minimum or constrained minimum
            break;
        }
    }
    
    // Final check inside triangle (already projected, but check bounds for validity of 3pt update)
    if (u >= -1e-8 && v >= -1e-8 && u+v <= 1.0 + 1e-8) {
         Real r[3];
         for(int k=0; k<3; ++k) r[k] = v_target[k] - (u*e1[k] + v*e2[k]);
         Real final = d1 + u*dd1 + v*dd2 + std::sqrt(detail::norm_sq(r));
         val = std::min(val, final);
    }
    
    return val;
}

} // namespace cutfemx::level_set
