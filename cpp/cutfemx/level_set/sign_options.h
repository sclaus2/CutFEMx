#pragma once

#include <array>

namespace cutfemx::level_set {

enum class SignMode {
    LocalNormalBand,   ///< Method 1: Local closest triangle normal (fast, band only)
    ComponentAnchor,   ///< Method 2: Global sign via mesh connectivity barriers
    WindingNumber      ///< Method 3: Robust via generalized winding number
};

/// Options for sign computation
struct SignOptions {
    SignMode mode = SignMode::LocalNormalBand;

    // Method 1 options
    double band_max_dist = -1.0;  // -1 = use all computed nearfield (default)
    bool propagate_sign = true;   // Propagate sign from nearfield to farfield

    // Method 2 options
    bool use_boundary_as_outside_anchor = true;
    bool use_user_anchor_point = false;
    std::array<double, 3> anchor_point = {0, 0, 0};

    // Method 3 options
    double winding_threshold = 0.5;
    int max_triangles_per_query = 10000;
};

} // namespace cutfemx::level_set
