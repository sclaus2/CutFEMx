// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cutfemx/distance/eikonal_update.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>

using namespace cutfemx::distance;

namespace
{
// min over the triangle of d(p) + |x - p| on a barycentric grid
double brute_force_face_update(const double* x, const double* x1, double d1,
                               const double* x2, double d2, const double* x3,
                               double d3, int n)
{
  double best = std::numeric_limits<double>::infinity();
  for (int i = 0; i <= n; ++i)
  {
    for (int j = 0; i + j <= n; ++j)
    {
      const double s = static_cast<double>(i) / n;
      const double t = static_cast<double>(j) / n;
      double dist_sq = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        const double p = x1[k] + s * (x2[k] - x1[k]) + t * (x3[k] - x1[k]);
        dist_sq += (x[k] - p) * (x[k] - p);
      }
      best = std::min(best, d1 + s * (d2 - d1) + t * (d3 - d1)
                                + std::sqrt(dist_sq));
    }
  }
  return best;
}
} // namespace

TEST_CASE("Closed-form face update matches the Hopf-Lax minimum",
          "[eikonal]")
{
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> coord(-1.0, 1.0);
  std::uniform_real_distribution<double> value(0.0, 1.0);

  int num_interior = 0;
  for (int trial = 0; trial < 500; ++trial)
  {
    std::array<double, 3> x, x1, x2, x3;
    for (int k = 0; k < 3; ++k)
    {
      x[k] = coord(gen);
      x1[k] = coord(gen);
      x2[k] = coord(gen);
      x3[k] = coord(gen);
    }

    // Values of a distance field, so that they are 1-Lipschitz on the face
    const std::array<double, 3> source{3.0 * coord(gen), 3.0 * coord(gen),
                                       3.0 * coord(gen)};
    auto distance = [&](const std::array<double, 3>& p)
    {
      return std::hypot(p[0] - source[0], p[1] - source[1], p[2] - source[2])
             + 0.1 * value(gen);
    };
    const double d1 = distance(x1);
    const double d2 = distance(x2);
    const double d3 = distance(x3);

    const int n = 400;
    const double exact = brute_force_face_update(
        x.data(), x1.data(), d1, x2.data(), d2, x3.data(), d3, n);
    const double update = update_3pt(x.data(), x1.data(), d1, x2.data(), d2,
                                     x3.data(), d3);

    // The grid minimum is an upper bound. The objective is Lipschitz in p
    // with constant 1 + |grad d|, so the grid minimum is within that times
    // the grid spacing of the true one.
    double e[2][3];
    for (int k = 0; k < 3; ++k)
    {
      e[0][k] = x2[k] - x1[k];
      e[1][k] = x3[k] - x1[k];
    }
    const double g11 = e[0][0] * e[0][0] + e[0][1] * e[0][1] + e[0][2] * e[0][2];
    const double g12 = e[0][0] * e[1][0] + e[0][1] * e[1][1] + e[0][2] * e[1][2];
    const double g22 = e[1][0] * e[1][0] + e[1][1] * e[1][1] + e[1][2] * e[1][2];
    const double det = g11 * g22 - g12 * g12;
    const double dd1 = d2 - d1, dd2 = d3 - d1;
    const double grad_sq = (g22 * dd1 * dd1 - 2 * g12 * dd1 * dd2 + g11 * dd2 * dd2) / det;
    const double spacing = std::sqrt(std::max({g11, g22, g11 + g22 - 2 * g12})) / n;
    CHECK(update <= exact + 1e-12);
    CHECK(update >= exact - (1.0 + std::sqrt(grad_sq)) * spacing);

    const double interior = update_3pt_interior(
        x.data(), x1.data(), d1, x2.data(), d2, x3.data(), d3);
    if (interior < std::numeric_limits<double>::infinity())
    {
      ++num_interior;
      CHECK_THAT(interior, Catch::Matchers::WithinAbs(update, 1e-12));
    }
  }
  // Make sure the interior branch is exercised
  CHECK(num_interior > 20);
}

TEST_CASE("Edge update matches the Hopf-Lax minimum", "[eikonal]")
{
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> coord(-1.0, 1.0);

  for (int trial = 0; trial < 1000; ++trial)
  {
    std::array<double, 3> x, x1, x2;
    for (int k = 0; k < 3; ++k)
    {
      x[k] = coord(gen);
      x1[k] = coord(gen);
      x2[k] = coord(gen);
    }
    const double length = std::hypot(x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]);
    const double d1 = 1.0 + coord(gen);
    // Equal values are common on symmetric meshes; the old quadratic lost
    // the interior minimum to a rounding-negative discriminant there.
    const double d2 = (trial % 2 == 0) ? d1 : d1 + 0.9 * length * coord(gen);

    const int n = 20000;
    double exact = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= n; ++i)
    {
      const double t = static_cast<double>(i) / n;
      double dist_sq = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        const double p = x1[k] + t * (x2[k] - x1[k]);
        dist_sq += (x[k] - p) * (x[k] - p);
      }
      exact = std::min(exact, d1 + t * (d2 - d1) + std::sqrt(dist_sq));
    }

    const double update = update_2pt(x.data(), x1.data(), d1, x2.data(), d2);
    CHECK(update <= exact + 1e-12);
    CHECK(update >= exact - 2.0 * length / n);
  }
}

TEST_CASE("Face update of a plane wave is exact", "[eikonal]")
{
  // d(p) = n . p for a unit vector n: the update reproduces it whenever the
  // characteristic through x crosses the face.
  const double n[3] = {0.48, 0.6, 0.64};
  auto plane = [&](const double* p)
  { return n[0] * p[0] + n[1] * p[1] + n[2] * p[2]; };

  const double x1[3] = {0.0, 0.0, 0.0};
  const double x2[3] = {1.0, 0.0, 0.0};
  const double x3[3] = {0.0, 1.0, 0.0};
  const double x[3] = {0.5, 0.5, 0.25};

  const double interior
      = update_3pt_interior(x, x1, plane(x1), x2, plane(x2), x3, plane(x3));
  CHECK_THAT(interior, Catch::Matchers::WithinAbs(plane(x), 1e-14));
}

TEST_CASE("Face update without interior minimum falls back to the edges",
          "[eikonal]")
{
  const double x1[3] = {0.0, 0.0, 0.0};
  const double x2[3] = {1.0, 0.0, 0.0};
  const double x3[3] = {0.0, 1.0, 0.0};

  // Target far outside the prism over the face
  const double x[3] = {3.0, 3.0, 0.5};
  CHECK(update_3pt_interior(x, x1, 0.0, x2, 0.0, x3, 0.0)
        == std::numeric_limits<double>::infinity());
  const double edge = update_2pt(x, x2, 0.0, x3, 0.0);
  CHECK_THAT(update_3pt(x, x1, 0.0, x2, 0.0, x3, 0.0),
             Catch::Matchers::WithinAbs(edge, 1e-14));

  // Target in the face plane
  const double y[3] = {0.25, 0.25, 0.0};
  CHECK(update_3pt_interior(y, x1, 0.0, x2, 0.1, x3, 0.2)
        == std::numeric_limits<double>::infinity());
}
