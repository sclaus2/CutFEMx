// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>

#include <cutcells/edge_root.h>

namespace cutfemx::geometry
{
namespace detail
{
template <typename X>
using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

template <typename X>
using mdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

template <std::floating_point T>
double coordinate_diameter(const std::vector<T>& coordinates, int gdim)
{
  const std::size_t n = coordinates.size() / static_cast<std::size_t>(gdim);
  double diameter = 0.0;
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = i + 1; j < n; ++j)
    {
      double r2 = 0.0;
      for (int d = 0; d < gdim; ++d)
      {
        const double diff = static_cast<double>(
            coordinates[i * static_cast<std::size_t>(gdim) + d]
            - coordinates[j * static_cast<std::size_t>(gdim) + d]);
        r2 += diff * diff;
      }
      diameter = std::max(diameter, std::sqrt(r2));
    }
  }
  if (diameter <= 0.0)
    throw std::runtime_error("correction_distance encountered a zero-size cell.");
  return diameter;
}
} // namespace detail

/// Evaluate signed distances rho such that phi(x_q + rho d_q) = 0.
///
/// The root search is performed in physical coordinates along the supplied
/// physical direction field. Level-set values are evaluated through the parent
/// cell finite-element basis after pulling shifted physical points back to the
/// parent reference cell.
template <std::floating_point T>
std::vector<double> evaluate_correction_distances(
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
    std::span<const T> points, std::size_t num_points,
    std::size_t point_dim, std::span<const std::int32_t> offsets,
    std::span<const std::int32_t> parent_map,
    std::span<const double> directions, std::size_t direction_dim,
    double max_distance, double max_distance_factor, double tolerance,
    int max_iterations)
{
  if (!level_set)
    throw std::runtime_error("correction_distance requires a level set.");
  if (offsets.size() != parent_map.size() + 1)
  {
    throw std::runtime_error(
        "correction_distance expects offsets.size() == parent_map.size() + 1.");
  }
  if (points.size() != num_points * point_dim)
    throw std::runtime_error("correction_distance point array has inconsistent shape.");
  if (directions.size() != num_points * direction_dim)
  {
    throw std::runtime_error(
        "correction_distance direction array has inconsistent shape.");
  }
  if (tolerance <= 0.0)
    throw std::runtime_error("correction_distance tolerance must be positive.");
  if (max_iterations < 1)
    throw std::runtime_error("correction_distance max_iterations must be positive.");
  if (max_distance <= 0.0 && max_distance_factor <= 0.0)
  {
    throw std::runtime_error(
        "correction_distance requires max_distance > 0 or "
        "max_distance_factor > 0.");
  }

  auto V = level_set->function_space();
  auto mesh = V->mesh();
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();
  if (static_cast<int>(point_dim) != tdim)
  {
    throw std::runtime_error(
        "correction_distance points must have cell reference dimension.");
  }
  if (static_cast<int>(direction_dim) != gdim)
  {
    throw std::runtime_error(
        "correction_distance directions must have mesh geometry dimension.");
  }
  if (tdim <= 0 || tdim > 3 || gdim <= 0 || gdim > 3)
  {
    throw std::runtime_error(
        "correction_distance currently supports topological and geometric "
        "dimensions from one to three.");
  }

  auto cell_map = mesh->topology()->index_map(tdim);
  if (!cell_map)
    throw std::runtime_error("Cell index map is unavailable.");
  const std::int32_t num_cells = static_cast<std::int32_t>(
      cell_map->size_local() + cell_map->num_ghosts());

  auto element = V->element();
  const int ndofs = element->space_dimension();
  auto dofmap = V->dofmap();
  const std::span<const T>& level_set_values = level_set->x()->array();

  const dolfinx::fem::CoordinateElement<T>& cmap
      = mesh->geometry().cmaps().front();
  auto x_dofmap = mesh->geometry().dofmaps().front();
  const std::size_t num_geometry_dofs = cmap.dim();
  std::span<const T> x = mesh->geometry().x();
  if (x_dofmap.extent(1) != num_geometry_dofs)
  {
    throw std::runtime_error(
        "Mesh geometry dofmap does not match coordinate element dimension.");
  }

  std::array<std::size_t, 2> points_shape = {num_points, point_dim};
  const std::array<std::size_t, 4> cmap_basis_shape
      = cmap.tabulate_shape(0, num_points);
  std::vector<T> cmap_basis_b(
      std::reduce(cmap_basis_shape.begin(), cmap_basis_shape.end(),
                  std::size_t(1), std::multiplies{}));
  cmap.tabulate(0, points, points_shape, cmap_basis_b);
  detail::mdspan4_t<const T> cmap_basis(cmap_basis_b.data(), cmap_basis_shape);

  std::array<std::size_t, 4> phi0_shape = cmap.tabulate_shape(1, 1);
  std::vector<T> phi0_b(std::reduce(phi0_shape.begin(), phi0_shape.end(),
                                    std::size_t(1), std::multiplies{}));
  detail::mdspan4_t<const T> phi0(phi0_b.data(), phi0_shape);
  std::vector<T> reference_zero(static_cast<std::size_t>(tdim), T(0));
  cmap.tabulate(1, std::span<const T>(reference_zero.data(), reference_zero.size()),
                {1, static_cast<std::size_t>(tdim)}, phi0_b);
  auto dphi0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      phi0, std::pair(1, tdim + 1), 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, 1);
  std::vector<T> basis_b(std::reduce(basis_shape.begin(), basis_shape.end(),
                                     std::size_t(1), std::multiplies{}));
  detail::mdspan4_t<const T> basis(basis_b.data(), basis_shape);

  std::vector<T> coord_dofs_b(
      num_geometry_dofs * static_cast<std::size_t>(gdim));
  detail::mdspan2_t<T> coord_dofs(coord_dofs_b.data(), num_geometry_dofs,
                                  static_cast<std::size_t>(gdim));
  std::vector<T> J_b(static_cast<std::size_t>(gdim * tdim), T(0));
  detail::mdspan2_t<T> J(J_b.data(), static_cast<std::size_t>(gdim),
                         static_cast<std::size_t>(tdim));
  std::vector<T> K_b(static_cast<std::size_t>(tdim * gdim), T(0));
  detail::mdspan2_t<T> K(K_b.data(), static_cast<std::size_t>(tdim),
                         static_cast<std::size_t>(gdim));
  std::vector<T> xp_b(static_cast<std::size_t>(gdim), T(0));
  detail::mdspan2_t<T> xp(xp_b.data(), std::size_t(1),
                          static_cast<std::size_t>(gdim));
  std::vector<T> Xp_b(static_cast<std::size_t>(tdim), T(0));
  detail::mdspan2_t<T> Xp(Xp_b.data(), std::size_t(1),
                          static_cast<std::size_t>(tdim));
  std::vector<T> level_set_dofs(static_cast<std::size_t>(ndofs), T(0));
  std::vector<double> values(num_points, 0.0);

  auto evaluate_reference_phi = [&element, &basis_b, &basis, &level_set_dofs,
                                 ndofs, tdim](std::span<const T> X) -> double
  {
    element->tabulate(basis_b, X, {1, static_cast<std::size_t>(tdim)}, 0);
    double value = 0.0;
    for (int j = 0; j < ndofs; ++j)
    {
      value += static_cast<double>(
          basis(0, 0, static_cast<std::size_t>(j), 0)
          * level_set_dofs[static_cast<std::size_t>(j)]);
    }
    return value;
  };

  for (std::size_t rule = 0; rule < parent_map.size(); ++rule)
  {
    const std::int32_t cell = parent_map[rule];
    if (cell < 0 || cell >= num_cells)
    {
      throw std::out_of_range(
          "correction_distance parent_map contains an invalid cell.");
    }

    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      for (int d = 0; d < gdim; ++d)
      {
        coord_dofs(i, static_cast<std::size_t>(d))
            = x[3 * static_cast<std::size_t>(x_dofs[i])
                + static_cast<std::size_t>(d)];
      }
    }

    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
    if (static_cast<int>(dofs.size()) != ndofs)
    {
      throw std::runtime_error(
          "Level-set dofmap size does not match element dimension.");
    }
    for (std::size_t i = 0; i < dofs.size(); ++i)
      level_set_dofs[i] = level_set_values[dofs[i]];

    std::ranges::fill(J_b, T(0));
    std::ranges::fill(K_b, T(0));
    dolfinx::fem::CoordinateElement<T>::compute_jacobian(dphi0, coord_dofs, J);
    dolfinx::fem::CoordinateElement<T>::compute_jacobian_inverse(J, K);
    std::array<T, 3> x0{0, 0, 0};
    for (int d = 0; d < gdim; ++d)
      x0[static_cast<std::size_t>(d)] = coord_dofs(0, static_cast<std::size_t>(d));

    const double hK = detail::coordinate_diameter(coord_dofs_b, gdim);
    const double bound
        = max_distance > 0.0 ? max_distance : max_distance_factor * hK;

    const std::int32_t q0 = offsets[rule];
    const std::int32_t q1 = offsets[rule + 1];
    if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
      throw std::runtime_error("correction_distance offsets are out of range.");

    for (std::int32_t q = q0; q < q1; ++q)
    {
      const std::size_t point = static_cast<std::size_t>(q);
      for (int d = 0; d < gdim; ++d)
      {
        double value = 0.0;
        for (std::size_t j = 0; j < num_geometry_dofs; ++j)
        {
          value += static_cast<double>(
              coord_dofs(j, static_cast<std::size_t>(d))
              * cmap_basis(0, point, j, 0));
        }
        xp_b[static_cast<std::size_t>(d)] = static_cast<T>(value);
      }
      for (int d = 0; d < tdim; ++d)
      {
        Xp_b[static_cast<std::size_t>(d)] =
            points[point * static_cast<std::size_t>(point_dim)
                   + static_cast<std::size_t>(d)];
      }

      double direction_norm = 0.0;
      for (int d = 0; d < gdim; ++d)
      {
        const double value
            = directions[point * static_cast<std::size_t>(direction_dim)
                         + static_cast<std::size_t>(d)];
        direction_norm += value * value;
      }
      direction_norm = std::sqrt(direction_norm);
      if (direction_norm < 1.0e-14)
      {
        throw std::runtime_error(
            "correction_distance received a degenerate direction at "
            "quadrature point " + std::to_string(point) + ".");
      }

      auto phi_at_physical_point = [&](std::span<const T> physical) -> T
      {
        for (int d = 0; d < gdim; ++d)
          xp(0, static_cast<std::size_t>(d)) = physical[static_cast<std::size_t>(d)];

        if (cmap.is_affine())
        {
          dolfinx::fem::CoordinateElement<T>::pull_back_affine(Xp, K, x0, xp);
        }
        else
        {
          cmap.pull_back_nonaffine(Xp, xp, coord_dofs, tolerance,
                                   max_iterations);
        }

        return static_cast<T>(evaluate_reference_phi(
            std::span<const T>(Xp_b.data(), static_cast<std::size_t>(tdim))));
      };

      std::array<T, 3> p0{T(0), T(0), T(0)};
      std::array<T, 3> p1{T(0), T(0), T(0)};
      for (int d = 0; d < gdim; ++d)
      {
        const double direction
            = directions[point * static_cast<std::size_t>(direction_dim)
                         + static_cast<std::size_t>(d)]
              / direction_norm;
        p0[static_cast<std::size_t>(d)]
            = xp_b[static_cast<std::size_t>(d)]
              - static_cast<T>(bound * direction);
        p1[static_cast<std::size_t>(d)]
            = xp_b[static_cast<std::size_t>(d)]
              + static_cast<T>(bound * direction);
      }

      auto phi_on_segment = [&](std::span<const T> physical) -> T
      {
        return phi_at_physical_point(physical);
      };

      const auto info = cutcells::cell::edge_root::find_root_parameter_info<T>(
          std::span<const T>(p0.data(), static_cast<std::size_t>(gdim)),
          std::span<const T>(p1.data(), static_cast<std::size_t>(gdim)),
          phi_on_segment, cutcells::cell::edge_root::method::brent, T(0),
          max_iterations, static_cast<T>(tolerance),
          static_cast<T>(tolerance));
      if (!info.converged)
      {
        throw std::runtime_error(
            "correction_distance CutCells root solve did not converge at "
            "quadrature point " + std::to_string(point) + ", parent cell "
            + std::to_string(cell) + ", residual "
            + std::to_string(static_cast<double>(info.residual)) + ".");
      }
      values[point] = -bound + 2.0 * bound * static_cast<double>(info.t);
    }
  }

  return values;
}
} // namespace cutfemx::geometry
