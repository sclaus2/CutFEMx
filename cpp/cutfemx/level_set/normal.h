// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>

#include <dolfinx/common/types.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>

namespace cutfemx::level_set
{
template <typename X>
using normal_mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

/// Evaluate sign * grad(phi) / |grad(phi)| at parent-cell reference points.
///
/// The points are grouped by per-entity runtime quadrature offsets. The output
/// is flattened row-major with shape (num_points, gdim).
template <std::floating_point T>
std::vector<double> evaluate_normals(
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
    std::span<const dolfinx::scalar_value_t<T>> points, std::size_t num_points,
    std::size_t point_dim, std::span<const std::int32_t> offsets,
    std::span<const std::int32_t> parent_map, double sign)
{
  if (!level_set)
    throw std::runtime_error("Cannot evaluate normals without a level set.");
  if (offsets.size() != parent_map.size() + 1)
    throw std::runtime_error(
        "Normal evaluation expects offsets.size() == parent_map.size() + 1.");
  if (points.size() != num_points * point_dim)
    throw std::runtime_error("Normal evaluation point array has inconsistent shape.");

  using U = dolfinx::scalar_value_t<T>;

  auto V = level_set->function_space();
  auto mesh = V->mesh();
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();
  if (static_cast<int>(point_dim) != tdim)
    throw std::runtime_error("Normal evaluation points must have cell reference dimension.");

  auto cell_map = mesh->topology()->index_map(tdim);
  if (!cell_map)
    throw std::runtime_error("Cell index map is unavailable.");
  const std::int32_t num_cells = static_cast<std::int32_t>(
      cell_map->size_local() + cell_map->num_ghosts());

  auto element = V->element();
  const int ndofs = element->space_dimension();
  auto dofmap = V->dofmap();
  const std::span<const T>& level_set_values = level_set->x()->array();

  const dolfinx::fem::CoordinateElement<U>& cmap = mesh->geometry().cmap();
  auto x_dofmap = mesh->geometry().dofmap();
  const std::size_t num_geometry_dofs = cmap.dim();
  std::span<const U> x = mesh->geometry().x();

  std::array<std::size_t, 2> points_shape = {num_points, point_dim};

  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(1, num_points);
  std::vector<T> basis_b(std::reduce(basis_shape.begin(), basis_shape.end(),
                                     std::size_t(1), std::multiplies{}));
  element->tabulate(basis_b, points, points_shape, 1);
  cmdspan4_t basis(basis_b.data(), basis_shape);

  std::array<std::size_t, 4> cmap_basis_shape
      = cmap.tabulate_shape(1, num_points);
  std::vector<U> cmap_basis_b(
      std::reduce(cmap_basis_shape.begin(), cmap_basis_shape.end(),
                  std::size_t(1), std::multiplies{}));
  cmap.tabulate(1, points, points_shape, cmap_basis_b);
  using cmapspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const U, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  cmapspan4_t cmap_basis(cmap_basis_b.data(), cmap_basis_shape);

  std::vector<U> coord_dofs_b(num_geometry_dofs * static_cast<std::size_t>(gdim));
  normal_mdspan2_t<U> coord_dofs(coord_dofs_b.data(), num_geometry_dofs,
                                 static_cast<std::size_t>(gdim));
  std::vector<U> J_b(static_cast<std::size_t>(gdim * tdim));
  normal_mdspan2_t<U> J(J_b.data(), static_cast<std::size_t>(gdim),
                        static_cast<std::size_t>(tdim));
  std::vector<U> K_b(static_cast<std::size_t>(tdim * gdim));
  normal_mdspan2_t<U> K(K_b.data(), static_cast<std::size_t>(tdim),
                        static_cast<std::size_t>(gdim));

  std::vector<T> level_set_dofs(static_cast<std::size_t>(ndofs));
  std::vector<double> grad_ref(static_cast<std::size_t>(tdim));
  std::vector<double> grad_phys(static_cast<std::size_t>(gdim));
  std::vector<double> values(num_points * static_cast<std::size_t>(gdim));

  for (std::size_t rule = 0; rule < parent_map.size(); ++rule)
  {
    const std::int32_t cell = parent_map[rule];
    if (cell < 0 || cell >= num_cells)
      throw std::out_of_range("Normal evaluation parent_map contains an invalid cell.");

    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cell, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
      for (int j = 0; j < gdim; ++j)
        coord_dofs(i, static_cast<std::size_t>(j))
            = x[3 * x_dofs[i] + static_cast<std::size_t>(j)];

    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
    if (static_cast<int>(dofs.size()) != ndofs)
      throw std::runtime_error("Level-set dofmap size does not match element dimension.");
    for (std::size_t i = 0; i < dofs.size(); ++i)
      level_set_dofs[i] = level_set_values[dofs[i]];

    const std::int32_t q0 = offsets[rule];
    const std::int32_t q1 = offsets[rule + 1];
    if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
      throw std::runtime_error("Normal evaluation offsets are out of range.");

    for (std::int32_t q = q0; q < q1; ++q)
    {
      auto basis_deriv = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis, std::pair(1, tdim + 1), static_cast<std::size_t>(q),
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
      auto cmap_basis_deriv = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          cmap_basis, std::pair(1, tdim + 1), static_cast<std::size_t>(q),
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

      std::ranges::fill(J_b, U(0));
      std::ranges::fill(K_b, U(0));
      cmap.compute_jacobian(cmap_basis_deriv, coord_dofs, J);
      cmap.compute_jacobian_inverse(J, K);

      std::ranges::fill(grad_ref, 0.0);
      std::ranges::fill(grad_phys, 0.0);

      for (int i = 0; i < tdim; ++i)
        for (int j = 0; j < ndofs; ++j)
          grad_ref[static_cast<std::size_t>(i)]
              += static_cast<double>(
                  basis_deriv(static_cast<std::size_t>(i),
                              static_cast<std::size_t>(j))
                  * level_set_dofs[static_cast<std::size_t>(j)]);

      for (int i = 0; i < gdim; ++i)
        for (int j = 0; j < tdim; ++j)
          grad_phys[static_cast<std::size_t>(i)]
              += static_cast<double>(
                  K(static_cast<std::size_t>(j), static_cast<std::size_t>(i)))
                 * grad_ref[static_cast<std::size_t>(j)];

      double norm = 0.0;
      for (double value : grad_phys)
        norm += value * value;
      norm = std::sqrt(norm);
      if (norm < 1.0e-14)
        norm = 1.0e-14;

      for (int i = 0; i < gdim; ++i)
      {
        values[static_cast<std::size_t>(q * gdim + i)]
            = sign * grad_phys[static_cast<std::size_t>(i)] / norm;
      }
    }
  }

  return values;
}
} // namespace cutfemx::level_set
