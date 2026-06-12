// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <array>
#include <concepts>
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
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/mesh/Mesh.h>

namespace cutfemx::level_set
{
/// Evaluate phi at parent-cell reference points.
///
/// The points are grouped by per-entity runtime quadrature offsets. The output
/// is flattened row-major with shape (num_points, 1).
template <std::floating_point T>
std::vector<double> evaluate_values(
    std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
    std::span<const dolfinx::scalar_value_t<T>> points, std::size_t num_points,
    std::size_t point_dim, std::span<const std::int32_t> offsets,
    std::span<const std::int32_t> parent_map)
{
  if (!level_set)
    throw std::runtime_error("Cannot evaluate level-set values without a level set.");
  if (offsets.size() != parent_map.size() + 1)
    throw std::runtime_error(
        "Level-set value evaluation expects offsets.size() == parent_map.size() + 1.");
  if (points.size() != num_points * point_dim)
    throw std::runtime_error(
        "Level-set value evaluation point array has inconsistent shape.");

  auto V = level_set->function_space();
  auto mesh = V->mesh();
  const int tdim = mesh->topology()->dim();
  if (static_cast<int>(point_dim) != tdim)
    throw std::runtime_error(
        "Level-set value evaluation points must have cell reference dimension.");

  auto cell_map = mesh->topology()->index_map(tdim);
  if (!cell_map)
    throw std::runtime_error("Cell index map is unavailable.");
  const std::int32_t num_cells = static_cast<std::int32_t>(
      cell_map->size_local() + cell_map->num_ghosts());

  auto element = V->element();
  const int ndofs = element->space_dimension();
  auto dofmap = V->dofmap();
  const std::span<const T>& level_set_values = level_set->x()->array();

  std::array<std::size_t, 2> points_shape = {num_points, point_dim};

  using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
  std::array<std::size_t, 4> basis_shape
      = element->basix_element().tabulate_shape(0, num_points);
  std::vector<T> basis_b(std::reduce(basis_shape.begin(), basis_shape.end(),
                                     std::size_t(1), std::multiplies{}));
  element->tabulate(basis_b, points, points_shape, 0);
  cmdspan4_t basis(basis_b.data(), basis_shape);

  std::vector<T> level_set_dofs(static_cast<std::size_t>(ndofs));
  std::vector<double> values(num_points);

  for (std::size_t rule = 0; rule < parent_map.size(); ++rule)
  {
    const std::int32_t cell = parent_map[rule];
    if (cell < 0 || cell >= num_cells)
    {
      throw std::out_of_range(
          "Level-set value evaluation parent_map contains an invalid cell.");
    }

    std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell);
    if (static_cast<int>(dofs.size()) != ndofs)
    {
      throw std::runtime_error(
          "Level-set dofmap size does not match element dimension.");
    }
    for (std::size_t i = 0; i < dofs.size(); ++i)
      level_set_dofs[i] = level_set_values[dofs[i]];

    const std::int32_t q0 = offsets[rule];
    const std::int32_t q1 = offsets[rule + 1];
    if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
      throw std::runtime_error("Level-set value evaluation offsets are out of range.");

    for (std::int32_t q = q0; q < q1; ++q)
    {
      double value = 0.0;
      for (int j = 0; j < ndofs; ++j)
      {
        value += static_cast<double>(
            basis(0, static_cast<std::size_t>(q), static_cast<std::size_t>(j), 0)
            * level_set_dofs[static_cast<std::size_t>(j)]);
      }
      values[static_cast<std::size_t>(q)] = value;
    }
  }

  return values;
}
} // namespace cutfemx::level_set
