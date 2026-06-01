// Copyright (c) 2022-2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cutcells/cell_flags.h>
#include <dolfinx_custom_data/fem/Form.h>

namespace cutfemx::fem
{
template <dolfinx::scalar T, std::floating_point U>
struct ActiveDomain;

namespace impl
{
template <dolfinx::scalar T, std::floating_point U>
void mark_cell_dofs(std::span<T> values, const dolfinx::fem::DofMap& dofmap,
                    std::int32_t cell, T value)
{
  const int bs = dofmap.bs();
  std::span<const std::int32_t> dofs = dofmap.cell_dofs(cell);
  for (std::int32_t dof : dofs)
    for (int k = 0; k < bs; ++k)
      values[bs * dof + k] = value;
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<std::int32_t>
deactivated_dofs(const dolfinx::fem::Function<T, U>& xi, bool owned_only)
{
  const std::vector<T>& values = xi.x()->array();
  std::size_t end = values.size();
  if (owned_only)
    end = xi.x()->index_map()->size_local() * xi.x()->bs();

  std::vector<std::int32_t> rows;
  rows.reserve(end);
  for (std::size_t dof = 0; dof < end; ++dof)
  {
    if (std::abs(values[dof]) < 1.0e-8)
      rows.push_back(static_cast<std::int32_t>(dof));
  }
  return rows;
}

inline std::vector<std::int32_t>
deduplicate_cells(std::vector<std::int32_t> cells)
{
  std::erase_if(cells, [](std::int32_t cell) { return cell < 0; });
  std::ranges::sort(cells);
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

template <dolfinx::scalar T, std::floating_point U>
void validate_bilinear_target_space(
    const dolfinx_custom_data::fem::Form<T, U>& form)
{
  if (form.rank() != 2)
  {
    throw std::invalid_argument(
        "cutfemx.fem.active_domain requires a rank-2 bilinear CutForm");
  }

  const auto& spaces = form.function_spaces();
  if (spaces.size() != 2 || !spaces[0] || !spaces[1])
    throw std::invalid_argument("CutForm has invalid function spaces");

  if (spaces[0]->mesh() != spaces[1]->mesh())
  {
    throw std::invalid_argument(
        "cutfemx.fem.active_domain requires trial and test spaces on the same mesh");
  }

  if (!(*spaces[0]->dofmap() == *spaces[1]->dofmap()))
  {
    throw std::invalid_argument(
        "cutfemx.fem.active_domain requires structurally equivalent trial "
        "and test spaces");
  }
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<std::int32_t>
collect_active_cells(const dolfinx_custom_data::fem::Form<T, U>& form)
{
  namespace cdf = dolfinx_custom_data::fem;
  const int tdim = form.mesh()->topology()->dim();
  auto cell_map = form.mesh()->topology()->index_map(tdim);
  if (!cell_map)
    throw std::runtime_error("Form mesh has no cell index map");
  const std::int32_t num_owned_cells
      = static_cast<std::int32_t>(cell_map->size_local());

  std::vector<std::int32_t> cells;
  for (const cdf::IntegralType type : form.integral_types())
  {
    const int num_integrals = form.num_integrals(type, 0);
    for (int idx = 0; idx < num_integrals; ++idx)
    {
      const std::span<const std::int32_t> entities = form.domain(type, idx, 0);
      switch (type)
      {
      case cdf::IntegralType::cell:
        for (const std::int32_t cell : entities)
          if (cell >= 0 && cell < num_owned_cells)
            cells.push_back(cell);
        break;
      case cdf::IntegralType::exterior_facet:
        if (entities.size() % 2 != 0)
          throw std::runtime_error("Exterior facet domain has invalid shape");
        for (std::size_t i = 0; i < entities.size(); i += 2)
          if (entities[i] >= 0 && entities[i] < num_owned_cells)
            cells.push_back(entities[i]);
        break;
      case cdf::IntegralType::interior_facet:
        if (entities.size() % 4 != 0)
          throw std::runtime_error("Interior facet domain has invalid shape");
        for (std::size_t i = 0; i < entities.size(); i += 4)
        {
          if (entities[i] >= 0 && entities[i] < num_owned_cells)
            cells.push_back(entities[i]);
          if (entities[i + 2] >= 0 && entities[i + 2] < num_owned_cells)
            cells.push_back(entities[i + 2]);
        }
        break;
      default:
        throw std::invalid_argument(
            "cutfemx.fem.active_domain currently supports cell and facet "
            "integrals only");
      }
    }
  }

  cells = deduplicate_cells(std::move(cells));
  if (cells.empty())
  {
    throw std::invalid_argument(
        "cutfemx.fem.active_domain found no active background cells");
  }
  return cells;
}

template <dolfinx::scalar T, std::floating_point U>
std::shared_ptr<dolfinx::fem::Function<T, U>> build_active_indicator(
    const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& V,
    std::span<const std::int32_t> active_cells)
{
  auto indicator = std::make_shared<dolfinx::fem::Function<T, U>>(V);
  indicator->name = "active_domain";
  std::vector<T>& values = indicator->x()->array();
  std::ranges::fill(values, T(0));

  const auto dofmap = V->dofmap();
  if (!dofmap)
    throw std::runtime_error("Active-domain function space has no dofmap");
  for (const std::int32_t cell : active_cells)
    mark_cell_dofs<T, U>(values, *dofmap, cell, T(1));

  indicator->x()->scatter_rev(std::plus<T>());
  indicator->x()->scatter_fwd();
  return indicator;
}

template <dolfinx::scalar T>
void set_vector_entries(std::span<T> values,
                        std::span<const std::int32_t> rows, T value)
{
  for (std::int32_t row : rows)
  {
    if (row < 0)
      throw std::runtime_error("Cannot set a negative vector row index.");
    const std::size_t index = static_cast<std::size_t>(row);
    if (index >= values.size())
      throw std::runtime_error("Deactivated vector row is out of range.");
    values[index] = value;
  }
}

template <dolfinx::scalar T, std::floating_point U>
void validate_inactive_rows(
    std::span<const std::int32_t> rows,
    const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& V)
{
  const auto dofmap = V->dofmap();
  if (!dofmap || !dofmap->index_map)
    throw std::runtime_error("Active-domain function space has no index map");
  const std::int32_t owned_rows
      = static_cast<std::int32_t>(dofmap->index_map->size_local()
                                  * dofmap->index_map_bs());
  if (!rows.empty() && rows.back() >= owned_rows)
  {
    throw std::runtime_error(
        "ActiveDomain inactive rows exceed the owned function-space row range");
  }
}

template <dolfinx::scalar T>
void validate_matrix_rows(const dolfinx::la::MatrixCSR<T>& A,
                          std::span<const std::int32_t> rows)
{
  const auto row_bs = A.block_size()[0];
  const std::int32_t local_rows
      = static_cast<std::int32_t>(A.index_map(0)->size_local() * row_bs);
  if (!rows.empty() && rows.back() >= local_rows)
  {
    throw std::runtime_error(
        "ActiveDomain inactive rows are incompatible with the matrix row map");
  }
}

template <dolfinx::scalar T>
std::int32_t owned_scalar_rows(const dolfinx::la::MatrixCSR<T>& A)
{
  return static_cast<std::int32_t>(A.num_owned_rows() * A.block_size()[0]);
}

template <dolfinx::scalar T>
bool scalar_row_has_nonzero(const dolfinx::la::MatrixCSR<T>& A,
                            std::int32_t row, double tol)
{
  if (row < 0 || row >= owned_scalar_rows(A))
    throw std::runtime_error("Matrix row is out of range for zero-row scan");

  const std::array<int, 2> bs = A.block_size();
  const std::int32_t block_row = row / bs[0];
  const int row_component = row % bs[0];
  const int block_size = bs[0] * bs[1];
  const auto& row_ptr = A.row_ptr();
  const auto& values = A.values();

  for (std::int64_t p = row_ptr[block_row]; p < row_ptr[block_row + 1]; ++p)
  {
    for (int col_component = 0; col_component < bs[1]; ++col_component)
    {
      const T value = values[p * block_size + row_component * bs[1]
                             + col_component];
      if (std::abs(value) > tol)
        return true;
    }
  }
  return false;
}

template <dolfinx::scalar T>
std::vector<std::int32_t> zero_rows(const dolfinx::la::MatrixCSR<T>& A,
                                    double tol = 0.0)
{
  std::vector<std::int32_t> rows;
  const std::int32_t num_rows = owned_scalar_rows(A);
  rows.reserve(num_rows);
  for (std::int32_t row = 0; row < num_rows; ++row)
    if (!scalar_row_has_nonzero(A, row, tol))
      rows.push_back(row);
  return rows;
}

template <dolfinx::scalar T>
std::vector<std::vector<std::int32_t>> zero_block_rows(
    const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
    double tol = 0.0)
{
  if (A_blocks.empty())
    throw std::runtime_error("Zero-row scan requires at least one block row");

  const std::size_t num_blocks = A_blocks.size();
  std::vector<std::vector<std::int32_t>> rows(num_blocks);
  for (std::size_t i = 0; i < num_blocks; ++i)
  {
    if (A_blocks[i].size() != num_blocks)
      throw std::runtime_error("Zero-row scan requires a square block matrix");
    if (A_blocks[i][i] == nullptr)
      throw std::runtime_error("Zero-row scan requires every diagonal matrix block");

    const std::int32_t num_rows = owned_scalar_rows(*A_blocks[i][i]);
    rows[i].reserve(num_rows);
    for (std::size_t j = 0; j < num_blocks; ++j)
    {
      if (A_blocks[i][j] != nullptr
          && owned_scalar_rows(*A_blocks[i][j]) != num_rows)
      {
        throw std::runtime_error(
            "Zero-row scan found incompatible row maps in a block row");
      }
    }

    for (std::int32_t row = 0; row < num_rows; ++row)
    {
      bool nonzero = false;
      for (std::size_t j = 0; j < num_blocks && !nonzero; ++j)
      {
        if (A_blocks[i][j] != nullptr)
          nonzero = scalar_row_has_nonzero(*A_blocks[i][j], row, tol);
      }
      if (!nonzero)
        rows[i].push_back(row);
    }
  }
  return rows;
}

template <dolfinx::scalar T>
void validate_block_rhs_inputs(
    const std::vector<dolfinx::la::Vector<T>*>& b_blocks,
    std::size_t num_blocks)
{
  if (b_blocks.size() != num_blocks)
  {
    throw std::runtime_error(
        "Block deactivation requires one RHS vector per block row");
  }
  for (dolfinx::la::Vector<T>* b : b_blocks)
  {
    if (b == nullptr)
      throw std::runtime_error("Block deactivation received a null RHS vector");
  }
}
} // namespace impl

template <dolfinx::scalar T, std::floating_point U>
struct ActiveDomain
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> function_space;
  std::shared_ptr<dolfinx::fem::Function<T, U>> indicator;
  std::vector<std::int32_t> active_cells;
  std::vector<std::int32_t> inactive_dofs;
};

namespace impl
{
template <dolfinx::scalar T, std::floating_point U>
void validate_block_deactivation_inputs(
    const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
    const std::vector<ActiveDomain<T, U>*>& active_domains)
{
  if (A_blocks.empty())
    throw std::runtime_error("Block deactivation requires at least one block row");
  if (A_blocks.size() != active_domains.size())
  {
    throw std::runtime_error(
        "Block deactivation requires one ActiveDomain per block row");
  }

  const std::size_t num_blocks = A_blocks.size();
  for (std::size_t i = 0; i < num_blocks; ++i)
  {
    if (A_blocks[i].size() != num_blocks)
    {
      throw std::runtime_error(
          "Block deactivation requires a square block matrix");
    }
    if (active_domains[i] == nullptr)
    {
      throw std::runtime_error(
          "Block deactivation received a null ActiveDomain");
    }
    if (A_blocks[i][i] == nullptr)
    {
      throw std::runtime_error(
          "Block deactivation requires every diagonal matrix block");
    }
    validate_matrix_rows(*A_blocks[i][i], active_domains[i]->inactive_dofs);
  }
}
} // namespace impl

template <dolfinx::scalar T, std::floating_point U>
ActiveDomain<T, U>
active_domain(const dolfinx_custom_data::fem::Form<T, U>& form)
{
  impl::validate_bilinear_target_space(form);
  ActiveDomain<T, U> out;
  out.function_space = form.function_spaces().at(0);
  out.active_cells = impl::collect_active_cells(form);
  out.indicator = impl::build_active_indicator<T, U>(out.function_space,
                                                    out.active_cells);
  out.inactive_dofs = impl::deactivated_dofs(*out.indicator, true);
  impl::validate_inactive_rows<T, U>(out.inactive_dofs, out.function_space);
  return out;
}

template <dolfinx::scalar T, std::floating_point U>
ActiveDomain<T, U>& deactivate_outside(
    auto set_fn, ActiveDomain<T, U>& active_domain, T diagonal = 1.0)
{
  dolfinx::fem::set_diagonal(set_fn, active_domain.inactive_dofs, diagonal);
  return active_domain;
}

template <dolfinx::scalar T, std::floating_point U>
ActiveDomain<T, U>& deactivate_outside(
    auto set_fn, std::span<T> b, ActiveDomain<T, U>& active_domain,
    T diagonal = 1.0, T rhs_value = 0)
{
  dolfinx::fem::set_diagonal(set_fn, active_domain.inactive_dofs, diagonal);
  impl::set_vector_entries<T>(b, active_domain.inactive_dofs, rhs_value);
  return active_domain;
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<ActiveDomain<T, U>*> deactivate_outside_blocks(
    const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
    const std::vector<ActiveDomain<T, U>*>& active_domains,
    T diagonal = 1.0)
{
  impl::validate_block_deactivation_inputs<T, U>(A_blocks, active_domains);

  for (std::size_t i = 0; i < A_blocks.size(); ++i)
  {
    // Inactive rows come only from the row-support ActiveDomain. Off-diagonal
    // blocks are intentionally untouched; inactive coupling entries indicate a
    // form/domain consistency bug rather than something to clean here.
    deactivate_outside(A_blocks[i][i]->mat_set_values(), *active_domains[i],
                       diagonal);
  }
  return active_domains;
}

template <dolfinx::scalar T, std::floating_point U>
std::vector<ActiveDomain<T, U>*> deactivate_outside_blocks(
    const std::vector<std::vector<dolfinx::la::MatrixCSR<T>*>>& A_blocks,
    const std::vector<dolfinx::la::Vector<T>*>& b_blocks,
    const std::vector<ActiveDomain<T, U>*>& active_domains, T diagonal = 1.0,
    T rhs_value = 0)
{
  impl::validate_block_deactivation_inputs<T, U>(A_blocks, active_domains);
  impl::validate_block_rhs_inputs<T>(b_blocks, A_blocks.size());

  for (std::size_t i = 0; i < A_blocks.size(); ++i)
  {
    auto& values = b_blocks[i]->array();
    deactivate_outside(A_blocks[i][i]->mat_set_values(),
                       std::span<T>(values.data(), values.size()),
                       *active_domains[i], diagonal, rhs_value);
  }
  return active_domains;
}
} // namespace cutfemx::fem
