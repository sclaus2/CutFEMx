// Copyright (c) 2022-2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/la/MatrixCSR.h>
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

  if (spaces[0]->element()->is_mixed() || spaces[1]->element()->is_mixed())
  {
    throw std::invalid_argument(
        "cutfemx.fem.active_domain does not support mixed spaces in v1");
  }
  if (*spaces[0]->element() != *spaces[1]->element()
      || !(*spaces[0]->dofmap() == *spaces[1]->dofmap()))
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
} // namespace impl

template <dolfinx::scalar T, std::floating_point U>
struct ActiveDomain
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> function_space;
  std::shared_ptr<dolfinx::fem::Function<T, U>> indicator;
  std::vector<std::int32_t> active_cells;
  std::vector<std::int32_t> inactive_dofs;
};

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
} // namespace cutfemx::fem
