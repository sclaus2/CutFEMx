// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
//#include "interpolate.h"
#pragma once

#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>

#include <basix/finite-element.h>

#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>

#include "../mesh/cut_mesh.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cutfemx::fem
{
  template <dolfinx::scalar T, std::floating_point U>
  dolfinx::fem::Function<T,U> create_cut_function(const dolfinx::fem::Function<T,U>& u,
                                             const cutfemx::mesh::CutMesh<U>& sub_mesh)
  {
    if (!sub_mesh._cut_mesh)
      throw std::invalid_argument("Cannot create a cut function on an empty cut mesh");

    if (sub_mesh._parent_index.size() != sub_mesh._is_cut_cell.size())
      throw std::runtime_error("CutMesh parent_index and is_cut_cell sizes differ");

    auto element_in = u.function_space()->element();
    const basix::FiniteElement<U>& basix_element = element_in->basix_element();
    auto sub_mesh_cell_type = sub_mesh._cut_mesh->topology()->cell_type();

    basix::FiniteElement<U> basix_out = basix::create_element<U>(
        basix_element.family(),
        dolfinx::mesh::cell_type_to_basix_type(sub_mesh_cell_type),
        basix_element.degree(), basix_element.lagrange_variant(),
        basix_element.dpc_variant(), basix_element.discontinuous());

    std::optional<std::vector<std::size_t>> value_shape = std::nullopt;
    if (!element_in->value_shape().empty())
    {
      value_shape = std::vector<std::size_t>(element_in->value_shape().begin(),
                                             element_in->value_shape().end());
    }

    auto element_out = std::make_shared<const dolfinx::fem::FiniteElement<U>>(
        basix_out, value_shape, element_in->symmetric());
    auto V = std::make_shared<const dolfinx::fem::FunctionSpace<U>>(
        dolfinx::fem::create_functionspace(sub_mesh._cut_mesh, element_out));
    dolfinx::fem::Function<T, U> u_out(V);

    auto cell_map = sub_mesh._cut_mesh->topology()->index_map(
        sub_mesh._cut_mesh->topology()->dim());
    if (!cell_map)
      throw std::runtime_error("Cut mesh has no cell index map");
    const std::size_t num_local_cells = cell_map->size_local();
    if (num_local_cells != sub_mesh._parent_index.size())
      throw std::runtime_error("CutMesh parent_index does not match cut mesh cells");

    std::vector<U> dof_coordinates = V->tabulate_dof_coordinates(false);
    const std::size_t num_dofs = dof_coordinates.size() / 3;
    std::vector<std::int32_t> parent_for_dof(num_dofs, -1);

    auto dofmap_out = V->dofmap();
    for (std::size_t c = 0; c < sub_mesh._parent_index.size(); ++c)
    {
      std::span<const std::int32_t> dofs = dofmap_out->cell_dofs(c);
      for (std::int32_t dof : dofs)
      {
        if (dof < 0 || static_cast<std::size_t>(dof) >= parent_for_dof.size())
          throw std::runtime_error("Cut function dof index is out of range");
        if (parent_for_dof[static_cast<std::size_t>(dof)] < 0)
          parent_for_dof[static_cast<std::size_t>(dof)]
              = sub_mesh._parent_index[c];
      }
    }

    std::vector<U> points;
    std::vector<std::int32_t> cells;
    std::vector<std::int32_t> dofs;
    points.reserve(3 * num_dofs);
    cells.reserve(num_dofs);
    dofs.reserve(num_dofs);
    for (std::size_t dof = 0; dof < parent_for_dof.size(); ++dof)
    {
      if (parent_for_dof[dof] < 0)
        continue;
      points.insert(points.end(), dof_coordinates.begin() + 3 * dof,
                    dof_coordinates.begin() + 3 * dof + 3);
      cells.push_back(parent_for_dof[dof]);
      dofs.push_back(static_cast<std::int32_t>(dof));
    }

    const std::size_t num_points = cells.size();
    const int value_size = element_in->value_size();
    const int bs = dofmap_out->index_map_bs();
    if (bs != value_size)
    {
      throw std::runtime_error(
          "create_cut_function currently supports blocked point-evaluation "
          "spaces with dofmap block size equal to value size");
    }

    std::vector<T> values(num_points * static_cast<std::size_t>(value_size));
    u.eval(std::span<const U>(points.data(), points.size()), {num_points, 3},
           std::span<const std::int32_t>(cells.data(), cells.size()),
           std::span<T>(values.data(), values.size()),
           {num_points, static_cast<std::size_t>(value_size)}, 1.0e-6, 15);

    auto& out_values = u_out.x()->array();
    std::ranges::fill(out_values, T(0));
    for (std::size_t p = 0; p < dofs.size(); ++p)
    {
      const std::size_t dof = static_cast<std::size_t>(dofs[p]);
      for (int k = 0; k < bs; ++k)
        out_values[bs * dof + k] = values[p * static_cast<std::size_t>(bs) + k];
    }

    return u_out;
  }

  /// @brief Interpolate an expression onto a CutMesh function space using parent cell mapping.
  ///
  /// The expression is evaluated at its built-in interpolation points, but using the
  /// parent cell indices from sub_mesh to ensure evaluation within valid CutFEM cells.
  /// This is essential for expressions involving gradients (e.g., stress), where values
  /// outside the active domain would be polluted by deactivation.
  ///
  /// @tparam T Scalar type (e.g., double)
  /// @tparam U Geometry type (e.g., double)
  /// @param expr The compiled expression (with interpolation points already set)
  /// @param V_cut Target function space on the CutMesh
  /// @param sub_mesh The CutMesh containing parent_index mapping
  /// @return Function on V_cut with expression values
  template <dolfinx::scalar T, std::floating_point U>
  dolfinx::fem::Function<T,U> interpolate_cut_expression(
      dolfinx::fem::Expression<T, U>& expr,
      const std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>& V_cut,
      const cutfemx::mesh::CutMesh<U>& sub_mesh)
  {
    // 1. Create output function
    dolfinx::fem::Function<T,U> u_out(V_cut);
    
    // 2. Get mesh info
    const std::size_t num_cells = sub_mesh._parent_index.size();
    const std::size_t value_size = expr.value_size();
    
    // Number of points per cell in the expression
    // Expression stores points as (num_points, gdim) flat array
    auto [x_ref_vec, x_ref_shape] = expr.X();
    const std::size_t num_pts_per_cell = x_ref_shape[0];
    
    // 3. Evaluate expression for each cut cell using its parent cell index
    // We call eval once per cell with the parent cell index
    auto dofmap_out = V_cut->dofmap();
    const int bs_dof = dofmap_out->bs();
    auto u_out_values = u_out.x()->mutable_array();
    
    // Buffer for one cell's worth of values
    std::vector<T> cell_values(num_pts_per_cell * value_size);
    
    for (std::size_t c = 0; c < num_cells; ++c)
    {
      int32_t parent_c = sub_mesh._parent_index[c];
      
      // Evaluate expression at this parent cell
      // Expression::eval(mesh, entities, values, vshape)
      std::array<int32_t, 1> entities = {parent_c};
      expr.eval(*sub_mesh._bg_mesh, 
                std::span<const std::int32_t>(entities.data(), 1),
                std::span<T>(cell_values.data(), cell_values.size()),
                {1, num_pts_per_cell * value_size});
      
      // Assign to output function
      std::span<const std::int32_t> dofs_out = dofmap_out->cell_dofs(c);
      
      // For DG0: num_pts_per_cell == 1, dofs_out.size() == 1
      // For higher order: need proper mapping
      for (std::size_t p = 0; p < std::min(num_pts_per_cell, dofs_out.size()); ++p)
      {
        int32_t dof = dofs_out[p];
        for (std::size_t k = 0; k < value_size; ++k)
        {
          u_out_values[bs_dof * dof + k] = cell_values[p * value_size + k];
        }
      }
    }
    
    return u_out;
  }
}
