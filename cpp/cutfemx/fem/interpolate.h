// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
//#include "interpolate.h"
#pragma once

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Expression.h>

#include <basix/finite-element.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/utils.h>

#include <cutcells/cell_flags.h>
#include <cutcells/utils.h>
#include "../mesh/cut_mesh.h"

#include <span>
#include <string>
#include <memory>

namespace cutfemx::fem
{
  template <dolfinx::scalar T, std::floating_point U>
  dolfinx::fem::Function<T,U> create_cut_function(const dolfinx::fem::Function<T,U>& u,
                                             const cutfemx::mesh::CutMesh<U>& sub_mesh)
  {
    //use same function space as incoming function u
    auto element_in = u.function_space()->element();
    auto dofmap_in = u.function_space()->dofmap();
    const auto& _value_shape = u.function_space()->value_shape();
    std::vector<std::size_t> value_shape;
    value_shape.assign(_value_shape.begin(), _value_shape.end());
    int bs = dofmap_in->index_map_bs();

    const basix::FiniteElement<T>& basix_element = element_in->basix_element();
    auto sub_mesh_cell_type = sub_mesh._cut_mesh->topology()->cell_type();

    basix::FiniteElement<T> e = basix::create_element<T>(
      basix_element.family(),
      dolfinx::mesh::cell_type_to_basix_type(sub_mesh_cell_type), basix_element.degree(),
      basix_element.lagrange_variant(),
      basix_element.dpc_variant(),
      basix_element.discontinuous());

    const std::span<const T>& u_in_values = u.x()->array();
    auto V = std::make_shared<dolfinx::fem::FunctionSpace<U>>(dolfinx::fem::create_functionspace(sub_mesh._cut_mesh,e,value_shape));

    dolfinx::fem::Function<T,U> u_out(V);

    auto dofmap_out = V->dofmap();
    const int bs_dof = dofmap_out->bs();
    auto u_out_values = u_out.x()->mutable_array();

    assert(bs == bs_dof);

    //cell to vertex map for new cells
    std::size_t tdim = sub_mesh._cut_mesh->topology()->dim();
    std::size_t gdim = sub_mesh._cut_mesh->geometry().dim();

    auto x_dofmap
        = sub_mesh._cut_mesh->geometry().dofmap();
    std::span<const U> x_nodes = sub_mesh._cut_mesh->geometry().x();
    const std::size_t num_dofs_g = sub_mesh._cut_mesh->geometry().cmap().dim();

    std::set<int32_t> treated_dofs;

    // first copy in all cells which do not change
    for(int c=0;c<sub_mesh._parent_index.size();c++)
    {
      bool is_cut_cell = sub_mesh._is_cut_cell[c];
      if(is_cut_cell)
      {
        // do not do anything here treated in next loop
      }
      else{

        int parent_index = sub_mesh._parent_index[c];

        //copy
        std::span<const std::int32_t> dofs_in = dofmap_in->cell_dofs(parent_index);
        std::span<const std::int32_t> dofs_out = dofmap_out->cell_dofs(c);

        assert(dofs_in.size() == dofs_out.size()); //need to have same size to copy over value

        for(int dof=0;dof<dofs_out.size();dof++)
        {
          int32_t dof1 = dofs_out[dof];
          if(treated_dofs.find(dof1) == treated_dofs.end()) // dof not yet treated
          {
            for (int k = 0; k < bs_dof; ++k)
            {
              u_out_values[bs_dof * dofs_out[dof] + k]
                  = u_in_values[bs * dofs_in[dof] + k];
            }
            treated_dofs.insert(dofs_out[dof]);
          }
        }
      }
    }

    std::vector<U> new_points;
    std::vector<int32_t> new_points_cell_index; //index of cell points belong to
    std::vector<int32_t> interpolated_dofs;

    //Iterate over all cells of submesh if cell has not changed geometry fill in values
    // if it has changed geometry value needs to be interpolated in parent cell
    for(int c=0;c<sub_mesh._parent_index.size();c++)
    {
      int parent_index = sub_mesh._parent_index[c];
      bool is_cut_cell = sub_mesh._is_cut_cell[c];

      //FIXME: this should not be cell is new but rather vertex is new
      if(is_cut_cell)
      {
        //interpolate
        auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        std::span<const std::int32_t> dofs_out = dofmap_out->cell_dofs(c);

        for (std::size_t dof = 0; dof < x_dofs.size(); ++dof)
        {
          int32_t dof1 = dofs_out[dof];
          if (treated_dofs.find(dof1) != treated_dofs.end())
          {
            //dof already treated
          }
          else
          {
            std::vector<U> coordinate_dofs(3 * num_dofs_g);

            // Get cell coordinates/geometry
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
              std::copy_n(std::next(x_nodes.begin(), 3 * x_dofs[i]), 3,
                          std::next(coordinate_dofs.begin(), 3 * i));
            }
            std::vector<U> new_point(3);

            //get coordinate of vertex
            for(std::size_t j=0;j<3;j++)
            {
              new_point[j] = coordinate_dofs[dof*3+j];
            }

            int id = cutcells::utils::vertex_exists<T>(new_points, new_point, 0, 3);

            if(id==-1) //point does not exist yet
            {
              // check if dof has already been taken care of in "old" cells
              //FIXME: this might not work for high order approximation
              //vertex does not exist, and dof also not yet
              // add vertex to new_points
              new_points.insert(new_points.end(),new_point.begin(),new_point.end());
              new_points_cell_index.push_back(parent_index);
              interpolated_dofs.push_back(dof1);
            }
            else
            {
              //point already treated
            }
          }
        }
      }
    }

    // do the interpolation in new points
    int num_new_points = new_points.size() / 3;
    // Evaluate the interpolating function where possible
    const std::size_t value_size = u.function_space()->value_size();

    std::vector<T> send_values(num_new_points * value_size);
    u.eval(new_points, {static_cast<std::size_t>(num_new_points), (std::size_t)3},
          new_points_cell_index, send_values, {static_cast<std::size_t>(num_new_points), (std::size_t)value_size});

    // assign interpolated values
    for(int p=0;p<num_new_points;p++)
    {
      int32_t dof1 = interpolated_dofs[p];

      for(int j=0;j<bs_dof;j++)
      {
        u_out_values[bs_dof*dof1+j] = send_values[p*bs_dof+j];
      }
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
