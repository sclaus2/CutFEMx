// Copyright (c) 2022 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <span>
#include <basix/mdspan.hpp>

#include <petscsystypes.h>
#include <basix/finite-element.h>


#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/CoordinateElement.h>

#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/Mesh.h>

namespace cutfemx::level_set
{
  /// mdspan typedef
  template <typename X>
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

   template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void compute_normal(std::shared_ptr<dolfinx::fem::Function<T>> normal, std::shared_ptr<const dolfinx::fem::Function<T>> level_set, std::span<const int32_t> entities)
  {
    //Geometry data
    auto mesh = level_set->function_space()->mesh();
    assert(mesh == normal->function_space()->mesh());
    auto V = normal->function_space();

    const int tdim = mesh->topology()->dim();
    const int gdim = mesh->geometry().dim();
    auto cell_type = mesh->topology()->cell_type();

    const dolfinx::fem::CoordinateElement<U>& cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const U> x_g = mesh->geometry().x();

    // Evaluate coordinate map basis at reference interpolation points
    using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
    std::vector<T> coordinate_dofs(3 * x_dofmap.extent(1));

    //get finite element from level set function space and the number of shape functions
    std::shared_ptr<const dolfinx::fem::FiniteElement<U>> e0 = level_set->function_space()->element();
    const int ndofs0 = e0->space_dimension();

    auto normal_values = normal->x()->mutable_array();

    // Get dofmap
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();
    assert(dofmap);
    const std::span<const T>& ls_values = level_set->x()->array();

    std::shared_ptr<const dolfinx::fem::DofMap> dofmap_normal = V->dofmap();

    // evaluate basis and first order derivatives of basis of level set function
    // on dof points for function space of normal
    // Array of dof coordinates on reference element with shape `(num_points, tdim)`
    const auto [X, Xshape] = V->element()->interpolation_points();

    // give shape for up to first order derivatives at Xshape[0] points
    // [tdim + 1, Xshape[0], ndofs0, 1]
    auto basix_element = level_set->function_space()->element()->basix_element();
    std::array<std::size_t, 4> basis_shape = basix_element.tabulate_shape(1, Xshape[0]);
    std::vector<T> basis_b(
      std::reduce(basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
    cmdspan4_t basis(basis_b.data(), basis_shape);

    std::vector<U> coord_dofs_b(num_dofs_g * gdim);
    mdspan2_t<U> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

    //compute shape functions and their first order derivatives
    // at dof coordinate points of normal space in reference element
    e0->tabulate(basis_b, X, Xshape, 1);

    // Tabulate coordinate map basis derivatives at interpolation points
    // This is needed for computing the Jacobian correctly
    std::array<std::size_t, 4> cmap_basis_shape = cmap.tabulate_shape(1, Xshape[0]);
    std::vector<U> cmap_basis_b(
        std::reduce(cmap_basis_shape.begin(), cmap_basis_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(1, X, {Xshape[0], Xshape[1]}, cmap_basis_b);

    //allocate memory for Jacobian and inverse of Jacobian
    std::vector<U> J_b(gdim * tdim);
    mdspan2_t<U> J(J_b.data(), gdim, tdim);
    std::vector<U> K_b(tdim * gdim);
    mdspan2_t<U> K(K_b.data(), tdim, gdim);

    std::vector<U> det_scratch(2 * tdim * gdim);
    T detJ=0;

    //Iterate over given entities and compute normal on each of the given entities
    for(std::size_t i=0;i<entities.size();i++)
    {
      int entity_index = entities[i];

    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, entity_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      for (std::size_t j = 0; j < gdim; ++j)
        coord_dofs(i, j) = x_g[3 * x_dofs[i] + j];
    }

      // get level set values in dofs
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(entity_index);

      std::vector<T> ls_vals(dofs.size());
      for (std::size_t i = 0; i < dofs.size(); ++i)
          ls_vals[i] = ls_values[dofs[i]];

      // dofs_normal should be the points given by p
      std::span<const std::int32_t> dofs_normal = dofmap_normal->cell_dofs(entity_index);

      assert(dofs_normal.size()==Xshape[0]);

      for (std::size_t p = 0; p < Xshape[0]; ++p)
      {
        // The first index is the derivative, the second index is the point index
        // the third index is the basis function index, the fourth index is the basis function component.
        auto basis_deriv = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          basis, std::pair(1, tdim + 1), p,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
        
        // Use coordinate map basis derivatives for Jacobian computation
        // cmap_basis_shape is [tdim+1, Xshape[0], num_dofs_g, gdim]
        cmdspan4_t cmap_basis(cmap_basis_b.data(), cmap_basis_shape);
        auto cmap_basis_deriv = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            cmap_basis, std::pair(1, tdim + 1), p,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
        
        std::fill(J_b.begin(), J_b.end(), 0);
        std::fill(K_b.begin(), K_b.end(), 0);
        cmap.compute_jacobian(cmap_basis_deriv, coord_dofs, J);
        cmap.compute_jacobian_inverse(J, K);
        detJ = cmap.compute_jacobian_determinant(J, det_scratch);

        // Compute dX = K * dphi
        std::vector<T> grad_ls(tdim);
        std::fill(grad_ls.begin(), grad_ls.end(), 0);

        std::vector<T> tmp(tdim);
        std::fill(tmp.begin(), tmp.end(), 0);

        // Reference Space (tdim): grad(X) = (sum_ic^ndofs (ls(X_i)*dphi_i(X)/dX), sum_ic^ndofs (ls(X_i)*dphi_i(X)/dY)
        for (std::size_t i = 0; i < tdim; ++i)
        {
          for (std::size_t j = 0; j < ndofs0; ++j)
          {
            tmp[i] += basis_deriv(i,j)*ls_vals[j]; // K(i,j)*
          }
        }

        // Physical Space (gdim): grad_ls(x_i) = {J^-1}^T*grad(X), J^{-1} = K
        for (std::size_t i = 0; i < gdim; ++i)
        {
          for (std::size_t j = 0; j < tdim; ++j)
          {
            grad_ls[i] += K(j,i)*tmp[j]; // K(i,j)*
          }
        }

        // compute norm
        int dof = dofs_normal[p];
        T norm_grad_ls = 0.0;

        for (int i = 0; i < tdim; ++i)
        {
          norm_grad_ls += grad_ls[i]*grad_ls[i];
        }

        norm_grad_ls = sqrt(norm_grad_ls);

        double tol = 1e-14;
        if(std::fabs(norm_grad_ls)<1e-14)
          norm_grad_ls = tol;

        // set normal to n_K=grad(ls)/norm(grad(ls))
        for (int i = 0; i < tdim; ++i)
        {
          normal_values[tdim*dof+i] = grad_ls[i]/norm_grad_ls;
        }
      }
    }

    normal->x()->scatter_fwd();
  }

}