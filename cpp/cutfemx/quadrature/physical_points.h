// Copyright (c) 2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "quadrature.h"

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>

#include <dolfinx/io/XDMFFile.h>

namespace cutfemx::quadrature
{
  template <typename T, std::size_t ndim>
using mdspand_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;

  //Map quadrature points from reference space to physical space in background mesh
  //this can be used for visualisation and debugging of runtime quadrature rules
  template <std::floating_point T>
  std::vector<T> physical_points(const QuadratureRules<T>& runtime_rules, const dolfinx::mesh::Mesh<T>& mesh)
  {
    int cnt = 0;

    //get background mesh geometry data
    // Get geometry data
    const dolfinx::fem::CoordinateElement<T>& cmap = mesh.geometry().cmap();
    auto x_dofmap = mesh.geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const T> x_g = mesh.geometry().x();

    std::size_t gdim = mesh.geometry().dim();
    auto tdim = mesh.topology()->dim();

    // Evaluate coordinate map basis at reference interpolation points
    using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
    std::vector<T> coordinate_dofs(3 * x_dofmap.extent(1));

    std::size_t num_points = 0;

    for(auto& rule: runtime_rules._quadrature_rules)
    {
      num_points += rule._weights.size();
    }

    std::vector<T> points_phys(num_points*gdim);
    mdspand_t<T, 2> p_phys(points_phys.data(), num_points, gdim);
    std::size_t offset =0;

    // map points to physical space
    for(auto& rule: runtime_rules._quadrature_rules)
    {
      auto parent_index = runtime_rules._parent_map[cnt];

      // Tabulate basis functions at quadrature points
      // nd, num_points
      auto e_shape = cmap.tabulate_shape(0, rule._weights.size());
      std::size_t length
          = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
      std::vector<T> phi_b(length);
      mdspand_t<T, 4> phi(phi_b.data(), e_shape);

      // nd, mdspan of points, phi
      cmap.tabulate(0, std::span(rule._points.data(),rule._points.size()), {rule._weights.size(),tdim}, std::span(phi_b.data(),phi_b.size()));

        // void tabulate(int nd, std::span<const T> X, std::array<std::size_t, 2> shape,
        //         std::span<T> basis)

      // Get cell coordinates/geometry
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, parent_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      //x[ip] = sum(v) vc[v]*p[ip]
      /// derivative, point index, basis function index, basis function component
      for(std::size_t ip=0;ip<rule._weights.size();ip++)
      {
        for(std::size_t i=0;i<gdim;i++)
        {
          for(std::size_t v=0;v<x_dofs.size();v++) //shape function index
          {
            points_phys[(ip+offset)*gdim+i] += coordinate_dofs[v*3+i]*phi(0, ip, v, 0);
          }
        }
      }
      offset += rule._weights.size();
      cnt++;
    }

    return std::move(points_phys);
  }
}