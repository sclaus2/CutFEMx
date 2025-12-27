// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once


#include "../quadrature/quadrature.h"
#include "CutForm.h"
#include "typedefs.h"
#include "generate_tables.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

#include <ufcx.h>

#include <basix/finite-element.h>

#include <dolfinx/fem/assembler.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Topology.h>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/utils.h>

namespace cutfemx::fem
{
  /// Assemble vector over cells
  template <dolfinx::scalar T, int _bs = -1>
  void assemble_cells(
      dolfinx::fem::DofTransformKernel<T> auto P0, std::span<T> b, mdspan2_t x_dofmap,
      std::span<const scalar_value_type_t<T>> x,
      std::shared_ptr<const cutfemx::quadrature::QuadratureRules<scalar_value_type_t<T>>> quadrature_rules,
      std::vector<std::pair<std::shared_ptr<basix::FiniteElement<scalar_value_type_t<T>>>, std::int32_t>>& elements,
      std::function<void(T*, const T*, const T*, const scalar_value_type_t<T>*,
                      const int*, const uint8_t*,
                      const int*, const scalar_value_type_t<T>*, const scalar_value_type_t<T>*,
                      const T*, const size_t*)>
      kernel,
      std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap,
      std::span<const T> constants,
      std::span<const T> coeffs, int cstride,
      std::span<const std::uint32_t> cell_info0)
  {
    if (quadrature_rules->_parent_map.empty())
      return;

    const auto [dmap, bs, cells0] = dofmap;
    assert(_bs < 0 or _bs == bs);

    // Create data structures used in assembly
    std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));
    std::vector<T> be(bs * dmap.extent(1));
    std::span<T> _be(be);

    // Iterate over active cells
    for (std::size_t index = 0; index < quadrature_rules->_parent_map.size(); ++index)
    {
      // Integration domain cell and test function cell
      std::int32_t c = quadrature_rules->_parent_map[index];
      std::int32_t c0 = cells0[index];

      // Get cell coordinates/geometry
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      // unpack quadrature rule
      const auto& points = quadrature_rules->_quadrature_rules[index]._points;
      const auto& weights = quadrature_rules->_quadrature_rules[index]._weights;

      const int num_points = weights.size();

      std::vector<T> FE;
      std::vector<std::size_t> shape;

      generate_tables(elements,points,num_points, FE, shape);

      // Tabulate vector for cell
      std::ranges::fill(be, 0);
      kernel(be.data(), coeffs.data() + index * cstride, constants.data(),
            coordinate_dofs.data(), nullptr, nullptr,
            &num_points,  points.data(), weights.data(), FE.data(), shape.data());
      P0(_be, cell_info0, c0, 1);

      // Scatter cell vector to 'global' vector array
      auto dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          dmap, c0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      if constexpr (_bs > 0)
      {
        for (std::size_t i = 0; i < dofs.size(); ++i)
          for (int k = 0; k < _bs; ++k)
            b[_bs * dofs[i] + k] += be[_bs * i + k];
      }
      else
      {
        for (std::size_t i = 0; i < dofs.size(); ++i)
          for (int k = 0; k < bs; ++k)
            b[bs * dofs[i] + k] += be[bs * i + k];
      }
    }
  }

  template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void assemble_vector(
      std::span<T> b, const CutForm<T, U>& L, mdspan2_t x_dofmap,
      std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
      const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                    std::pair<std::span<const T>, int>>& coefficients)
  {
    // Integration domain mesh
    std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = L.mesh();
    assert(mesh);

    // Test function mesh
    auto mesh0 = L._form->function_spaces().at(0)->mesh();
    assert(mesh0);

    // Get dofmap data
    assert(L._form->function_spaces().at(0));
    auto element = L._form->function_spaces().at(0)->element();
    assert(element);
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap
        = L._form->function_spaces().at(0)->dofmap();
    assert(dofmap);
    auto dofs = dofmap->map();
    const int bs = dofmap->bs();

    dolfinx::fem::DofTransformKernel<T> auto P0
        = element->template dof_transformation_fn<T>(dolfinx::fem::doftransform::standard);

    std::span<const std::uint32_t> cell_info0;
    if (element->needs_dof_transformations() or L._form->needs_facet_permutations())
    {
      mesh0->topology_mutable()->create_entity_permutations();
      cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    }

    for (int i : L.integral_ids(cutfemx::fem::IntegralType::cutcell))
    {
      auto fn = L.kernel(cutfemx::fem::IntegralType::cutcell, i);
      assert(fn);
      auto& [coeffs, cstride] = coefficients.at({cutfemx::fem::IntegralType::cutcell, i});
      std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules
          = L.quadrature_rules(cutfemx::fem::IntegralType::cutcell, i);
      
      // Resolve element sources to basix elements
      std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>> elements;
      resolve_element_sources(L.element_sources(cutfemx::fem::IntegralType::cutcell, i),
                              *L.form(), elements);
      
      if (bs == 1)
      {
        assemble_cells<T, 1>(
            P0, b, x_dofmap, x, quadrature_rules, elements, fn,
            {dofs, bs, quadrature_rules->_parent_map}, constants,
            coeffs, cstride, cell_info0);
      }
      else if (bs == 3)
      {
        assemble_cells<T, 3>(
            P0, b, x_dofmap, x, quadrature_rules, elements, fn,
            {dofs, bs, quadrature_rules->_parent_map}, constants,
            coeffs, cstride, cell_info0);
      }
      else
      {
        assemble_cells(P0, b, x_dofmap, x, quadrature_rules, elements, fn,
                            {dofs, bs, quadrature_rules->_parent_map},
                            constants, coeffs, cstride, cell_info0);
      }
    }

  }

} // namespace cutfemx::fem
