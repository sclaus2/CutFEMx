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
  /// Assemble functional over cells

  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  T assemble_cells(mdspan2_t x_dofmap, std::span<const U> x,
                    std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules,
                    std::vector<std::pair<std::shared_ptr<basix::FiniteElement<U>>, std::int32_t>>& elements,
                    std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const uint8_t*,
                                    const int*, const U*, const U*,
                                    const T*, const size_t*)>
                    fn,
                    std::span<const T> constants, std::span<const T> coeffs,
                    int cstride)
  {
    T value(0);
    if (quadrature_rules->_parent_map.empty())
      return value;

    // Create data structures used in assembly
    std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

    // Iterate over all cells
    for (std::size_t index = 0; index < quadrature_rules->_parent_map.size(); ++index)
    {
      std::int32_t c = quadrature_rules->_parent_map[index];

      const auto& points = quadrature_rules->_quadrature_rules[index]._points;
      const auto& weights = quadrature_rules->_quadrature_rules[index]._weights;

      const int num_points = weights.size();

      std::vector<T> FE;
      std::vector<std::size_t> shape;

      generate_tables(elements,points,num_points, FE, shape);

      // Get cell coordinates/geometry
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      const T* coeff_cell = coeffs.data() + index * cstride;
      fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(), nullptr,
        nullptr, &num_points,  points.data(), weights.data(), FE.data(), shape.data());;
    }

    return value;
  }

  template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  T assemble_scalar(
    const fem::CutForm<T, U>& M, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
  {
    T value = 0;
    for (int i : M.integral_ids(cutfemx::fem::IntegralType::cutcell))
    {
      auto fn = M.kernel(cutfemx::fem::IntegralType::cutcell, i);
      assert(fn);
      auto& [coeffs, cstride] = coefficients.at({cutfemx::fem::IntegralType::cutcell, i});
      std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules
          = M.quadrature_rules(cutfemx::fem::IntegralType::cutcell, i);
      std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>> elements
          = M.elements(cutfemx::fem::IntegralType::cutcell, i);
      value += assemble_cells(x_dofmap, x, quadrature_rules, elements, fn, constants, coeffs,
                                    cstride);
    }

    return value;
  }

} // namespace cutfemx::fem
