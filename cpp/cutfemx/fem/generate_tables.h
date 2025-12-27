// Copyright (c) 2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "typedefs.h"

#include <basix/mdspan.hpp>
#include <basix/finite-element.h>

#include <dolfinx/common/types.h>

#include <concepts>
#include <vector>
#include <utility>
#include <memory>

namespace cutfemx::fem
{
 template <std::floating_point T>
    void generate_tables(std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>>& elements,
                          const std::vector<T>& points,
                          const int& num_points,
                          std::vector<T>& FE,
                          std::vector<std::size_t>& shape)
    {
      std::vector<std::vector<T>> phi_b(elements.size());
      std::vector<std::array<std::size_t, 4>> e_shapes(elements.size());
      std::vector<std::size_t> length(elements.size());

      for(std::size_t i =0 ;i < elements.size(); i++)
      {
        std::shared_ptr<basix::FiniteElement<T>> el = elements[i].first;
        std::int32_t deriv_order = elements[i].second;

        auto e_shape = el->tabulate_shape(deriv_order, num_points);
        e_shapes[i] = e_shape;

        // tabulate data
        length[i] = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
        phi_b[i].resize(length[i]);
        mdspan_t<T, 4> phi(phi_b[i].data(), e_shape);
        mdspand_t<const T, 2> points_span(points.data(), num_points, points.size()/num_points);

        el->tabulate(deriv_order, points_span, phi);
        
        // Check for NaN in tabulated values
        bool has_nan = false;
        for(std::size_t j = 0; j < length[i]; ++j) {
          if(std::isnan(phi_b[i][j])) {
            has_nan = true;
            break;
          }
        }
        if(has_nan) {
          std::cout << "  WARNING: Element " << i << " tabulation contains NaN!" << std::endl;
        }
      }

      //Concatenate tab_data and shape to pass to integration kernel function
      std::size_t total_length=0;
      for(std::size_t i =0 ;i < elements.size(); i++)
      {
        total_length += length[i];
      }

      FE.reserve(total_length);
      shape.reserve(4*elements.size());

      for(std::size_t i =0 ;i < elements.size(); i++)
      {
        FE.insert(FE.end(),phi_b[i].begin(),phi_b[i].end());
        shape.insert(shape.end(),e_shapes[i].begin(),e_shapes[i].end());
      }
      
    }
} // namespace cutfemx::fem
