// Copyright (c) 2022 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <vector>
#include <concepts>

/// @brief Implementation of affine quadrature rules for cut cells
/// routines for non-affine elements are not implemented at the moment
namespace cutfemx::quadrature
{
  template <std::floating_point T>
  struct QuadratureRule
  {
      std::vector<T> _points;
      std::vector<T> _weights;
  };

  // One quadrature rule per cut cell per parent element/ pair of parent elements for interface
  template <std::floating_point T>
  struct QuadratureRules
  {
      std::vector<QuadratureRule<T>> _quadrature_rules;
      std::vector<std::int32_t> _parent_map;
  };
}