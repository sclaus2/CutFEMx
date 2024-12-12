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

  // One custom quadrature rule per cut cell / parent element
  // Custom quadrature rules in physical space
  // For each parent cell this quadrature needs to be pulled back before assembly
  template <std::floating_point T>
  struct QuadratureRules
  {
      std::vector<QuadratureRule<T>> _quadrature_rules;
      std::vector<std::int32_t> _parent_map;
  };

   //introduce notion of interface with pair of parent_cells ()
   template <std::floating_point T>
   struct QuadratureRulesInterface
   {
        std::vector<QuadratureRule<T>> _quadrature_rules;
        std::vector<std::pair<std::int32_t,std::int32_t>> _parent_map;
   };
}