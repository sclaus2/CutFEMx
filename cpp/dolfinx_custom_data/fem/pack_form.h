// Copyright (C) 2013-2025 Johan Hake, Jan Blechta, Garth N. Wells and Paul T.
// Kühner
//
// This file is derived from DOLFINx (https://www.fenicsproject.org)
// `cpp/dolfinx_custom_data/fem/pack.h`, reduced to the adapter needed for
// `dolfinx_custom_data::fem::Form`.
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Form.h"
#include "forward.h"
#include <algorithm>
#include <basix/mdspan.hpp>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/pack.h>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace dolfinx_custom_data::fem
{
/// @brief Allocate storage for coefficients of a pair `(integral_type, id)`
/// from a CutFEMx form.
template <dolfinx::scalar T, std::floating_point U>
std::pair<std::vector<T>, int>
allocate_coefficient_storage(const Form<T, U>& form, IntegralType integral_type,
                             int id)
{
  std::size_t num_entities = 0;
  int cstride = 0;
  if (const auto& coefficients = form.coefficients(); !coefficients.empty())
  {
    const std::vector<int> offsets = form.coefficient_offsets();
    cstride = offsets.back();
    num_entities = form.domain(integral_type, id, 0).size();
    if (integral_type != IntegralType::cell)
      num_entities /= 2;
  }

  return {std::vector<T>(num_entities * cstride), cstride};
}

/// @brief Allocate storage for all coefficient blocks of a CutFEMx form.
template <dolfinx::scalar T, std::floating_point U>
std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>>
allocate_coefficient_storage(const Form<T, U>& form)
{
  std::map<std::pair<IntegralType, int>, std::pair<std::vector<T>, int>> coeffs;
  for (IntegralType type : form.integral_types())
  {
    for (int i = 0; i < form.num_integrals(type, 0); ++i)
    {
      coeffs.emplace_hint(coeffs.end(), std::pair{type, i},
                          allocate_coefficient_storage(form, type, i));
    }
  }

  return coeffs;
}

/// @brief Pack coefficients of a CutFEMx form.
template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const Form<T, U>& form,
                       std::map<std::pair<IntegralType, int>,
                                std::pair<std::vector<T>, int>>& coeffs)
{
  const auto& coefficients = form.coefficients();
  const std::vector<int> offsets = form.coefficient_offsets();

  for (auto& [integral_data, coeff_data] : coeffs)
  {
    auto [integral_type, id] = integral_data;
    std::vector<T>& c = coeff_data.first;
    int cstride = coeff_data.second;
    if (coefficients.empty())
      continue;

    switch (integral_type)
    {
    case IntegralType::cell:
      for (int coeff : form.active_coeffs(IntegralType::cell, id))
      {
        auto mesh = coefficients[coeff]->function_space()->mesh();
        assert(mesh);
        if (int codim = form.mesh()->topology()->dim() - mesh->topology()->dim();
            codim > 0)
        {
          throw std::runtime_error("Should not pack coefficients with codim > "
                                   "0 in a cell integral.");
        }

        std::span<const std::int32_t> cells_b
            = form.domain_coeff(IntegralType::cell, id, coeff);
        md::mdspan cells(cells_b.data(), cells_b.size());
        std::span<const std::uint32_t> cell_info
            = dolfinx::fem::impl::get_cell_orientation_info(
                *coefficients[coeff]);
        dolfinx::fem::impl::pack_coefficient_entity(
            std::span(c), cstride, *coefficients[coeff], cell_info, cells,
            offsets[coeff]);
      }
      break;

    case IntegralType::interior_facet:
      for (int coeff : form.active_coeffs(IntegralType::interior_facet, id))
      {
        std::span<const std::int32_t> facets_b
            = form.domain_coeff(IntegralType::interior_facet, id, coeff);
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 4>>
            facets(facets_b.data(), facets_b.size() / 4, 4);
        std::span<const std::uint32_t> cell_info
            = dolfinx::fem::impl::get_cell_orientation_info(
                *coefficients[coeff]);

        auto cells0 = md::submdspan(facets, md::full_extent, 0);
        dolfinx::fem::impl::pack_coefficient_entity(
            std::span(c), 2 * cstride, *coefficients[coeff], cell_info, cells0,
            2 * offsets[coeff]);

        auto cells1 = md::submdspan(facets, md::full_extent, 2);
        dolfinx::fem::impl::pack_coefficient_entity(
            std::span(c), 2 * cstride, *coefficients[coeff], cell_info, cells1,
            offsets[coeff] + offsets[coeff + 1]);
      }
      break;

    case IntegralType::exterior_facet:
    case IntegralType::vertex:
    case IntegralType::ridge:
      for (int coeff : form.active_coeffs(integral_type, id))
      {
        std::span<const std::int32_t> entities_b
            = form.domain_coeff(integral_type, id, coeff);
        md::mdspan<const std::int32_t,
                   md::extents<std::size_t, md::dynamic_extent, 2>>
            entities(entities_b.data(), entities_b.size() / 2, 2);
        std::span<const std::uint32_t> cell_info
            = dolfinx::fem::impl::get_cell_orientation_info(
                *coefficients[coeff]);
        dolfinx::fem::impl::pack_coefficient_entity(
            std::span(c), cstride, *coefficients[coeff], cell_info,
            md::submdspan(entities, md::full_extent, 0), offsets[coeff]);
      }
      break;

    default:
      throw std::runtime_error(
          "Could not pack coefficient. Integral type not supported.");
    }
  }
}

/// @brief Pack constants of a CutFEMx form.
template <dolfinx::scalar T, std::floating_point U>
std::vector<T> pack_constants(const Form<T, U>& form)
{
  std::vector<std::reference_wrapper<const dolfinx::fem::Constant<T>>> c;
  std::ranges::transform(
      form.constants(), std::back_inserter(c),
      [](auto c) -> const dolfinx::fem::Constant<T>& { return *c; });
  return dolfinx::fem::pack_constants(c);
}
} // namespace dolfinx_custom_data::fem
