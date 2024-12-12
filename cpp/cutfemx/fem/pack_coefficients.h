// Copyright (c) 2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "../quadrature/quadrature.h"
#include "CutForm.h"

#include "typedefs.h"
#include "assemble_scalar.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>

#include <basix/finite-element.h>

#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Constant.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace cutfemx::fem
{

template <dolfinx::scalar T, std::floating_point U>
std::pair<std::vector<T>, int>
allocate_coefficient_storage(const CutForm<T, U>& form, dolfinx::fem::IntegralType integral_type,
                             int id)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>& coefficients
      = form._form->coefficients();
  const std::vector<int> offsets = form._form->coefficient_offsets();

  std::size_t num_entities = 0;
  int cstride = 0;
  if (!coefficients.empty())
  {
    cstride = offsets.back();
    num_entities = form.quadrature_rules(integral_type, id)->_parent_map.size();
    if (integral_type == dolfinx::fem::IntegralType::exterior_facet
        or integral_type == dolfinx::fem::IntegralType::interior_facet)
    {
      num_entities /= 2;
    }
  }

  return {std::vector<T>(num_entities * cstride), cstride};
}

template <dolfinx::scalar T, std::floating_point U>
std::map<std::pair<dolfinx::fem::IntegralType, int>, std::pair<std::vector<T>, int>>
allocate_coefficient_storage(const CutForm<T, U>& form)
{
  std::map<std::pair<dolfinx::fem::IntegralType, int>, std::pair<std::vector<T>, int>> coeffs;
  for (auto integral_type : form.integral_types())
  {
    for (int id : form.integral_ids(integral_type))
    {
      coeffs.emplace_hint(
          coeffs.end(), std::pair(integral_type, id),
          allocate_coefficient_storage(form, integral_type, id));
    }
  }

  return coeffs;
}

template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const CutForm<T, U>& form, dolfinx::fem::IntegralType integral_type,
                       int id, std::span<T> c, int cstride)
{
  // Get form coefficient offsets and dofmaps
  const std::vector<std::shared_ptr<const dolfinx::fem::Function<T, U>>>& coefficients
      = form._form->coefficients();
  const std::vector<int> offsets = form._form->coefficient_offsets();

  // Indicator for packing coefficients
  std::vector<int> active_coefficient(coefficients.size(), 0);
  if (!coefficients.empty())
  {
    switch (integral_type)
    {
    case dolfinx::fem::IntegralType::cell:
    {
      // Get indicator for all coefficients that are active in cell
      // integrals
      for (std::size_t i = 0; i < form.num_integrals(dolfinx::fem::IntegralType::cell); ++i)
      {
        for (auto idx : form.active_coeffs(dolfinx::fem::IntegralType::cell, i))
          active_coefficient[idx] = 1;
      }

      // Iterate over coefficients
      for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
      {
        if (!active_coefficient[coeff])
          continue;

        // Get coefficient mesh
        auto mesh = coefficients[coeff]->function_space()->mesh();
        assert(mesh);

        // Other integrals in the form might have coefficients defined over
        // entities of codim > 0, which don't make sense for cell integrals, so
        // don't pack them.
        if (const int codim
            = form._form->mesh()->topology()->dim() - mesh->topology()->dim();
            codim > 0)
        {
          throw std::runtime_error("Should not be packing coefficients with "
                                   "codim>0 in a cell integral");
        }

        std::vector<std::int32_t> cells
            = form.quadrature_rules(dolfinx::fem::IntegralType::cell, id)->_parent_map;
        std::span<const std::uint32_t> cell_info
            = dolfinx::fem::impl::get_cell_orientation_info(*coefficients[coeff]);
        dolfinx::fem::impl::pack_coefficient_entity(
            c, cstride, *coefficients[coeff], cell_info, cells, 1,
            [](auto entity) { return entity.front(); }, offsets[coeff]);
      }
      break;
    }
    // case IntegralType::exterior_facet:
    // {
    //   // Get indicator for all coefficients that are active in exterior
    //   // facet integrals
    //   for (std::size_t i = 0;
    //        i < form.num_integrals(IntegralType::exterior_facet); ++i)
    //   {
    //     for (auto idx : form.active_coeffs(IntegralType::exterior_facet, i))
    //       active_coefficient[idx] = 1;
    //   }

    //   // Iterate over coefficients
    //   for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
    //   {
    //     if (!active_coefficient[coeff])
    //       continue;

    //     auto mesh = coefficients[coeff]->function_space()->mesh();
    //     std::vector<std::int32_t> facets
    //         = form.domain(IntegralType::exterior_facet, id, *mesh);
    //     std::span<const std::uint32_t> cell_info
    //         = impl::get_cell_orientation_info(*coefficients[coeff]);
    //     impl::pack_coefficient_entity(
    //         c, cstride, *coefficients[coeff], cell_info, facets, 2,
    //         [](auto entity) { return entity.front(); }, offsets[coeff]);
    //   }
    //   break;
    // }
    // case IntegralType::interior_facet:
    // {
    //   // Get indicator for all coefficients that are active in interior
    //   // facet integrals
    //   for (std::size_t i = 0;
    //        i < form.num_integrals(IntegralType::interior_facet); ++i)
    //   {
    //     for (auto idx : form.active_coeffs(IntegralType::interior_facet, i))
    //       active_coefficient[idx] = 1;
    //   }

    //   // Iterate over coefficients
    //   for (std::size_t coeff = 0; coeff < coefficients.size(); ++coeff)
    //   {
    //     if (!active_coefficient[coeff])
    //       continue;

    //     auto mesh = coefficients[coeff]->function_space()->mesh();
    //     std::vector<std::int32_t> facets
    //         = form.domain(IntegralType::interior_facet, id, *mesh);

    //     std::span<const std::uint32_t> cell_info
    //         = impl::get_cell_orientation_info(*coefficients[coeff]);

    //     // Pack coefficient ['+']
    //     impl::pack_coefficient_entity(
    //         c, 2 * cstride, *coefficients[coeff], cell_info, facets, 4,
    //         [](auto entity) { return entity[0]; }, 2 * offsets[coeff]);

    //     // Pack coefficient ['-']
    //     impl::pack_coefficient_entity(
    //         c, 2 * cstride, *coefficients[coeff], cell_info, facets, 4,
    //         [](auto entity) { return entity[2]; },
    //         offsets[coeff] + offsets[coeff + 1]);
    //   }
    //   break;
    // }
    default:
      throw std::runtime_error(
          "Could not pack coefficient. Integral type not supported.");
    }
  }
}

template <dolfinx::scalar T, std::floating_point U>
void pack_coefficients(const CutForm<T, U>& form,
                       std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                std::pair<std::vector<T>, int>>& coeffs)
{
  for (auto& [key, val] : coeffs)
    pack_coefficients<T>(form, key.first, key.second, val.first, val.second);
}

}