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

#include <dolfinx/la/utils.h>

namespace cutfemx::fem
{
template <dolfinx::scalar T>
void assemble_cells(
    dolfinx::la::MatSet<T> auto mat_set, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x,
    std::shared_ptr<const cutfemx::quadrature::QuadratureRules<scalar_value_type_t<T>>> quadrature_rules,
    std::vector<std::pair<std::shared_ptr<basix::FiniteElement<scalar_value_type_t<T>>>, std::int32_t>>& elements,
    std::function<void(T*, const T*, const T*, const scalar_value_type_t<T>*,
                      const int*, const uint8_t*,
                      const int*, const scalar_value_type_t<T>*, const scalar_value_type_t<T>*,
                      const T*, const size_t*)>
    kernel,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap0,
    dolfinx::fem::DofTransformKernel<T> auto P0,
    std::tuple<mdspan2_t, int, std::span<const std::int32_t>> dofmap1,
    dolfinx::fem::DofTransformKernel<T> auto P1T, std::span<const std::int8_t> bc0,
    std::span<const std::int8_t> bc1,
    std::span<const T> coeffs, int cstride, std::span<const T> constants,
    std::span<const std::uint32_t> cell_info0,
    std::span<const std::uint32_t> cell_info1)
{
  if (quadrature_rules->_parent_map.empty())
    return;

  const auto [dmap0, bs0, cells0] = dofmap0;
  const auto [dmap1, bs1, cells1] = dofmap1;

  // Iterate over active cells
  const int num_dofs0 = dmap0.extent(1);
  const int num_dofs1 = dmap1.extent(1);
  const int ndim0 = bs0 * num_dofs0;
  const int ndim1 = bs1 * num_dofs1;
  std::vector<T> Ae(ndim0 * ndim1);
  std::span<T> _Ae(Ae);
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * x_dofmap.extent(1));

  // Iterate over active cells
  assert(cells0.size() == cells.size());
  assert(cells1.size() == cells.size());
  for (std::size_t index = 0; index < quadrature_rules->_parent_map.size(); ++index)
  {
    // Cell index in integration domain mesh (c), test function mesh
    // (c0) and trial function mesh (c1)
    std::int32_t c = quadrature_rules->_parent_map[index];
    std::int32_t c0 = cells0[index];
    std::int32_t c1 = cells1[index];

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

    // Tabulate tensor
    std::ranges::fill(Ae, 0);
    kernel(Ae.data(), coeffs.data() + index * cstride, constants.data(),
           coordinate_dofs.data(), nullptr, nullptr,
           &num_points,  points.data(), weights.data(), FE.data(), shape.data());

    // Compute A = P_0 \tilde{A} P_1^T (dof transformation)
    P0(_Ae, cell_info0, c0, ndim1);  // B = P0 \tilde{A}
    P1T(_Ae, cell_info1, c1, ndim0); // A =  B P1_T

    // Zero rows/columns for essential bcs
    auto dofs0 = std::span(dmap0.data_handle() + c0 * num_dofs0, num_dofs0);
    auto dofs1 = std::span(dmap1.data_handle() + c1 * num_dofs1, num_dofs1);

    if (!bc0.empty())
    {
      for (int i = 0; i < num_dofs0; ++i)
      {
        for (int k = 0; k < bs0; ++k)
        {
          if (bc0[bs0 * dofs0[i] + k])
          {
            // Zero row bs0 * i + k
            const int row = bs0 * i + k;
            std::fill_n(std::next(Ae.begin(), ndim1 * row), ndim1, 0);
          }
        }
      }
    }

    if (!bc1.empty())
    {
      for (int j = 0; j < num_dofs1; ++j)
      {
        for (int k = 0; k < bs1; ++k)
        {
          if (bc1[bs1 * dofs1[j] + k])
          {
            // Zero column bs1 * j + k
            const int col = bs1 * j + k;
            for (int row = 0; row < ndim0; ++row)
              Ae[row * ndim1 + col] = 0;
          }
        }
      }
    }

    mat_set(dofs0, dofs1, Ae);
  }
}

template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(
    dolfinx::la::MatSet<T> auto mat_set, const CutForm<T, U>& a, mdspan2_t x_dofmap,
    std::span<const scalar_value_type_t<T>> x, std::span<const T> constants,
    const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    std::span<const std::int8_t> bc0, std::span<const std::int8_t> bc1)
{
  // Integration domain mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);
  // Test function mesh
  auto mesh0 = a._form->function_spaces().at(0)->mesh();
  assert(mesh0);
  // Trial function mesh
  auto mesh1 = a._form->function_spaces().at(1)->mesh();
  assert(mesh1);

  // Get dofmap data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
      = a._form->function_spaces().at(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
      = a._form->function_spaces().at(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  auto dofs0 = dofmap0->map();
  const int bs0 = dofmap0->bs();
  auto dofs1 = dofmap1->map();
  const int bs1 = dofmap1->bs();

  auto element0 = a._form->function_spaces().at(0)->element();
  assert(element0);
  auto element1 = a._form->function_spaces().at(1)->element();
  assert(element1);
  dolfinx::fem::DofTransformKernel<T> auto P0
      = element0->template dof_transformation_fn<T>(dolfinx::fem::doftransform::standard);
  dolfinx::fem::DofTransformKernel<T> auto P1T
      = element1->template dof_transformation_right_fn<T>(
          dolfinx::fem::doftransform::transpose);

  std::span<const std::uint32_t> cell_info0;
  std::span<const std::uint32_t> cell_info1;
  if (element0->needs_dof_transformations()
      or element1->needs_dof_transformations() or a._form->needs_facet_permutations())
  {
    mesh0->topology_mutable()->create_entity_permutations();
    mesh1->topology_mutable()->create_entity_permutations();
    cell_info0 = std::span(mesh0->topology()->get_cell_permutation_info());
    cell_info1 = std::span(mesh1->topology()->get_cell_permutation_info());
  }

  for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
  {
    auto fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
    assert(fn);
    auto& [coeffs, cstride] = coefficients.at({dolfinx::fem::IntegralType::cell, i});
    std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules
    = a.quadrature_rules(dolfinx::fem::IntegralType::cell, i);
    std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>> elements
    = a.elements(dolfinx::fem::IntegralType::cell, i);
    assemble_cells(
        mat_set, x_dofmap, x, quadrature_rules, elements, fn,
        {dofs0, bs0, quadrature_rules->_parent_map}, P0,
        {dofs1, bs1, quadrature_rules->_parent_map}, P1T, bc0, bc1,
        coeffs, cstride, constants, cell_info0, cell_info1);
  }
}


} // namespace cutfemx::fem
