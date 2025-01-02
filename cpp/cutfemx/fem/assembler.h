// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "../quadrature/quadrature.h"
#include "CutForm.h"

#include "typedefs.h"
#include "assemble_scalar.h"
#include "assemble_vector.h"
#include "assemble_matrix.h"
#include "pack_coefficients.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>

#include <basix/finite-element.h>

#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/sparsitybuild.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace cutfemx::fem
{
    template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  T assemble_scalar(
      const CutForm<T, U>& M, std::span<const T> constants,
      const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                    std::pair<std::span<const T>, int>>& coefficients,
      const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                    std::pair<std::span<const T>, int>>& coeffs_rt)
  {
    std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = M._form->mesh();
    assert(mesh);

    T val = dolfinx::fem::impl::assemble_scalar<T,U>(*M._form, mesh->geometry().dofmap(),
                                  mesh->geometry().x(), constants, coefficients);
    T val_cut = assemble_scalar(M, mesh->geometry().dofmap(),
                                  mesh->geometry().x(), constants, coeffs_rt);
    val += val_cut;

    return val;
  }

    template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  T assemble_scalar(const CutForm<T, U>& M)
  {
    const std::vector<T> constants = dolfinx::fem::pack_constants(*M._form);
    auto coefficients = dolfinx::fem::allocate_coefficient_storage(*M._form);
    dolfinx::fem::pack_coefficients(*M._form, coefficients);

    //pack coefficients of runtime integrals
    auto coeffs_rt = cutfemx::fem::allocate_coefficient_storage(M);
    cutfemx::fem::pack_coefficients(M,coeffs_rt);

    T val = assemble_scalar(M, std::span(constants),
                          dolfinx::fem::make_coefficients_span(coefficients),
                          dolfinx::fem::make_coefficients_span(coeffs_rt));

    return val;
  }

  template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void assemble_vector(
      std::span<T> b, const CutForm<T, U>& L, std::span<const T> constants,
      const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                    std::pair<std::span<const T>, int>>& coefficients,
      const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                    std::pair<std::span<const T>, int>>& coeffs_rt)
  {
    std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = L._form->mesh();
    assert(mesh);

    dolfinx::fem::impl::assemble_vector(b, *L._form, mesh->geometry().dofmap(), mesh->geometry().x(),
                    constants, coefficients);

    assemble_vector(b, L, mesh->geometry().dofmap(), mesh->geometry().x(),
                    constants, coeffs_rt);
  }

  template <dolfinx::scalar T, std::floating_point U>
  void assemble_vector(std::span<T> b, const CutForm<T, U>& L)
  {
    auto coefficients = dolfinx::fem::allocate_coefficient_storage(*L._form);
    dolfinx::fem::pack_coefficients(*L._form, coefficients);
    //pack coefficients of runtime integrals
    auto coeffs_rt = cutfemx::fem::allocate_coefficient_storage(L);
    cutfemx::fem::pack_coefficients(L,coeffs_rt);

    const std::vector<T> constants = dolfinx::fem::pack_constants(*L._form);
    assemble_vector(b, L, std::span(constants),
                    dolfinx::fem::make_coefficients_span(coefficients),
                    dolfinx::fem::make_coefficients_span(coeffs_rt));
  }

      //Initialize diagonal for all rows
    template <dolfinx::scalar T, std::floating_point U>
    void init_diagonal(const CutForm<T,U>& a, dolfinx::la::SparsityPattern& pattern)
    {
      const dolfinx::fem::DofMap& dofmap = *a._form->function_spaces().at(0)->dofmap();

      //Insert one value in diagonal for all dofs
      auto map0 = a._form->function_spaces().at(0)->dofmap()->index_map;
      auto bs0 = a._form->function_spaces().at(0)->dofmap()->index_map_bs();

      const std::int32_t num_rows = (map0->size_local() + map0->num_ghosts());

      std::vector<std::int32_t> rows(num_rows);
      std::iota (std::begin(rows), std::end(rows), 0);

      pattern.insert_diagonal(rows);
    }

template <dolfinx::scalar T, std::floating_point U>
void create_sparsity_pattern(const CutForm<T,U>& a, dolfinx::la::SparsityPattern& pattern)
{
  // Get dof maps and mesh
  std::array<std::reference_wrapper<const dolfinx::fem::DofMap>, 2> dofmaps{
      *a._form->function_spaces().at(0)->dofmap(),
      *a._form->function_spaces().at(1)->dofmap()};

  const std::set<cutfemx::fem::IntegralType> types = a.integral_types();

  for (auto type : types)
  {
    std::vector<int> ids = a.integral_ids(type);
    switch (type)
    {
    case cutfemx::fem::IntegralType::cutcell:
      for (int id : ids)
      {
        std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules
        = a.quadrature_rules(type, id);
        dolfinx::fem::sparsitybuild::cells(
            pattern, {quadrature_rules->_parent_map, quadrature_rules->_parent_map},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }
}

  template <dolfinx::scalar T, std::floating_point U>
dolfinx::la::SparsityPattern create_sparsity_pattern(const CutForm<T, U>& a)
{
  if (a._form->rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear.");
  }

  // create sparsity pattern with stadard cells
  dolfinx::la::SparsityPattern pattern = dolfinx::fem::create_sparsity_pattern(*a._form);
  init_diagonal(a,pattern);
    // Get dof maps and mesh
  std::array<std::reference_wrapper<const dolfinx::fem::DofMap>, 2> dofmaps{
      *a._form->function_spaces().at(0)->dofmap(),
      *a._form->function_spaces().at(1)->dofmap()};

  const std::set<cutfemx::fem::IntegralType> types = a.integral_types();

  for (auto type : types)
  {
    std::vector<int> ids = a.integral_ids(type);
    switch (type)
    {
    case cutfemx::fem::IntegralType::cutcell:
      for (int id : ids)
      {
        std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules
        = a.quadrature_rules(type, id);
        dolfinx::fem::sparsitybuild::cells(
            pattern, {quadrature_rules->_parent_map, quadrature_rules->_parent_map},
            {{dofmaps[0], dofmaps[1]}});
      }
      break;
    default:
      throw std::runtime_error("Unsupported integral type");
    }
  }

  return pattern;
}

template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
void assemble_matrix(
    dolfinx::la::MatSet<T> auto mat_add, const CutForm<T, U>& a,
    std::span<const T> constants,
    const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                std::pair<std::span<const T>, int>>& coeffs_rt,
    std::span<const std::int8_t> dof_marker0,
    std::span<const std::int8_t> dof_marker1)

{
  std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = a.mesh();
  assert(mesh);

  dolfinx::fem::impl::assemble_matrix(mat_add, *a._form, mesh->geometry().dofmap(),
                          mesh->geometry().x(), constants, coefficients,
                          dof_marker0, dof_marker1);
  assemble_matrix(mat_add, a, mesh->geometry().dofmap(),
                          mesh->geometry().x(), constants, coeffs_rt,
                          dof_marker0, dof_marker1);
}

template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
void assemble_matrix(
    auto mat_add, const CutForm<T, U>& a, std::span<const T> constants,
    const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients,
    const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                std::pair<std::span<const T>, int>>& coeffs_rt,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
{
  // Index maps for dof ranges
  auto map0 = a._form->function_spaces().at(0)->dofmap()->index_map;
  auto map1 = a._form->function_spaces().at(1)->dofmap()->index_map;
  auto bs0 = a._form->function_spaces().at(0)->dofmap()->index_map_bs();
  auto bs1 = a._form->function_spaces().at(1)->dofmap()->index_map_bs();

  // Build dof markers
  std::vector<std::int8_t> dof_marker0, dof_marker1;
  assert(map0);
  std::int32_t dim0 = bs0 * (map0->size_local() + map0->num_ghosts());
  assert(map1);
  std::int32_t dim1 = bs1 * (map1->size_local() + map1->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (a._form->function_spaces().at(0)->contains(*bcs[k]->function_space()))
    {
      dof_marker0.resize(dim0, false);
      bcs[k]->mark_dofs(dof_marker0);
    }

    if (a._form->function_spaces().at(1)->contains(*bcs[k]->function_space()))
    {
      dof_marker1.resize(dim1, false);
      bcs[k]->mark_dofs(dof_marker1);
    }
  }

  // Assemble
  assemble_matrix(mat_add, a, constants, coefficients, coeffs_rt, dof_marker0,
                  dof_marker1);
}

template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
void assemble_matrix(
    auto mat_add, const CutForm<T, U>& a,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>>& bcs)
{
  // Prepare constants and coefficients
  const std::vector<T> constants = dolfinx::fem::pack_constants(*a._form);
  auto coefficients = dolfinx::fem::allocate_coefficient_storage(*a._form);
  dolfinx::fem::pack_coefficients(*a._form, coefficients);
  //pack coefficients of runtime integrals
  auto coeffs_rt = cutfemx::fem::allocate_coefficient_storage(a);
  cutfemx::fem::pack_coefficients(a,coeffs_rt);

  // Assemble
  assemble_matrix(mat_add, a, std::span(constants),
                  dolfinx::fem::make_coefficients_span(coefficients),
                  dolfinx::fem::make_coefficients_span(coeffs_rt), bcs);
}

template <dolfinx::scalar T, std::floating_point U>
void assemble_matrix(auto mat_add, const CutForm<T, U>& a,
                     std::span<const std::int8_t> dof_marker0,
                     std::span<const std::int8_t> dof_marker1)

{
  // Prepare constants and coefficients
  const std::vector<T> constants = dolfinx::fem::pack_constants(*a._form);
  auto coefficients = dolfinx::fem::allocate_coefficient_storage(*a._form);
  dolfinx::fem::pack_coefficients(*a._form, coefficients);
  //pack coefficients of runtime integrals
  auto coeffs_rt = cutfemx::fem::allocate_coefficient_storage(a);
  cutfemx::fem::pack_coefficients(a,coeffs_rt);

  // Assemble
  assemble_matrix(mat_add, a, std::span(constants),
                  dolfinx::fem::make_coefficients_span(coefficients),
                  dolfinx::fem::make_coefficients_span(coeffs_rt), dof_marker0,
                  dof_marker1);
}

} // namespace cutfemx::fem
