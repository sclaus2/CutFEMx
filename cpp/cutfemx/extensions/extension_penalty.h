// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/common/types.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>

#include "../cut/cut.h"
#include "cell_aggregation.h"

namespace cutfemx::extensions
{

struct ExtensionPair
{
  std::int32_t bad_cell;
  std::int32_t root_cell;
  std::int32_t aggregate_id;
  std::int32_t propagation_depth;
};

template <std::floating_point T>
struct ExtensionQuadrature
{
  std::vector<ExtensionPair> pairs;
  std::vector<std::int32_t> offsets;
  std::vector<T> bad_reference_points;
  std::vector<T> root_reference_points;
  std::vector<T> weights;
  int tdim = 0;
};

template <std::floating_point T>
std::vector<ExtensionPair>
extension_pairs(const CellAggregation<T>& aggregation);

template <std::floating_point T>
ExtensionQuadrature<T> extension_quadrature(
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    int quadrature_degree);

template <dolfinx::scalar S, std::floating_point T>
dolfinx::la::MatrixCSR<S> create_extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation);

template <dolfinx::scalar S, std::floating_point T>
void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>, std::span<const S>)>
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    S beta, int quadrature_degree);

template <dolfinx::scalar S, std::floating_point T>
void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>, std::span<const S>)>
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    std::span<const S> beta_cell_values, int quadrature_degree);

template <std::floating_point T>
void insert_extension_penalty_sparsity(
    dolfinx::la::SparsityPattern& pattern,
    const dolfinx::fem::FunctionSpace<T>& V,
    const CellAggregation<T>& aggregation);

template <dolfinx::scalar S, std::floating_point T>
void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<S>& A, const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    S beta, int quadrature_degree);

template <dolfinx::scalar S, std::floating_point T>
void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<S>& A, const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    std::span<const S> beta_cell_values, int quadrature_degree);

template <dolfinx::scalar S, std::floating_point T>
void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<S>& A, const dolfinx::fem::FunctionSpace<T>& V,
    const ExtensionQuadrature<T>& quadrature, S beta);

template <dolfinx::scalar S, std::floating_point T>
void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<S>& A, const dolfinx::fem::FunctionSpace<T>& V,
    const ExtensionQuadrature<T>& quadrature,
    std::span<const S> beta_cell_values);

template <dolfinx::scalar S, std::floating_point T>
dolfinx::la::MatrixCSR<S> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    S beta, int quadrature_degree);

template <dolfinx::scalar S, std::floating_point T>
dolfinx::la::MatrixCSR<S> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    std::span<const S> beta_cell_values, int quadrature_degree);

} // namespace cutfemx::extensions
