// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <concepts>
#include <cstdint>
#include <string_view>
#include <vector>

#include "../cut/cut.h"

namespace cutfemx::extensions
{

enum class RootPolicy
{
  interior_only,
  interior_or_well_cut
};

template <std::floating_point T>
struct CellAggregation
{
  std::vector<std::int32_t> active_cells;
  std::vector<std::int32_t> cut_cells;
  std::vector<std::int32_t> interior_cells;
  std::vector<std::int32_t> well_posed_cells;
  std::vector<std::int32_t> ill_posed_cells;
  std::vector<std::int32_t> root_cell;
  std::vector<std::int32_t> aggregate_id;
  std::vector<std::int32_t> propagation_depth;
  std::vector<std::int32_t> rootless_cells;
  std::vector<T> cut_volume_fraction;
};

RootPolicy root_policy_from_string(std::string_view policy);

template <std::floating_point T>
CellAggregation<T> create_cell_aggregation(
    const cutfemx::CutData<T>& cut_data, std::string_view selector,
    T volume_fraction_threshold,
    RootPolicy root_policy = RootPolicy::interior_or_well_cut,
    int max_iterations = -1, bool allow_rootless = false);

} // namespace cutfemx::extensions
