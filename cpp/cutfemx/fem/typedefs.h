// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once


#include <basix/mdspan.hpp>

#include <dolfinx/common/types.h>
#include <concepts>

namespace cutfemx::fem
{
  /// mdspan typedef
template <typename T, std::size_t ndim>
using mdspand_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;

  template <typename T, std::size_t D>
  using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;

  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const std::int32_t,
    MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

} // namespace cutfemx::fem
