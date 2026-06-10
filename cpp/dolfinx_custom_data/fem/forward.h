// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/mdspan.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/traits.h>
#include <type_traits>

namespace dolfinx::common
{
class IndexMap;
} // namespace dolfinx::common

namespace dolfinx::graph
{
template <typename LinkData, typename NodeData>
class AdjacencyList;
} // namespace dolfinx::graph

namespace dolfinx::la
{
class SparsityPattern;
} // namespace dolfinx::la

namespace dolfinx::mesh
{
class EntityMap;
enum class CellType : std::int8_t;
class Topology;
template <std::floating_point T>
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::fem
{
class DofMap;
class ElementDofLayout;
enum class doftransform : std::uint8_t;

template <dolfinx::scalar T>
class Constant;

template <std::floating_point T>
class CoordinateElement;

template <dolfinx::scalar T, std::floating_point U>
class DirichletBC;

template <dolfinx::scalar T, std::floating_point U>
class Expression;

template <std::floating_point T>
class FiniteElement;

template <dolfinx::scalar T, std::floating_point U>
class Function;

template <std::floating_point T>
class FunctionSpace;

namespace sparsitybuild
{
}
} // namespace dolfinx::fem

namespace dolfinx_custom_data::fem
{
namespace common = dolfinx::common;
namespace graph = dolfinx::graph;
namespace la = dolfinx::la;
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;
namespace mesh = dolfinx::mesh;
namespace sparsitybuild = dolfinx::fem::sparsitybuild;

using dolfinx::scalar;
using dolfinx::scalar_value_t;
using dolfinx::fem::doftransform;

using DofMap = dolfinx::fem::DofMap;
using ElementDofLayout = dolfinx::fem::ElementDofLayout;

template <dolfinx::scalar T>
using Constant = dolfinx::fem::Constant<T>;

template <std::floating_point T>
using CoordinateElement = dolfinx::fem::CoordinateElement<T>;

template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
using DirichletBC = dolfinx::fem::DirichletBC<T, U>;

template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
using Expression = dolfinx::fem::Expression<T, U>;

template <std::floating_point T>
using FiniteElement = dolfinx::fem::FiniteElement<T>;

template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_t<T>>
using Function = dolfinx::fem::Function<T, U>;

template <std::floating_point T>
using FunctionSpace = dolfinx::fem::FunctionSpace<T>;

template <class U, class T>
concept DofTransformKernel = dolfinx::fem::DofTransformKernel<U, T>;

template <class Kernel, class T, class U = scalar_value_t<T>>
concept FEkernel
    = std::is_invocable_v<Kernel, T*, const T*, const T*, const U*,
                          const int*, const std::uint8_t*, void*>;

template <class T>
concept MDSpan2 = dolfinx::fem::MDSpan2<T>;
} // namespace dolfinx_custom_data::fem
