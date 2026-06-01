// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include "extension_penalty.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <basix/mdspan.hpp>
#include <basix/quadrature.h>

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>

namespace cutfemx::extensions
{
namespace
{
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

template <std::floating_point T>
void validate_dof_space(const dolfinx::fem::FunctionSpace<T>& V)
{
  const auto element = V.element();
  const auto dofmap = V.dofmap();
  if (!element || !dofmap)
    throw std::invalid_argument("Extension penalty requires a valid function space.");

  if (dofmap->bs() != 1 || dofmap->index_map_bs() != 1)
  {
    throw std::invalid_argument(
        "Extension penalty v1 supports unblocked finite element spaces only. "
        "Use DOLFINx subspaces directly for mixed PDE unknown blocks.");
  }

  if (V.mesh()->topology()->cell_types().size() != 1)
  {
    throw std::invalid_argument(
        "Extension penalty v1 supports meshes with one cell type only.");
  }
}

template <std::floating_point T>
void validate_space(const dolfinx::fem::FunctionSpace<T>& V,
                    const CutData<T>& cut_data)
{
  if (V.mesh().get() != cut_data.mesh_owner.get())
    throw std::invalid_argument("Extension penalty space must live on the cut mesh.");
  validate_dof_space(V);
}

template <std::floating_point T>
std::vector<T> coordinate_dofs(const dolfinx::mesh::Mesh<T>& mesh,
                               std::int32_t cell)
{
  const int gdim = mesh.geometry().dim();
  const auto& cmap = mesh.geometry().cmap();
  const auto x_dofmap = mesh.geometry().dofmap();
  const std::span<const T> x = mesh.geometry().x();
  const std::size_t num_coordinate_dofs = cmap.dim();

  if (x_dofmap.extent(1) != num_coordinate_dofs)
    throw std::runtime_error("Geometry dofmap and coordinate element mismatch.");

  std::vector<T> out(num_coordinate_dofs * static_cast<std::size_t>(gdim));
  for (std::size_t i = 0; i < num_coordinate_dofs; ++i)
  {
    const std::int32_t geometry_dof = x_dofmap(cell, i);
    for (int j = 0; j < gdim; ++j)
      out[i * static_cast<std::size_t>(gdim) + static_cast<std::size_t>(j)]
          = x[3 * static_cast<std::size_t>(geometry_dof)
              + static_cast<std::size_t>(j)];
  }
  return out;
}

template <std::floating_point T>
void pull_back_to_cell(const dolfinx::mesh::Mesh<T>& mesh,
                       std::int32_t cell,
                       std::span<const T> physical_points,
                       std::size_t num_points,
                       std::vector<T>& reference_points)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology()->dim();
  const auto& cmap = mesh.geometry().cmap();
  const std::size_t num_coordinate_dofs = cmap.dim();

  using mdspan2_t = md::mdspan<T, md::dextents<std::size_t, 2>>;
  using cmdspan2_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
  using cmdspan4_t = md::mdspan<const T, md::dextents<std::size_t, 4>>;

  std::vector<T> coordinate_dofs_storage = coordinate_dofs(mesh, cell);
  cmdspan2_t cell_geometry(coordinate_dofs_storage.data(), num_coordinate_dofs,
                           static_cast<std::size_t>(gdim));

  reference_points.assign(num_points * static_cast<std::size_t>(tdim), T(0));
  mdspan2_t X_ref(reference_points.data(), num_points,
                  static_cast<std::size_t>(tdim));
  cmdspan2_t x_phys(physical_points.data(), num_points,
                    static_cast<std::size_t>(gdim));

  if (cmap.is_affine())
  {
    const std::array<std::size_t, 4> phi_shape = cmap.tabulate_shape(1, 1);
    std::vector<T> phi_storage(std::reduce(
        phi_shape.begin(), phi_shape.end(), std::size_t(1),
        std::multiplies<std::size_t>{}));
    std::vector<T> origin(static_cast<std::size_t>(tdim), T(0));
    cmap.tabulate(1, origin, {1, static_cast<std::size_t>(tdim)},
                  std::span<T>(phi_storage.data(), phi_storage.size()));
    cmdspan4_t phi(phi_storage.data(), phi_shape);
    auto dphi = md::submdspan(phi, std::pair(1, tdim + 1), 0,
                              md::full_extent, 0);

    std::vector<T> J_storage(static_cast<std::size_t>(gdim * tdim), T(0));
    std::vector<T> K_storage(static_cast<std::size_t>(tdim * gdim), T(0));
    mdspan2_t J(J_storage.data(), static_cast<std::size_t>(gdim),
                static_cast<std::size_t>(tdim));
    mdspan2_t K(K_storage.data(), static_cast<std::size_t>(tdim),
                static_cast<std::size_t>(gdim));
    dolfinx::fem::CoordinateElement<T>::compute_jacobian(dphi, cell_geometry,
                                                         J);
    dolfinx::fem::CoordinateElement<T>::compute_jacobian_inverse(J, K);

    std::array<T, 3> x0 = {0, 0, 0};
    for (int j = 0; j < gdim; ++j)
      x0[static_cast<std::size_t>(j)] = cell_geometry(0, j);
    dolfinx::fem::CoordinateElement<T>::pull_back_affine(X_ref, K, x0, x_phys);
  }
  else
  {
    cmap.pull_back_nonaffine(X_ref, x_phys, cell_geometry, 1.0e-12, 15);
  }
}

template <std::floating_point T>
void insert_pair_sparsity(dolfinx::la::SparsityPattern& pattern,
                          const dolfinx::fem::DofMap& dofmap,
                          const std::vector<ExtensionPair>& pairs)
{
  for (const ExtensionPair& pair : pairs)
  {
    const std::span<const std::int32_t> bad_dofs
        = dofmap.cell_dofs(pair.bad_cell);
    const std::span<const std::int32_t> root_dofs
        = dofmap.cell_dofs(pair.root_cell);

    pattern.insert(bad_dofs, bad_dofs);
    pattern.insert(bad_dofs, root_dofs);
    pattern.insert(root_dofs, bad_dofs);
    pattern.insert(root_dofs, root_dofs);
  }
}

template <std::floating_point T>
void assemble_l2_pair_block(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>, std::span<const T>)>&
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const ExtensionQuadrature<T>& quadrature,
    std::span<const T> beta_cell_values,
    T beta_scalar, bool use_cell_values);

template <std::floating_point T>
void assemble_l2_pair_block(dolfinx::la::MatrixCSR<T>& A,
                            const dolfinx::fem::FunctionSpace<T>& V,
                            const ExtensionQuadrature<T>& quadrature,
                            std::span<const T> beta_cell_values,
                            T beta_scalar, bool use_cell_values)
{
  assemble_l2_pair_block<T>(A.mat_add_values(), V, quadrature,
                            beta_cell_values, beta_scalar, use_cell_values);
}

template <std::floating_point T>
void assemble_l2_pair_block(
    const std::function<int(std::span<const std::int32_t>,
                            std::span<const std::int32_t>, std::span<const T>)>&
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const ExtensionQuadrature<T>& quadrature,
    std::span<const T> beta_cell_values,
    T beta_scalar, bool use_cell_values)
{
  const auto dofmap = V.dofmap();
  const auto element = V.element();
  const int ndofs = element->space_dimension();
  const int reference_value_size = element->reference_value_size();
  const int element_block_size = element->block_size();
  if (reference_value_size < 1)
    throw std::runtime_error("Extension penalty expects at least one basis value component.");

  std::vector<T> bad_root_block;
  std::vector<T> root_bad_block;
  std::vector<T> bad_bad_block;
  std::vector<T> root_root_block;

  for (std::size_t pair_index = 0; pair_index < quadrature.pairs.size();
       ++pair_index)
  {
    const ExtensionPair& pair = quadrature.pairs[pair_index];
    const std::int32_t q0 = quadrature.offsets[pair_index];
    const std::int32_t q1 = quadrature.offsets[pair_index + 1];
    if (q0 < 0 || q1 < q0
        || static_cast<std::size_t>(q1) > quadrature.weights.size())
      throw std::runtime_error("Extension quadrature offsets mismatch.");

    const std::size_t nq = static_cast<std::size_t>(q1 - q0);
    if (nq == 0)
      continue;

    const T beta = use_cell_values ? beta_cell_values[pair.bad_cell]
                                   : beta_scalar;
    const std::size_t point_offset
        = static_cast<std::size_t>(q0 * quadrature.tdim);
    const std::span<const T> bad_reference_points(
        quadrature.bad_reference_points.data() + point_offset,
        nq * static_cast<std::size_t>(quadrature.tdim));
    const std::span<const T> root_reference_points(
        quadrature.root_reference_points.data() + point_offset,
        nq * static_cast<std::size_t>(quadrature.tdim));

    const auto [bad_basis, bad_shape]
        = element->tabulate(bad_reference_points,
                            {nq, static_cast<std::size_t>(quadrature.tdim)}, 0);
    const auto [root_basis, root_shape]
        = element->tabulate(root_reference_points,
                            {nq, static_cast<std::size_t>(quadrature.tdim)}, 0);
    const bool direct_basis
        = bad_shape[2] == static_cast<std::size_t>(ndofs)
          && root_shape[2] == static_cast<std::size_t>(ndofs)
          && bad_shape[3] == static_cast<std::size_t>(reference_value_size)
          && root_shape[3] == static_cast<std::size_t>(reference_value_size);
    const bool blocked_scalar_basis
        = element_block_size > 1 && bad_shape[3] == 1 && root_shape[3] == 1
          && static_cast<int>(bad_shape[2]) * element_block_size == ndofs
          && static_cast<int>(root_shape[2]) * element_block_size == ndofs;
    const bool blocked_component_basis
        = element_block_size > 1
          && bad_shape[3] == static_cast<std::size_t>(element_block_size)
          && root_shape[3] == static_cast<std::size_t>(element_block_size)
          && static_cast<int>(bad_shape[2]) * element_block_size == ndofs
          && static_cast<int>(root_shape[2]) * element_block_size == ndofs;

    if (bad_shape[0] != 1 || root_shape[0] != 1 || bad_shape[1] != nq
        || root_shape[1] != nq
        || (!direct_basis && !blocked_scalar_basis
            && !blocked_component_basis))
    {
      std::ostringstream msg;
      msg << "Unexpected finite element basis tabulation shape: bad=("
          << bad_shape[0] << ", " << bad_shape[1] << ", " << bad_shape[2]
          << ", " << bad_shape[3] << "), root=(" << root_shape[0] << ", "
          << root_shape[1] << ", " << root_shape[2] << ", "
          << root_shape[3] << "), ndofs=" << ndofs
          << ", reference_value_size=" << reference_value_size;
      throw std::runtime_error(msg.str());
    }

    bad_bad_block.assign(static_cast<std::size_t>(ndofs * ndofs), T(0));
    bad_root_block.assign(static_cast<std::size_t>(ndofs * ndofs), T(0));
    root_bad_block.assign(static_cast<std::size_t>(ndofs * ndofs), T(0));
    root_root_block.assign(static_cast<std::size_t>(ndofs * ndofs), T(0));

    for (std::size_t q = 0; q < nq; ++q)
    {
      const T w = beta * quadrature.weights[static_cast<std::size_t>(q0) + q];
      for (int i = 0; i < ndofs; ++i)
      {
        for (int j = 0; j < ndofs; ++j)
        {
          T bad_bad = 0;
          T bad_root = 0;
          T root_bad = 0;
          T root_root = 0;
          if (blocked_scalar_basis || blocked_component_basis)
          {
            const int component_i = i % element_block_size;
            const int component_j = j % element_block_size;
            if (component_i == component_j)
            {
              const std::size_t scalar_ndofs = bad_shape[2];
              const std::size_t component_stride
                  = blocked_component_basis
                        ? static_cast<std::size_t>(element_block_size)
                        : std::size_t(1);
              const std::size_t bi_index
                  = (q * scalar_ndofs
                     + static_cast<std::size_t>(i / element_block_size))
                        * component_stride
                    + (blocked_component_basis
                           ? static_cast<std::size_t>(component_i)
                           : std::size_t(0));
              const std::size_t bj_index
                  = (q * scalar_ndofs
                     + static_cast<std::size_t>(j / element_block_size))
                        * component_stride
                    + (blocked_component_basis
                           ? static_cast<std::size_t>(component_j)
                           : std::size_t(0));
              const T bi = bad_basis[bi_index];
              const T bj = bad_basis[bj_index];
              const T ri = root_basis[bi_index];
              const T rj = root_basis[bj_index];
              bad_bad = bi * bj;
              bad_root = bi * rj;
              root_bad = ri * bj;
              root_root = ri * rj;
            }
          }
          else
          {
            for (int c = 0; c < reference_value_size; ++c)
            {
              const std::size_t bi_index
                  = ((q * static_cast<std::size_t>(ndofs)
                      + static_cast<std::size_t>(i))
                     * static_cast<std::size_t>(reference_value_size))
                    + static_cast<std::size_t>(c);
              const std::size_t bj_index
                  = ((q * static_cast<std::size_t>(ndofs)
                      + static_cast<std::size_t>(j))
                     * static_cast<std::size_t>(reference_value_size))
                    + static_cast<std::size_t>(c);
              const T bi = bad_basis[bi_index];
              const T bj = bad_basis[bj_index];
              const T ri = root_basis[bi_index];
              const T rj = root_basis[bj_index];
              bad_bad += bi * bj;
              bad_root += bi * rj;
              root_bad += ri * bj;
              root_root += ri * rj;
            }
          }
          const std::size_t index = static_cast<std::size_t>(i * ndofs + j);
          bad_bad_block[index] += w * bad_bad;
          bad_root_block[index] -= w * bad_root;
          root_bad_block[index] -= w * root_bad;
          root_root_block[index] += w * root_root;
        }
      }
    }

    const std::span<const std::int32_t> bad_dofs = dofmap->cell_dofs(pair.bad_cell);
    const std::span<const std::int32_t> root_dofs = dofmap->cell_dofs(pair.root_cell);
    mat_add(bad_dofs, bad_dofs, bad_bad_block);
    mat_add(bad_dofs, root_dofs, bad_root_block);
    mat_add(root_dofs, bad_dofs, root_bad_block);
    mat_add(root_dofs, root_dofs, root_root_block);
  }
}

} // namespace

template <std::floating_point T>
std::vector<ExtensionPair>
extension_pairs(const CellAggregation<T>& aggregation)
{
  std::vector<ExtensionPair> pairs;
  pairs.reserve(aggregation.ill_posed_cells.size());
  for (std::int32_t bad : aggregation.ill_posed_cells)
  {
    if (bad < 0
        || static_cast<std::size_t>(bad) >= aggregation.root_cell.size())
      continue;
    const std::int32_t root = aggregation.root_cell[bad];
    if (root < 0)
      continue;
    pairs.push_back({bad, root,
                     aggregation.aggregate_id[static_cast<std::size_t>(bad)],
                     aggregation.propagation_depth[static_cast<std::size_t>(bad)]});
  }
  return pairs;
}

template <std::floating_point T>
ExtensionQuadrature<T> extension_quadrature(
    const dolfinx::fem::FunctionSpace<T>& V, const CutData<T>& cut_data,
    const CellAggregation<T>& aggregation, int quadrature_degree)
{
  validate_space(V, cut_data);
  if (quadrature_degree < 0)
    throw std::invalid_argument("Extension penalty quadrature degree must be non-negative.");

  const auto mesh = V.mesh();
  const int tdim = mesh->topology()->dim();
  const int gdim = mesh->geometry().dim();
  const auto cell_type = mesh->topology()->cell_types().front();
  const auto [points_ref, weights_ref] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::Default,
      dolfinx::mesh::cell_type_to_basix_type(cell_type),
      basix::polyset::type::standard, quadrature_degree);
  const std::size_t nq = weights_ref.size();

  ExtensionQuadrature<T> out;
  out.pairs = extension_pairs(aggregation);
  out.tdim = tdim;
  out.offsets.reserve(out.pairs.size() + 1);
  out.offsets.push_back(0);
  out.bad_reference_points.reserve(out.pairs.size() * points_ref.size());
  out.root_reference_points.reserve(out.pairs.size() * points_ref.size());
  out.weights.reserve(out.pairs.size() * nq);

  using mdspan2_t = md::mdspan<T, md::dextents<std::size_t, 2>>;
  using cmdspan2_t = md::mdspan<const T, md::dextents<std::size_t, 2>>;
  using cmdspan4_t = md::mdspan<const T, md::dextents<std::size_t, 4>>;

  const auto& cmap = mesh->geometry().cmap();
  const std::array<std::size_t, 4> phi_shape
      = cmap.tabulate_shape(1, nq);
  std::vector<T> phi_storage(std::reduce(
      phi_shape.begin(), phi_shape.end(), std::size_t(1),
      std::multiplies<std::size_t>{}));
  cmap.tabulate(1, points_ref, {nq, static_cast<std::size_t>(tdim)},
                std::span<T>(phi_storage.data(), phi_storage.size()));
  cmdspan4_t phi(phi_storage.data(), phi_shape);
  auto phi0 = md::submdspan(phi, 0, md::full_extent, md::full_extent, 0);

  std::vector<T> physical_points(nq * static_cast<std::size_t>(gdim), T(0));
  std::vector<T> root_points;
  std::vector<T> J_storage(static_cast<std::size_t>(gdim * tdim), T(0));
  std::vector<T> det_scratch(std::max<std::size_t>(
      1, 2 * static_cast<std::size_t>(gdim * tdim)));
  mdspan2_t J(J_storage.data(), static_cast<std::size_t>(gdim),
              static_cast<std::size_t>(tdim));

  for (const ExtensionPair& pair : out.pairs)
  {
    std::vector<T> bad_coordinate_storage
        = coordinate_dofs(*mesh, pair.bad_cell);
    cmdspan2_t bad_coordinate_dofs(
        bad_coordinate_storage.data(), cmap.dim(),
        static_cast<std::size_t>(gdim));
    mdspan2_t x_phys(physical_points.data(), nq,
                     static_cast<std::size_t>(gdim));
    dolfinx::fem::CoordinateElement<T>::push_forward(
        x_phys, bad_coordinate_dofs, phi0);

    pull_back_to_cell(
        *mesh, pair.root_cell,
        std::span<const T>(physical_points.data(), physical_points.size()), nq,
        root_points);

    out.bad_reference_points.insert(out.bad_reference_points.end(),
                                    points_ref.begin(), points_ref.end());
    out.root_reference_points.insert(out.root_reference_points.end(),
                                     root_points.begin(), root_points.end());

    for (std::size_t q = 0; q < nq; ++q)
    {
      std::fill(J_storage.begin(), J_storage.end(), T(0));
      auto dphi = md::submdspan(phi, std::pair(1, tdim + 1), q,
                                md::full_extent, 0);
      dolfinx::fem::CoordinateElement<T>::compute_jacobian(
          dphi, bad_coordinate_dofs, J);
      const T detJ = static_cast<T>(
          dolfinx::fem::CoordinateElement<T>::compute_jacobian_determinant(
              J, det_scratch));
      out.weights.push_back(weights_ref[q] * std::abs(detJ));
    }
    out.offsets.push_back(static_cast<std::int32_t>(out.weights.size()));
  }

  return out;
}

template <std::floating_point T>
dolfinx::la::MatrixCSR<T> create_extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<T>& V, const CutData<T>& cut_data,
    const CellAggregation<T>& aggregation)
{
  validate_space(V, cut_data);
  const auto dofmap = V.dofmap();
  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> maps{
      dofmap->index_map, dofmap->index_map};
  const std::array<int, 2> bs{dofmap->index_map_bs(),
                              dofmap->index_map_bs()};
  dolfinx::la::SparsityPattern pattern(cut_data.mesh_owner->comm(), maps, bs);
  insert_pair_sparsity<T>(pattern, *dofmap, extension_pairs(aggregation));
  pattern.finalize();
  return dolfinx::la::MatrixCSR<T>(pattern);
}

template <std::floating_point T>
void insert_extension_penalty_sparsity(
    dolfinx::la::SparsityPattern& pattern,
    const dolfinx::fem::FunctionSpace<T>& V,
    const CellAggregation<T>& aggregation)
{
  validate_dof_space(V);
  const auto dofmap = V.dofmap();
  insert_pair_sparsity<T>(pattern, *dofmap, extension_pairs(aggregation));
}

template <std::floating_point T>
void assemble_extension_penalty(dolfinx::la::MatrixCSR<T>& A,
                                const dolfinx::fem::FunctionSpace<T>& V,
                                const ExtensionQuadrature<T>& quadrature,
                                T beta)
{
  assemble_l2_pair_block<T>(A, V, quadrature, {}, beta, false);
}

template <std::floating_point T>
void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>, std::span<const T>)>
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const ExtensionQuadrature<T>& quadrature,
    T beta)
{
  assemble_l2_pair_block<T>(mat_add, V, quadrature, {}, beta, false);
}

template <std::floating_point T>
void assemble_extension_penalty(dolfinx::la::MatrixCSR<T>& A,
                                const dolfinx::fem::FunctionSpace<T>& V,
                                const ExtensionQuadrature<T>& quadrature,
                                std::span<const T> beta_cell_values)
{
  if (beta_cell_values.empty())
    throw std::invalid_argument("Cellwise extension penalty coefficient is empty.");
  for (const ExtensionPair& pair : quadrature.pairs)
  {
    if (pair.bad_cell < 0
        || static_cast<std::size_t>(pair.bad_cell) >= beta_cell_values.size())
      throw std::invalid_argument(
          "Cellwise extension penalty coefficient has too few cell values.");
  }
  assemble_l2_pair_block<T>(A, V, quadrature, beta_cell_values, T(0), true);
}

template <std::floating_point T>
void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>, std::span<const T>)>
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const ExtensionQuadrature<T>& quadrature,
    std::span<const T> beta_cell_values)
{
  if (beta_cell_values.empty())
    throw std::invalid_argument("Cellwise extension penalty coefficient is empty.");
  for (const ExtensionPair& pair : quadrature.pairs)
  {
    if (pair.bad_cell < 0
        || static_cast<std::size_t>(pair.bad_cell) >= beta_cell_values.size())
      throw std::invalid_argument(
          "Cellwise extension penalty coefficient has too few cell values.");
  }
  assemble_l2_pair_block<T>(mat_add, V, quadrature, beta_cell_values, T(0),
                            true);
}

template <std::floating_point T>
void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation, T beta,
    int quadrature_degree)
{
  ExtensionQuadrature<T> quadrature
      = extension_quadrature(V, cut_data, aggregation, quadrature_degree);
  assemble_extension_penalty(A, V, quadrature, beta);
}

template <std::floating_point T>
void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>, std::span<const T>)>
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation, T beta,
    int quadrature_degree)
{
  ExtensionQuadrature<T> quadrature
      = extension_quadrature(V, cut_data, aggregation, quadrature_degree);
  assemble_extension_penalty<T>(mat_add, V, quadrature, beta);
}

template <std::floating_point T>
void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<T>& A, const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    std::span<const T> beta_cell_values, int quadrature_degree)
{
  ExtensionQuadrature<T> quadrature
      = extension_quadrature(V, cut_data, aggregation, quadrature_degree);
  assemble_extension_penalty(A, V, quadrature, beta_cell_values);
}

template <std::floating_point T>
void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>, std::span<const T>)>
        mat_add,
    const dolfinx::fem::FunctionSpace<T>& V,
    const CutData<T>& cut_data, const CellAggregation<T>& aggregation,
    std::span<const T> beta_cell_values, int quadrature_degree)
{
  ExtensionQuadrature<T> quadrature
      = extension_quadrature(V, cut_data, aggregation, quadrature_degree);
  assemble_extension_penalty<T>(mat_add, V, quadrature, beta_cell_values);
}

template <std::floating_point T>
dolfinx::la::MatrixCSR<T> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<T>& V, const CutData<T>& cut_data,
    const CellAggregation<T>& aggregation, T beta, int quadrature_degree)
{
  dolfinx::la::MatrixCSR<T> A
      = create_extension_penalty_matrix(V, cut_data, aggregation);
  assemble_extension_penalty(A, V, cut_data, aggregation, beta,
                             quadrature_degree);
  return A;
}

template <std::floating_point T>
dolfinx::la::MatrixCSR<T> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<T>& V, const CutData<T>& cut_data,
    const CellAggregation<T>& aggregation,
    std::span<const T> beta_cell_values, int quadrature_degree)
{
  dolfinx::la::MatrixCSR<T> A
      = create_extension_penalty_matrix(V, cut_data, aggregation);
  assemble_extension_penalty(A, V, cut_data, aggregation, beta_cell_values,
                             quadrature_degree);
  return A;
}

template std::vector<ExtensionPair> extension_pairs(
    const CellAggregation<double>&);
template std::vector<ExtensionPair> extension_pairs(
    const CellAggregation<float>&);

template ExtensionQuadrature<double> extension_quadrature(
    const dolfinx::fem::FunctionSpace<double>&, const CutData<double>&,
    const CellAggregation<double>&, int);
template ExtensionQuadrature<float> extension_quadrature(
    const dolfinx::fem::FunctionSpace<float>&, const CutData<float>&,
    const CellAggregation<float>&, int);

template dolfinx::la::MatrixCSR<double> create_extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<double>&, const CutData<double>&,
    const CellAggregation<double>&);
template dolfinx::la::MatrixCSR<float> create_extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<float>&, const CutData<float>&,
    const CellAggregation<float>&);

template void insert_extension_penalty_sparsity(
    dolfinx::la::SparsityPattern&,
    const dolfinx::fem::FunctionSpace<double>&,
    const CellAggregation<double>&);
template void insert_extension_penalty_sparsity(
    dolfinx::la::SparsityPattern&,
    const dolfinx::fem::FunctionSpace<float>&,
    const CellAggregation<float>&);

template void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>,
                      std::span<const double>)>,
    const dolfinx::fem::FunctionSpace<double>&,
    const CutData<double>&, const CellAggregation<double>&, double, int);
template void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>,
                      std::span<const float>)>,
    const dolfinx::fem::FunctionSpace<float>&,
    const CutData<float>&, const CellAggregation<float>&, float, int);

template void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>,
                      std::span<const double>)>,
    const dolfinx::fem::FunctionSpace<double>&,
    const CutData<double>&, const CellAggregation<double>&,
    std::span<const double>, int);
template void assemble_extension_penalty(
    std::function<int(std::span<const std::int32_t>,
                      std::span<const std::int32_t>,
                      std::span<const float>)>,
    const dolfinx::fem::FunctionSpace<float>&,
    const CutData<float>&, const CellAggregation<float>&,
    std::span<const float>, int);

template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<double>&,
    const dolfinx::fem::FunctionSpace<double>&,
    const ExtensionQuadrature<double>&, double);
template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<float>&,
    const dolfinx::fem::FunctionSpace<float>&,
    const ExtensionQuadrature<float>&, float);

template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<double>&,
    const dolfinx::fem::FunctionSpace<double>&,
    const ExtensionQuadrature<double>&, std::span<const double>);
template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<float>&,
    const dolfinx::fem::FunctionSpace<float>&,
    const ExtensionQuadrature<float>&, std::span<const float>);

template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<double>&,
    const dolfinx::fem::FunctionSpace<double>&, const CutData<double>&,
    const CellAggregation<double>&, double, int);
template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<float>&,
    const dolfinx::fem::FunctionSpace<float>&, const CutData<float>&,
    const CellAggregation<float>&, float, int);

template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<double>&,
    const dolfinx::fem::FunctionSpace<double>&, const CutData<double>&,
    const CellAggregation<double>&, std::span<const double>, int);
template void assemble_extension_penalty(
    dolfinx::la::MatrixCSR<float>&,
    const dolfinx::fem::FunctionSpace<float>&, const CutData<float>&,
    const CellAggregation<float>&, std::span<const float>, int);

template dolfinx::la::MatrixCSR<double> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<double>&, const CutData<double>&,
    const CellAggregation<double>&, double, int);
template dolfinx::la::MatrixCSR<float> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<float>&, const CutData<float>&,
    const CellAggregation<float>&, float, int);

template dolfinx::la::MatrixCSR<double> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<double>&, const CutData<double>&,
    const CellAggregation<double>&, std::span<const double>, int);
template dolfinx::la::MatrixCSR<float> extension_penalty_matrix(
    const dolfinx::fem::FunctionSpace<float>&, const CutData<float>&,
    const CellAggregation<float>&, std::span<const float>, int);

} // namespace cutfemx::extensions
