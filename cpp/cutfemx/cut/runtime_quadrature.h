// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <basix/mdspan.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cutcells/quadrature.h>

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>

namespace cutfemx
{

struct RuntimeSurfaceProvenance
{
  std::string selector;
  std::int32_t level_set_index = -1;
  std::vector<std::int32_t> cut_cell_ids;
  std::vector<std::int32_t> parent_cell_ids;
  std::vector<std::int32_t> local_zero_entity_ids;
  std::vector<std::int32_t> dimensions;

  bool empty() const noexcept { return cut_cell_ids.empty(); }
  std::size_t size() const noexcept { return cut_cell_ids.size(); }
};

template <std::floating_point T>
struct RuntimeQuadrature
{
  RuntimeQuadrature() = default;

  RuntimeQuadrature(cutcells::quadrature::QuadratureRules<T>&& rules,
                    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh)
      : cutcells_rules(std::move(rules)), _mesh(std::move(mesh))
  {
  }

  RuntimeQuadrature(cutcells::quadrature::QuadratureRules<T>&& rules,
                    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                    RuntimeSurfaceProvenance&& provenance)
      : cutcells_rules(std::move(rules)), surface_provenance(std::move(provenance)),
        _mesh(std::move(mesh))
  {
  }

  RuntimeQuadrature(cutcells::quadrature::QuadratureRules<T>&& rules,
                    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                    std::vector<T>&& physical_points, int gdim)
      : cutcells_rules(std::move(rules)), _mesh(std::move(mesh)),
        _physical_points(std::move(physical_points)),
        _physical_points_ready(true), _physical_gdim(gdim),
        _physical_num_points(cutcells_rules._weights.size())
  {
    if (_physical_points.size()
        != static_cast<std::size_t>(gdim) * cutcells_rules._weights.size())
    {
      throw std::runtime_error(
          "RuntimeQuadrature physical-point cache has inconsistent size");
    }
  }

  RuntimeQuadrature(cutcells::quadrature::QuadratureRules<T>&& rules,
                    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                    std::vector<T>&& physical_points, int gdim,
                    RuntimeSurfaceProvenance&& provenance)
      : cutcells_rules(std::move(rules)), surface_provenance(std::move(provenance)),
        _mesh(std::move(mesh)), _physical_points(std::move(physical_points)),
        _physical_points_ready(true), _physical_gdim(gdim),
        _physical_num_points(cutcells_rules._weights.size())
  {
    if (_physical_points.size()
        != static_cast<std::size_t>(gdim) * cutcells_rules._weights.size())
    {
      throw std::runtime_error(
          "RuntimeQuadrature physical-point cache has inconsistent size");
    }
  }

  int gdim() const
  {
    if (!_mesh)
      return 0;
    return _mesh->geometry().dim();
  }

  const std::vector<T>& physical_points() const
  {
    if (!_mesh)
      throw std::runtime_error("RuntimeQuadrature has no background mesh");

    const std::size_t num_points = cutcells_rules._weights.size();
    const int geometry_dim = _mesh->geometry().dim();
    if (_physical_points_ready && _physical_gdim == geometry_dim
        && _physical_num_points == num_points)
    {
      return _physical_points;
    }

    _physical_points_ready = false;
    _physical_gdim = geometry_dim;
    _physical_num_points = num_points;
    _physical_points.assign(static_cast<std::size_t>(geometry_dim) * num_points,
                            T(0));

    if (num_points == 0)
    {
      _physical_points_ready = true;
      return _physical_points;
    }

    const int reference_dim = cutcells_rules._tdim;
    if (reference_dim <= 0)
      throw std::runtime_error("RuntimeQuadrature has invalid reference dimension");
    if (cutcells_rules._points.size()
        != num_points * static_cast<std::size_t>(reference_dim))
    {
      throw std::runtime_error(
          "RuntimeQuadrature point and weight storage sizes are inconsistent");
    }
    if (cutcells_rules._offset.size()
        != cutcells_rules._parent_map.size() + std::size_t(1))
    {
      throw std::runtime_error(
          "RuntimeQuadrature offset and parent-map sizes are inconsistent");
    }

    const dolfinx::fem::CoordinateElement<T>& cmap
        = _mesh->geometry().cmaps().front();
    const auto x_dofmap = _mesh->geometry().dofmaps().front();
    std::span<const T> x = _mesh->geometry().x();
    const std::size_t num_coordinate_dofs = cmap.dim();
    if (x_dofmap.extent(1) != num_coordinate_dofs)
    {
      throw std::runtime_error(
          "Mesh geometry dofmap does not match coordinate element dimension");
    }

    using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;
    using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

    const std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(0, num_points);
    std::vector<T> phi_storage(std::reduce(
        phi_shape.begin(), phi_shape.end(), std::size_t(1),
        std::multiplies<std::size_t>{}));
    cmap.tabulate(0,
                  std::span<const T>(cutcells_rules._points.data(),
                                     cutcells_rules._points.size()),
                  {num_points, static_cast<std::size_t>(reference_dim)},
                  std::span<T>(phi_storage.data(), phi_storage.size()));
    cmdspan4_t phi(phi_storage.data(), phi_shape);

    std::vector<T> coordinate_dofs_storage(
        num_coordinate_dofs * static_cast<std::size_t>(geometry_dim));
    mdspan2_t coordinate_dofs(coordinate_dofs_storage.data(),
                              num_coordinate_dofs,
                              static_cast<std::size_t>(geometry_dim));

    for (std::size_t rule = 0; rule < cutcells_rules._parent_map.size(); ++rule)
    {
      const auto parent_cell = cutcells_rules._parent_map[rule];
      if (parent_cell < 0
          || static_cast<std::size_t>(parent_cell) >= x_dofmap.extent(0))
      {
        throw std::runtime_error(
            "RuntimeQuadrature parent_map contains an invalid local cell");
      }

      for (std::size_t j = 0; j < num_coordinate_dofs; ++j)
      {
        const std::int32_t geometry_dof = x_dofmap(parent_cell, j);
        for (int k = 0; k < geometry_dim; ++k)
        {
          coordinate_dofs(j, static_cast<std::size_t>(k))
              = x[3 * static_cast<std::size_t>(geometry_dof)
                  + static_cast<std::size_t>(k)];
        }
      }

      const std::int32_t q0 = cutcells_rules._offset[rule];
      const std::int32_t q1 = cutcells_rules._offset[rule + 1];
      if (q0 < 0 || q1 < q0 || static_cast<std::size_t>(q1) > num_points)
        throw std::runtime_error("RuntimeQuadrature offsets are inconsistent");

      for (std::int32_t q = q0; q < q1; ++q)
      {
        const std::size_t point = static_cast<std::size_t>(q);
        for (int k = 0; k < geometry_dim; ++k)
        {
          T value = 0;
          for (std::size_t j = 0; j < num_coordinate_dofs; ++j)
            value += coordinate_dofs(j, static_cast<std::size_t>(k))
                     * phi(0, point, j, 0);

          _physical_points[static_cast<std::size_t>(k) * num_points + point]
              = value;
        }
      }
    }

    _physical_points_ready = true;
    return _physical_points;
  }

  cutcells::quadrature::QuadratureRules<T> cutcells_rules;
  RuntimeSurfaceProvenance surface_provenance;

private:
  std::shared_ptr<const dolfinx::mesh::Mesh<T>> _mesh;
  mutable std::vector<T> _physical_points;
  mutable bool _physical_points_ready = false;
  mutable int _physical_gdim = 0;
  mutable std::size_t _physical_num_points = 0;
};

} // namespace cutfemx
