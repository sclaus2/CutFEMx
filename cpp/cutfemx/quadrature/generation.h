// Copyright (c) 2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include "quadrature.h"
#include "geometry.h"
#include "../mesh/convert.h"
#include "../level_set/locate_entities.h"
#include "../level_set/cut_entities.h"

#include "../fem/entity_dofmap.h"

#include <cutcells/cell_types.h>
#include <cutcells/cut_cell.h>

#include <basix/cell.h>
#include <basix/quadrature.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>

#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>

namespace cutfemx::quadrature
{
    /// mdspan typedef
  template <typename X>
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

  template <typename X>
  using cmdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

  template <typename T, std::size_t D>
  using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;

  template <typename T, std::size_t D>
  using cmdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;

  template <std::floating_point T>
  T compute_detJ(mdspan2_t<T> J, std::span<T> w)
  {
    if (J.extent(0) == J.extent(1))
      return math::det(J);
    else
    {
      assert(w.size() >= 2 * J.extent(0) * J.extent(1));
      mdspan2_t<T> B(w.data(), J.extent(1), J.extent(0));
      mdspan2_t<T> BA(w.data() + J.extent(0) * J.extent(1), B.extent(0),
                   J.extent(1));
      for (std::size_t i = 0; i < B.extent(0); ++i)
        for (std::size_t j = 0; j < B.extent(1); ++j)
          B(i, j) = J(j, i);

      // Zero working memory of BA
      std::fill_n(BA.data_handle(), BA.size(), 0);
      math::dot(B, J, BA);
      return std::sqrt(math::det(BA));
    }
  }

  template <std::floating_point T>
  void runtime_quadrature(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                     std::span<const int32_t> entities,
                     const int& tdim,
                     const std::string& ls_part,
                     const int& order, QuadratureRules<T>& runtime_rules)
  {
    bool triangulate = true;
    assert(level_set->function_space()->dofmap());
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();
    assert(level_set->function_space()->mesh());
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh = level_set->function_space()->mesh();

    //Background mesh cell data
    int gdim = mesh->geometry().dim();
    auto bg_cell_type = mesh->topology()->cell_type();

    //get background mesh geometry data
    // Get geometry data
    const dolfinx::fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const std::size_t num_dofs_g = cmap.dim();
    std::span<const T> x_g = mesh->geometry().x();

    // Evaluate coordinate map basis at reference interpolation points
    using cmdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;
    std::vector<T> coordinate_dofs(3 * x_dofmap.extent(1));

    //locate intersected entities
    dolfinx::mesh::CellType entity_type = dolfinx::mesh::cell_entity_type(bg_cell_type, tdim, 0);

    auto intersected_cells = cutfemx::level_set::locate_entities<T>(level_set,entities,tdim,"phi=0");

    int num_entities = intersected_cells.size();

    std::vector<cutcells::cell::CutCell<T>> cut_cells;
    cutfemx::level_set::cut_reference_entities<T>(level_set, std::span(intersected_cells.data(),intersected_cells.size()), tdim,
                                entity_type,
                                ls_part,
                                triangulate,
                                cut_cells);

    //Generate quadrature rules
    //get type of cells for this cut
    auto cutcells_cell_type = mesh::dolfinx_to_cutcells_cell_type(entity_type);

    std::vector<cutcells::cell::type> cutcells_types = cutcells::cell::cut_cell_types(cutcells_cell_type, ls_part);
    std::unordered_map<cutcells::cell::type, std::size_t> type_id_map;

    std::size_t id = 0;
    int num_types = cutcells_types.size();
    //Note this does not allow for mixed topological dimension in one cut mesh
    int ctdim = cutcells::cell::get_tdim(cutcells_types[0]);

    for(auto& type : cutcells_types)
    {
      type_id_map[type] = id;
      id++;
    }

    std::vector<basix::cell::type> basix_cell_types(num_types);

    for(std::size_t i=0;i<num_types;i++)
    {
      basix_cell_types[i] = cutfemx::mesh::cutcells_to_basix_cell_type(cutcells_types[i]);
    }

    //compute quadrature points in cut cells in reference element of parent
    auto family = basix::element::family::P;
    int degree = 1;
    auto variant = basix::element::lagrange_variant::unset;

    std::vector<std::vector<T>> wts_refs(num_types);
    std::vector<std::vector<T>> phi_b(num_types);
    std::vector<std::array<std::size_t, 4>> e_shapes(num_types);

    for(std::size_t i=0;i<num_types;i++)
    {
      auto [pts_ref_b, wts_ref] = basix::quadrature::make_quadrature<T>(
      basix::quadrature::type::Default, basix_cell_types[i], basix::polyset::type::standard, order);
      wts_refs[i] = wts_ref;

      // Create the lagrange element
      auto basix_element = basix::create_element<T>(family, basix_cell_types[i], degree, variant,
                                    basix::element::dpc_variant::unset, false);

      auto e_shape = basix_element.tabulate_shape(1, wts_ref.size());
      e_shapes[i] = e_shape;

      std::size_t length
          = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
      phi_b[i].resize(length);
      mdspan_t<T, 4> phi(phi_b[i].data(), e_shape);
      mdspan2_t<T> pts_ref(pts_ref_b.data(),wts_ref.size(), ctdim);
      basix_element.tabulate(1, pts_ref, phi); // phi(pts_ref) = pts_ref for linear triangle
    }

    // allocate memory to store one quadrature rule per cut cell
    runtime_rules._quadrature_rules.resize(num_entities);
    runtime_rules._parent_map.resize(num_entities);

    for(std::size_t i=0;i<num_entities;i++)
    {
      int parent_index = intersected_cells[i];
      runtime_rules._parent_map[i] = intersected_cells[i];

      for(std::size_t j=0; j< cut_cells[i]._connectivity.size();j++)
      {
        cutcells::cell::type cut_cell_type = cut_cells[i]._types[j];
        int num_points = wts_refs[type_id_map[cut_cell_type]].size();
        int num_vertices = cut_cells[i]._connectivity[j].size();

        std::vector<T> vertex_coords_b(num_vertices*tdim);
        mdspan2_t<T> vertex_coords(vertex_coords_b.data(),num_vertices, tdim);

        for(std::size_t k =0 ; k < num_vertices; k++)
        {
          for(std::size_t l = 0; l < tdim ; l++)
          {
            int vertex_id = cut_cells[i]._connectivity[j][k];
            vertex_coords(k,l) = cut_cells[i]._vertex_coords[vertex_id*tdim+l];
          }
        }

        //map points from integration rule onto cut part of reference cell of parent
        std::vector<T> points_b(num_points*tdim);
        mdspan2_t<T> points(points_b.data(),num_points,tdim);
        mdspan_t<T, 4> phi(phi_b[type_id_map[cut_cell_type]].data(), e_shapes[type_id_map[cut_cell_type]]);

        for(std::size_t ip=0;ip<num_points;ip++)
        {
          for(std::size_t d=0;d<tdim;d++)
          {
            for(std::size_t v=0;v<phi.extent(2);v++) //shape function index
            {
              points(ip,d) += vertex_coords(v,d)*phi(0, ip, v, 0); //vertex_coords -> coord_dofs of phi
            }
          }
        }

        // STRATEGY: 
        // 1. Map reference quadrature points (of the cut pieces) to physical space using CUT CELL geometry (linear)
        //    -> This gives x_phys and detJ (representing the physical volume of the cut piece)
        // 2. Pull back x_phys to the PARENT element's reference space using cmap (P2, etc.)
        //    -> This gives X_ref for basis function evaluation

        // --- Step 1: Map from Cut Cell Ref -> Physical (Linear) ---
        // Note: The 'phi' variable here (lines 200+) represents the linear shape functions on the cut cell
        // 'vertex_coords_phys' represents the physical coordinates of the cut cell vertices
        
        // We first need the physical coordinates of the cut cell vertices.
        // These must be computed using the PARENT map, because the cut cell vertices are defined in parent ref space.
        
        std::vector<T> vertex_coords_phys_b(num_vertices*gdim);
        mdspan2_t<T> vertex_coords_phys(vertex_coords_phys_b.data(),num_vertices, gdim);

        auto e_shape_phys = cmap.tabulate_shape(0, num_vertices);
        std::size_t length_phys
            = std::accumulate(e_shape_phys.begin(), e_shape_phys.end(), 1, std::multiplies<>{});
        std::vector<T> phi_phys_b(length_phys);
        mdspan_t<T, 4> phi_phys(phi_phys_b.data(), e_shape_phys);
        
        // Tabulate cmap at cut cell vertices (in parent ref space)
        cmap.tabulate(0, std::span(vertex_coords_b.data(),vertex_coords_b.size()), 
                     {static_cast<std::size_t>(num_vertices),static_cast<std::size_t>(tdim)}, 
                     std::span(phi_phys_b.data(),phi_phys_b.size()));

        auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, parent_index, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

        for (std::size_t i = 0; i < x_dofs.size(); ++i)
        {
          std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                      std::next(coordinate_dofs.begin(), 3 * i));
        }

        // Compute physical coordinates of cut cell vertices
        for(std::size_t ip=0;ip<num_vertices;ip++)
        {
          for(std::size_t i=0;i<gdim;i++)
          {
            for(std::size_t v=0;v<x_dofs.size();v++) 
            {
              vertex_coords_phys(ip,i) += coordinate_dofs[v*3+i]*phi_phys(0, ip, v, 0);
            }
          }
        }
        
        // Now map quadrature points to physical space using LINEAR interpolation on the cut cell
        std::vector<T> points_phys_b(num_points*gdim);
        mdspan2_t<T> points_phys(points_phys_b.data(), num_points, gdim);
        
        for(std::size_t ip=0;ip<num_points;ip++)
        {
            for(std::size_t d=0;d<gdim;d++)
            {
                for(std::size_t v=0;v<phi.extent(2);v++) 
                {
                    points_phys(ip,d) += vertex_coords_phys(v,d)*phi(0, ip, v, 0);
                }
            }
        }

        // Update the points to be inserted (unchanged, they are already in parent ref space)
        // points_b = points_ref_b; // REMOVED: Do not modify points!

        // --- Compute Weights using LINEAR Jacobian (from Step 1) ---
        // (This preserves the cut volume)
        std::vector<T> J_b(gdim * ctdim);
        mdspan2_t<T> J(J_b.data(), gdim, ctdim);

        std::vector<T> dphi_b(ctdim * phi.extent(2));
        mdspan2_t<T> dphi(dphi_b.data(), ctdim, phi.extent(2));

        std::vector<T> detJ(num_points);
        std::vector<T> det_scratch(2 * gdim * ctdim);

        std::vector<T> weights(num_points);

        for(int ip=0;ip<num_points;ip++)
        {
            // Compute Linear Jacobian
            std::fill(J_b.begin(), J_b.end(), 0.0);
            for (std::size_t d = 0; d < ctdim; ++d)
                for (std::size_t v = 0; v < phi.extent(2); ++v)
                {
                    dphi(d, v) = phi(d + 1, ip, v, 0);
                }

            math::dot(vertex_coords_phys, dphi, J, true);
            detJ[ip] = std::fabs(compute_detJ(J, std::span(det_scratch.data(),det_scratch.size())));
            weights[ip] = wts_refs[type_id_map[cut_cell_type]][ip]*detJ[ip];

        }

        runtime_rules._quadrature_rules[i]._weights.insert(runtime_rules._quadrature_rules[i]._weights.end(),weights.begin(),weights.end());
        runtime_rules._quadrature_rules[i]._points.insert(runtime_rules._quadrature_rules[i]._points.end(),points_b.begin(),points_b.end());
      }
    }
  }

  template <std::floating_point T>
  void runtime_quadrature(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                     const std::string& ls_part,
                     const int& order, QuadratureRules<T>& runtime_rules)
  {
    assert(level_set->function_space()->mesh());
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh = level_set->function_space()->mesh();
    int cell_dim = mesh->topology()->dim();
    int tdim = cell_dim;

    auto entity_map = mesh->topology()->index_map(tdim);
    if (!entity_map)
    {
      mesh->topology_mutable()->create_entities(tdim);
      entity_map = mesh->topology()->index_map(tdim);
    }

    std::int32_t num_entities = entity_map->size_local();
    std::vector<int32_t> entities(num_entities);
    std::iota (std::begin(entities), std::end(entities), 0);

    runtime_quadrature(level_set, std::span(entities.data(),entities.size()),tdim,ls_part, order, runtime_rules);
  }
}