// Copyright (c) 2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

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

#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>

namespace cutfemx::quadrature
{
    /// mdspan typedef
  template <typename X>
  using mdspan2_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      X, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>;

  template <typename T, std::size_t D>
  using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;



  template <std::floating_point T>
  void runtime_quadrature(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
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
    int cell_dim = mesh->topology()->dim();
    auto bg_cell_type = mesh->topology()->cell_type();

    //locate intersected entities
    int tdim = cell_dim;
    auto entity_map = mesh->topology()->index_map(tdim);
    if (!entity_map)
    {
      mesh->topology_mutable()->create_entities(tdim);
      entity_map = mesh->topology()->index_map(tdim);
    }

    std::int32_t num_entities = entity_map->size_local()+entity_map->num_ghosts();
    std::vector<int32_t> entities(num_entities);
    std::iota (std::begin(entities), std::end(entities), 0);

    auto intersected_cells = cutfemx::level_set::locate_entities<T>(level_set,std::span(entities.data(),entities.size()),tdim,"phi=0");

    num_entities = intersected_cells.size();
    std::vector<cutcells::cell::CutCell<T>> cut_cells;
    cutfemx::level_set::cut_reference_entities<T>(level_set, std::span(intersected_cells.data(),intersected_cells.size()), tdim,
                                bg_cell_type,
                                ls_part,
                                triangulate,
                                cut_cells);

    //Generate quadrature rules
    dolfinx::mesh::CellType entity_type = dolfinx::mesh::cell_entity_type(bg_cell_type, tdim, 0);
    auto cutcells_cell_type = mesh::dolfinx_to_cutcells_cell_type(entity_type);

    //get type of cells for this cut
    std::vector<cutcells::cell::type> cutcells_types = cutcells::cell::cut_cell_types(cutcells_cell_type, ls_part);
    std::unordered_map<cutcells::cell::type, std::size_t> type_id_map;
    std::size_t id = 0;
    int num_types = cutcells_types.size();

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
      basix::FiniteElement basix_element
          = basix::create_element<T>(family, basix_cell_types[i], degree, variant,
                                    basix::element::dpc_variant::unset, false);

      auto e_shape = basix_element.tabulate_shape(0, wts_ref.size());
      e_shapes[i] = e_shape;

      std::size_t length
          = std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
      phi_b[i].resize(length);
      mdspan_t<T, 4> phi(phi_b[i].data(), e_shape);
      mdspan2_t<T> pts_ref(pts_ref_b.data(),wts_ref.size(), tdim);
      basix_element.tabulate(0, pts_ref, phi); // phi(pts_ref) = pts_ref for linear triangle
    }

    // allocate memory to store one quadrature rule per cut cell
    runtime_rules._quadrature_rules.resize(num_entities);
    runtime_rules._parent_map.resize(num_entities);

    for(std::size_t i=0;i<num_entities;i++)
    {
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

        runtime_rules._quadrature_rules[i]._weights.insert(runtime_rules._quadrature_rules[i]._weights.end(),wts_refs[type_id_map[cut_cell_type]].begin(),wts_refs[type_id_map[cut_cell_type]].end());
        runtime_rules._quadrature_rules[i]._points.insert(runtime_rules._quadrature_rules[i]._points.end(),points_b.begin(),points_b.end());
      }
    }
  }
}