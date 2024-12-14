// Copyright (c) 2022-2023 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <vector>
#include <cutcells/cell_flags.h>
#include <cutcells/cell_types.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>

#include "../fem/entity_dofmap.h"

namespace cutfemx::level_set
{
   //with level set function and ids of intersected elements obtain
    //entities of ghost penalty facets
    // depending if "phi<0", "phi=0" or "phi>0" is required
   template <dolfinx::scalar T, std::floating_point U= dolfinx::scalar_value_type_t<T>>
   std::vector<int32_t> ghost_penalty_facets(std::shared_ptr<const dolfinx::fem::Function<T>>  level_set, std::span<const int32_t> cut_cell_ids, const std::string& ls_type)
   {
      std::vector<int32_t> facet_ids;
      std::vector<int32_t> all_facets;

      // Connection between cells and facets
      auto mesh = level_set->function_space()->mesh();
      const int tdim = level_set->function_space()->mesh()->topology()->dim();

      // Get adjacencylist from cell to vertices on current processor
      std::shared_ptr<const dolfinx::common::IndexMap>  facet_map = mesh->topology()->index_map(tdim-1);
      if (!facet_map)
      {
        mesh->topology_mutable()->create_entities(tdim - 1);
        facet_map = mesh->topology()->index_map(tdim-1);
      }

      // Facet to cell map
      mesh->topology_mutable()->create_connectivity(tdim, tdim-1);
      auto c_to_f = mesh->topology()->connectivity(tdim, tdim-1);
      assert(c_to_f);

      mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
      auto f_to_c = mesh->topology()->connectivity(tdim - 1, tdim);
      assert(f_to_c);

      //iterate over cut cell entities and obtain all facets
      std::int32_t num_cells = cut_cell_ids.size();

      // Loop over cells
      for (int c = 0; c < num_cells; ++c)
      {
        int cell_index = cut_cell_ids[c];
        //get facets and check each facet
        auto cell_faces = c_to_f->links(cell_index);

        //iterate over faces
        for(int f=0; f<cell_faces.size(); f++)
        {
          auto facet_index = cell_faces[f];
          auto cells = f_to_c->links(f);

          //interior facet
          if(cells.size()==2)
          {
            //check if facet is not already in list
            if(std::find(all_facets.begin(), all_facets.end(), facet_index) == std::end(all_facets))
            {
              all_facets.push_back(facet_index);
            }
          }
        }
      }

      //Get all the dofs for facets
      std::shared_ptr<const dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();
      assert(dofmap);
      std::vector<std::vector<std::int32_t>> entity_dofs(all_facets.size());
      cutfemx::fem::create_entity_dofmap(all_facets, tdim-1,
                         mesh,
                         dofmap,
                         entity_dofs);

      const std::span<const T>& ls_values = level_set->x()->array();

      for (int f = 0; f < all_facets.size(); ++f)
      {
        int facet_index = all_facets[f];
        auto dofs = entity_dofs[f];

        std::vector<T> ls_vals(dofs.size());

        for (std::size_t i = 0; i < dofs.size(); ++i)
        {
            ls_vals[i] = ls_values[dofs[i]];
        }

        cutcells::cell::domain marker_type = cutcells::cell::classify_cell_domain<T>(ls_vals);

        if(marker_type == cutcells::cell::domain::intersected)
        {
          facet_ids.push_back(facet_index);
        }
        if(((marker_type == cutcells::cell::domain::inside)&& (ls_type=="phi<0")))
        {
          facet_ids.push_back(facet_index);
        }
        if(((marker_type == cutcells::cell::domain::outside)&& (ls_type=="phi>0")))
        {
          facet_ids.push_back(facet_index);
        }
      }

      return facet_ids;
   }

  //Standard function to get ghost penalty facets of all cells
  template <dolfinx::scalar T, std::floating_point U= dolfinx::scalar_value_type_t<T>>
  std::vector<int32_t> ghost_penalty_facets(std::shared_ptr<const dolfinx::fem::Function<T>> level_set, const std::string& ls_type)
  {
    assert(level_set->function_space()->mesh());
    std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = level_set->function_space()->mesh();
    int cell_dim = mesh->geometry().dim();
    auto entity_map = mesh->topology()->index_map(cell_dim);

    std::int32_t num_entities = entity_map->size_local()+entity_map->num_ghosts();
    std::vector<int32_t> entities(num_entities);
    std::iota (std::begin(entities), std::end(entities), 0);
    auto intersected_cells = cutfemx::level_set::locate_entities<T>(level_set,entities,cell_dim,"phi=0");

    return ghost_penalty_facets(level_set, std::span(intersected_cells.data(),intersected_cells.size()), ls_type);
  }
}