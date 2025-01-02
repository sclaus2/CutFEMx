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

    void push_back_facet_topology(std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_to_f,
                                  std::int32_t cell0, std::int32_t cell1,
                                  std::int32_t facet_index,
                                  std::vector<std::int32_t>& facet_topology)
    {
      auto cell_faces1 = c_to_f->links(cell1);
      auto cell_faces0 = c_to_f->links(cell0);

      std::int32_t local_facet_index0 = -1;
      std::int32_t local_facet_index1 = -1;

      const auto it = std::find(cell_faces1.begin(), cell_faces1.end(), facet_index);
      if(it != cell_faces1.end())
      {
        local_facet_index1 = it - cell_faces1.begin();
      }
      else
      {
        throw std::runtime_error("Could not find local facet index for ghost penalty facet");
      }

      const auto it0 = std::find(cell_faces0.begin(), cell_faces0.end(), facet_index);
      if(it0 != cell_faces0.end())
      {
        local_facet_index0 = it0 - cell_faces0.begin();
      }
      else
      {
        throw std::runtime_error("Could not find local facet index for ghost penalty facet");
      }
      facet_topology.push_back(cell0);
      facet_topology.push_back(local_facet_index0);
      facet_topology.push_back(cell1);
      facet_topology.push_back(local_facet_index1);
    }

    template <std::floating_point U>
    void facet_topology(std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh, std::span<const std::int32_t> facet_ids, std::vector<std::int32_t>& facet_topology)
    {
      const int tdim = mesh->topology()->dim();

      // Cell to facet map and Facet to cell map
      mesh->topology_mutable()->create_connectivity(tdim, tdim-1);
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_to_f = mesh->topology()->connectivity(tdim, tdim-1);
      assert(c_to_f);

      mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> f_to_c = mesh->topology()->connectivity(tdim - 1, tdim);
      assert(f_to_c);
      int cnt = 0 ;

      for (auto& f: facet_ids)
      {
        auto cells = f_to_c->links(f);

        if(cells.size()==2)
        {
          auto cell0 = cells[0];
          auto cell1 = cells[1];

          push_back_facet_topology(c_to_f,cell0, cell1, f, facet_topology);
        }
      }
    }
   //with level set function and ids of intersected elements obtain
    //entities of ghost penalty facets
    // depending if "phi<0", "phi=0" or "phi>0" is required
   template <dolfinx::scalar T, std::floating_point U= dolfinx::scalar_value_type_t<T>>
   void ghost_penalty_facets(std::shared_ptr<const dolfinx::fem::Function<T>>  level_set,
                             std::span<const int32_t> cut_cell_ids, const std::string& ls_type,
                             std::vector<int32_t>& facet_ids)
   {
      // Connection between cells and facets
      auto mesh = level_set->function_space()->mesh();
      const int tdim = level_set->function_space()->mesh()->topology()->dim();
      assert(level_set->function_space()->dofmap());
      std::shared_ptr<const dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();

      // Get adjacencylist from cell to vertices on current processor
      std::shared_ptr<const dolfinx::common::IndexMap>  facet_map = mesh->topology()->index_map(tdim-1);
      if (!facet_map)
      {
        mesh->topology_mutable()->create_entities(tdim - 1);
        facet_map = mesh->topology()->index_map(tdim-1);
      }

      // Facet to cell map
      mesh->topology_mutable()->create_connectivity(tdim, tdim-1);
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> c_to_f = mesh->topology()->connectivity(tdim, tdim-1);
      assert(c_to_f);

      mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
      std::shared_ptr<const graph::AdjacencyList<std::int32_t>> f_to_c = mesh->topology()->connectivity(tdim - 1, tdim);
      assert(f_to_c);

      //iterate over cut cell entities and obtain all facets
      std::int32_t num_cells = cut_cell_ids.size();
      const std::span<const T>& ls_values = level_set->x()->array();

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
          auto cells = f_to_c->links(facet_index);

          //interior facet
          if(cells.size()==2)
          {
            auto cell0 = cells[0];
            auto cell1 = cells[1];

            std::span<const std::int32_t> dofs0 = dofmap->cell_dofs(cell0);
            std::span<const std::int32_t> dofs1 = dofmap->cell_dofs(cell1);

            std::vector<T> ls_vals0(dofs0.size());

            for (std::size_t i = 0; i < dofs0.size(); ++i)
            {
                ls_vals0[i] = ls_values[dofs0[i]];
            }

            std::vector<T> ls_vals1(dofs1.size());

            for (std::size_t i = 0; i < dofs1.size(); ++i)
            {
                ls_vals1[i] = ls_values[dofs1[i]];
            }

            cutcells::cell::domain marker_type0 = cutcells::cell::classify_cell_domain<T>(ls_vals0);
            cutcells::cell::domain marker_type1 = cutcells::cell::classify_cell_domain<T>(ls_vals1);

            if(((marker_type0 == cutcells::cell::domain::intersected)&&(marker_type1 == cutcells::cell::domain::intersected)))
            {
                facet_ids.push_back(facet_index);
            }


            if(ls_type == "phi<0")
            {
                if(((marker_type0 == cutcells::cell::domain::inside)&&(marker_type1 == cutcells::cell::domain::intersected))
                || ((marker_type0 == cutcells::cell::domain::intersected)&&(marker_type1 == cutcells::cell::domain::inside)))
                {
                  facet_ids.push_back(facet_index);
                }
            }

            if(ls_type == "phi>0")
            {
                if(((marker_type0 == cutcells::cell::domain::outside)&&(marker_type1 == cutcells::cell::domain::intersected))
                || ((marker_type0 == cutcells::cell::domain::intersected)&&(marker_type1 == cutcells::cell::domain::outside)))
                {
                  facet_ids.push_back(facet_index);
                }
            }
          }
        }
      }
   }

  //Standard function to get ghost penalty facets of all cells
  template <dolfinx::scalar T, std::floating_point U= dolfinx::scalar_value_type_t<T>>
  void ghost_penalty_facets(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                                            const std::string& ls_type,
                                            std::vector<int32_t>& facet_ids)
  {
    assert(level_set->function_space()->mesh());
    std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = level_set->function_space()->mesh();
    int cell_dim = mesh->geometry().dim();
    auto entity_map = mesh->topology()->index_map(cell_dim);

    std::int32_t num_entities = entity_map->size_local();
    std::vector<int32_t> entities(num_entities);
    std::iota (std::begin(entities), std::end(entities), 0);
    auto intersected_cells = cutfemx::level_set::locate_entities<T>(level_set,entities,cell_dim,"phi=0");

    ghost_penalty_facets(level_set, std::span(intersected_cells.data(),intersected_cells.size()), ls_type, facet_ids);
  }
}