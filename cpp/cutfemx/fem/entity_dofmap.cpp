// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#include "entity_dofmap.h"

namespace cutfemx::fem
{
    // Create map from index of entities vector to dofs
    template <std::floating_point U>
    void create_entity_dofmap(std::span<const int32_t> entities, const int &tdim,
                         std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh,
                         std::shared_ptr<const  dolfinx::fem::DofMap> dofmap,
                         std::vector<std::vector<std::int32_t>>& entity_dofs)
    {
      const int cell_dim = mesh->topology()->dim();

      //the entities are cells and therefore we have the dofs readily available in dofmap
      if(cell_dim==tdim)
      {
        for(std::size_t i=0;i<entities.size();i++)
        {
          int entity_index = entities[i];
          std::span<const std::int32_t> dofs = dofmap->cell_dofs(entity_index);
          entity_dofs[i].resize(dofs.size());
          for(std::size_t j=0;j<dofs.size();j++)
          {
            entity_dofs[i][j] = dofs[j];
          }
        }
      }
      //entity is of lower dimension than cell in this case extra steps are necessary to obtain 
      //the dofs for the entities
      else
      {
        // Initialise entity-cell connectivity
        mesh->topology_mutable()->create_entities(tdim);
        mesh->topology_mutable()->create_connectivity(tdim, cell_dim);
        mesh->topology_mutable()->create_connectivity(cell_dim, tdim);
        auto e_to_c = mesh->topology()->connectivity(tdim, cell_dim);
        assert(e_to_c);
        auto c_to_e = mesh->topology()->connectivity(cell_dim, tdim);
        assert(c_to_e);

        // Prepare an element - local dof layout
        // for example, dofs per local facet number
        const int num_cell_entities
            = dolfinx::mesh::cell_num_entities(mesh->topology()->cell_type(), tdim);
        std::vector<std::vector<int>> local_entity_dofs;
        for (int i = 0; i < num_cell_entities; ++i)
        {
          local_entity_dofs.push_back(
              dofmap->element_dof_layout().entity_closure_dofs(tdim, i));
        }

        for(std::size_t i=0;i<entities.size();i++)
        {
          std::size_t entity_index = entities[i];
          // Get first attached cell
          assert(e_to_c->num_links(entity_index) > 0);
          const int cell_index = e_to_c->links(entity_index)[0];

          // Get local index of facet with respect to the cell
          auto entities_d = c_to_e->links(cell_index);
          std::size_t entity_local_index = 0;

          for(std::size_t k=0;k<entities_d.size();k++)
          {
            if(entity_index==entities_d[k])
            {
              entity_local_index = k;
            }
          }

          // Get dof indices for local facet
          std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
          int num_entity_dofs = local_entity_dofs[entity_local_index].size();
          entity_dofs[i].resize(num_entity_dofs);

          for(std::size_t j=0;j<num_entity_dofs;j++)
          {
            std::size_t local_dof_index = local_entity_dofs[entity_local_index][j];
            entity_dofs[i][j] = dofs[local_dof_index];
          }
        }
      }
    }

    //instantiate
    template void create_entity_dofmap<double>(std::span<const int32_t> entities, const int &tdim,
                        std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
                        std::shared_ptr<const  dolfinx::fem::DofMap> dofmap,
                        std::vector<std::vector<std::int32_t>>& entity_dofs);

    template void create_entity_dofmap<float>(std::span<const int32_t> entities, const int &tdim,
                    std::shared_ptr<const dolfinx::mesh::Mesh<float>> mesh,
                    std::shared_ptr<const  dolfinx::fem::DofMap> dofmap,
                    std::vector<std::vector<std::int32_t>>& entity_dofs);
}