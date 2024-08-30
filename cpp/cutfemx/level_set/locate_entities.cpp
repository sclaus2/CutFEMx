// Copyright (c) 2022-2023 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include "locate_entities.h"
#include "../fem/entity_dofmap.h"

#include <dolfinx/fem/DofMap.h>
#include <cutcells/cell_flags.h>

namespace cutfemx::level_set
{
  template <std::floating_point T>
  std::vector<int32_t> locate_entities(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                                       const int& dim,
                                       const std::string& ls_part,
                                       bool include_ghost)
  {
    assert(level_set->function_space()->mesh());
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh = level_set->function_space()->mesh();

    auto entity_map = mesh->topology()->index_map(dim);
    if (!entity_map)
    {
      mesh->topology_mutable()->create_entities(dim);
      entity_map = mesh->topology()->index_map(dim);
    }

    std::int32_t num_entities = entity_map->size_local();
    if(include_ghost)
    {
      num_entities += entity_map->num_ghosts();
    }

    std::vector<int32_t> entities(num_entities);
    std::iota (std::begin(entities), std::end(entities), 0);

    const std::vector<int32_t>& marked_entities = locate_entities(level_set, entities,dim,ls_part);
    return marked_entities;
  }

  template <std::floating_point T>
  std::vector<int32_t> locate_entities(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                            std::span<const int32_t> entities,
                            const int& dim,
                            const std::string& ls_part)
  {
    assert(level_set->function_space()->mesh());
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh = level_set->function_space()->mesh();
    assert(level_set->function_space()->dofmap(););
    std::shared_ptr<const  dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();

    auto cell_dim = mesh->topology()->dim();

    // Construct a map from entity to dofs in order of entities vector
    std::vector<std::vector<std::int32_t>> entity_dofs(entities.size());
    fem::create_entity_dofmap(entities, dim, mesh, dofmap, entity_dofs);

    // Get level set values
    const std::span<const T>& ls_values = level_set->x()->array();

    //entities to return
    std::vector<int32_t> marked_entities;

    //count number of cells that are intersected for memory allocation
    for(std::size_t i=0;i<entities.size();i++)
    {
        int entity_index = entities[i];
        std::span<const std::int32_t> dofs = entity_dofs[i];

        // get level set values in cell
        std::vector<T> ls_vals(dofs.size());
        for (std::size_t i = 0; i < dofs.size(); ++i)
            ls_vals[i] = ls_values[dofs[i]];

        cutcells::cell::domain marker_type = cutcells::cell::classify_cell_domain(ls_vals);

        if((marker_type==cutcells::cell::domain::inside) && (ls_part=="phi<0"))
        {
          marked_entities.push_back(entity_index);
        }

        if((marker_type==cutcells::cell::domain::outside) && (ls_part=="phi>0"))
        {
          marked_entities.push_back(entity_index);
        }

        if((marker_type==cutcells::cell::domain::intersected) && (ls_part=="phi=0"))
        {
          marked_entities.push_back(entity_index);
        }

        if((ls_part=="phi<=0") &&
        ((marker_type==cutcells::cell::domain::intersected) || (marker_type==cutcells::cell::domain::inside)))
        {
          marked_entities.push_back(entity_index);
        }

        if((ls_part=="phi>=0") &&
        ((marker_type==cutcells::cell::domain::intersected) || (marker_type==cutcells::cell::domain::outside)))
        {
          marked_entities.push_back(entity_index);
        }
    }

    return marked_entities;
  }

//----------------------------------------------------------------------------------------
  template std::vector<int32_t> locate_entities<double>(std::shared_ptr<const dolfinx::fem::Function<double>> level_set,
                                       const int& dim,
                                       const std::string& ls_part,
                                       bool include_ghost);

  template std::vector<int32_t> locate_entities<double>(std::shared_ptr<const dolfinx::fem::Function<double>> level_set,
                            std::span<const int32_t> entities,
                            const int& dim,
                            const std::string& ls_part);
//----------------------------------------------------------------------------------------


}