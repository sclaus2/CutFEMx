// Copyright (c) 2022 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include "cut_entities.h"
#include "../fem/entity_dofmap.h"
#include "../mesh/convert.h"

#include <cutcells/cut_cell.h>
#include <cutcells/cell_flags.h>
#include <cutcells/cell_types.h>

#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/DofMap.h>

#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/utils.h>

namespace cutfemx::level_set
{
    template <std::floating_point U>
    cutcells::mesh::CutCells cut_entities(const std::shared_ptr<const dolfinx::fem::Function<U>> level_set,
                                          std::span<const U> dof_coordinates,
                                          std::span<const int32_t> entities,
                                          const int& tdim,
                                          const std::string& cut_type)
    {
        cutcells::mesh::CutCells cut_mesh;

        auto basix_element = level_set->function_space()->element()->basix_element();
        assert(basix_element);
        int degree = basix_element.degree();
        basix::element::family family = basix_element.family();
        if(family!=basix::element::family::P)
          throw std::invalid_argument( "Only Lagrange elements are supported for cutting algorithm" );

        assert(level_set->function_space()->dofmap(););
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();
        assert(level_set->function_space()->mesh());
        std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = level_set->function_space()->mesh();
        int gdim = mesh->geometry().dim();
        int cell_dim = mesh->topology()->dim();
        auto cell_type = mesh->topology()->cell_type();

        // Construct a map from entity to dofs in order of entities vector
        std::vector<std::vector<std::int32_t>> entity_dofs(entities.size());
        fem::create_entity_dofmap<U>(entities, tdim,mesh, dofmap, entity_dofs);

        mesh->topology_mutable()->create_connectivity(tdim, cell_dim);
        auto e_to_c = mesh->topology()->connectivity(tdim, cell_dim);
        assert(e_to_c);

        // //Print entity dofs
        // std::cout << "entity_dofs={";
        // for(std::size_t i=0;i<entities.size();i++)
        // {
        //   std::cout << entities[i] << ": ";
        //   for(std::size_t j=0;j<entity_dofs[i].size();j++)
        //   {
        //     std::cout << entity_dofs[i][j] << ",";
        //   }
        //   std::cout << std::endl;
        // }
        // std::cout << "}" << std::endl;

        //NOTE: this only works if cell has one type of facets
        // needs to be changed for prisms and pyramids
        dolfinx::mesh::CellType entity_type = dolfinx::mesh::cell_entity_type(cell_type, tdim, 0);
        auto cutcells_cell_type = mesh::dolfinx_to_cutcells_cell_type(entity_type);

        // Get level set values
        const std::span<const U>& ls_values = level_set->x()->array();

        int num_entities = entities.size();
        int num_cut_cells = 0;

        //count number of cells that are intersected for memory allocation
        for(std::size_t i=0;i<num_entities;i++)
        {
            int entity_index = entities[i];
            std::span<const std::int32_t> dofs = entity_dofs[i];

            // get level set values in cell
            std::vector<U> ls_vals(dofs.size());
            for (std::size_t i = 0; i < dofs.size(); ++i)
                ls_vals[i] = ls_values[dofs[i]];

            if(cutcells::cell::is_intersected(ls_vals))
            {
                num_cut_cells++;
            }
        }

        //Pre-allocate number of cut cells to be computed
        cut_mesh._cut_cells.resize(num_cut_cells);
        cut_mesh._parent_map.resize(num_cut_cells);

        int cut_cell_id = 0;
        int total_number_vertices = 0 ;

        for(std::size_t i=0;i<num_entities;i++)
        {
          int entity_index = entities[i];
          std::span<const std::int32_t> dofs = entity_dofs[i];

          // get level set values in cell
          std::vector<U> ls_vals(dofs.size());
          for (std::size_t i = 0; i < dofs.size(); ++i)
              ls_vals[i] = ls_values[dofs[i]];

          if(cutcells::cell::is_intersected(ls_vals))
          {
              std::vector<U> vertex_coordinates(dofs.size()*gdim);

              for (std::size_t i = 0; i < dofs.size(); ++i)
              {
                  for(std::size_t j=0;j<gdim;j++)
                  {
                    vertex_coordinates[i*gdim+j] = dof_coordinates[dofs[i]*3+j];
                  }
              }

              cutcells::cell::CutCell cut_cell;

              switch(degree)
              {
                case 1: {
                            cut(cutcells_cell_type,vertex_coordinates,gdim,ls_vals,cut_type,cut_cell,true);
                            break;
                        }
                case 2: {
                            cut_cell = higher_order_cut(cutcells_cell_type,vertex_coordinates,gdim,ls_vals,cut_type,true);
                            break;
                        }
                default:{
                          throw std::invalid_argument( "Cut routines are only implemented for linear and quadratic elements" );
                        }
              }

              cut_mesh._cut_cells[cut_cell_id] = cut_cell;
              int cell_index = entity_index;
              //FIXME: for cuts of edges/facets (lower tdim) parent element should still be a cell
              //to coincide with custom integral measure (which is assumed to be part of a cell)
              if(tdim<cell_dim)
              {
                  assert(e_to_c->num_links(entity_index) > 0);
                  //get first cell connected
                  cell_index = e_to_c->links(entity_index)[0];
              }
              cut_mesh._parent_map[cut_cell_id] = cell_index;
              int num_vertices = cut_cell._vertex_coords.size()/gdim;
              total_number_vertices += num_vertices;

              for(auto &type : cut_cell._types)
              {
                  if(std::find(cut_mesh._types.begin(), cut_mesh._types.end(), type) == cut_mesh._types.end())
                  {
                      cut_mesh._types.push_back(type);
                  }
              }

              cut_cell_id++;
          }
        }

        cut_mesh._num_vertices = total_number_vertices;

        return cut_mesh;
    }

    template <std::floating_point U>
    cutcells::mesh::CutCells cut_entities(const std::shared_ptr<const dolfinx::fem::Function<U>> level_set,
                std::span<const int32_t> entities, const int &tdim, const std::string& cut_type)
    {
      const std::vector<U> dof_coordinates = level_set->function_space()->tabulate_dof_coordinates(false);
      const cutcells::mesh::CutCells& cut_mesh = cut_entities<U>(level_set, dof_coordinates, entities, tdim, cut_type);

      return cut_mesh;
    }

//----------------------------------------------------------------------------------------
    template cutcells::mesh::CutCells cut_entities<double>(const std::shared_ptr<const dolfinx::fem::Function<double>> level_set,
                                          std::span<const double> dof_coordinates,
                                          std::span<const int32_t> entities,
                                          const int& tdim,
                                          const std::string& cut_type);
    template cutcells::mesh::CutCells cut_entities<double>(const std::shared_ptr<const dolfinx::fem::Function<double>> level_set,
                std::span<const int32_t> entities, const int &tdim, const std::string& cut_type);
//----------------------------------------------------------------------------------------

}