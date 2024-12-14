// Copyright (c) 2022 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once 

#include <dolfinx/la/petsc.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>

#include <dolfinx/io/VTKFile.h>

#include <cutcells/cell_flags.h>

#include <span>
#include <string>


namespace cutfemx::fem
{
  //if component = {-1} whole space is taken for dofmap
  template <dolfinx::scalar T, std::floating_point U>
  dolfinx::fem::Function<T,U> deactivate(auto set_fn, std::string deactivate_domain,
                  const std::shared_ptr<dolfinx::fem::Function<T,U>> level_set,
                  const std::shared_ptr<dolfinx::fem::FunctionSpace<U>> V,
                  const std::vector<int>& component,
                  T diagonal = 1.0)
  {
    //Iterate over mesh
    auto mesh = level_set->function_space()->mesh();
    // FIXME: ensure that FunctionSpace and level_set are defined on the same mesh
    // otherwise level set needs to be interpolated onto mesh of V
    // Note the functionspace does not need to be the same but the mesh does
    assert(mesh == V->mesh());

    // Get level set values on mesh and its dofmap
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = level_set->function_space()->dofmap();
    assert(dofmap);
    const std::span<const T>& ls_values = level_set->x()->array();

    const int tdim = mesh->topology()->dim();
    auto cell_map = mesh->topology()->index_map(tdim);
    assert(cell_map);

    std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();

    //Helper characteristic function that is initialized with zero
    // and get set to 1 for cells that should stay activated
    dolfinx::fem::Function<T,U> xi(V);
    auto xi_values = xi.x()->mutable_array();
    //denote all values as active initially
    //this is necessary for mixed spaces
    std::fill(xi_values.begin(),xi_values.end(),1.0);

    std::shared_ptr<const dolfinx::fem::DofMap> dofmap_xi;
    //not asking for subspace
    if(component[0]==-1)
    {
      dofmap_xi = xi.function_space()->dofmap();
    }
    else //extract subspace dofmap to deactivate dofs according to component of functionspace
    {
      dofmap_xi = xi.function_space()->sub(component).dofmap();
    }

    const int bs_dof_xi = dofmap_xi->bs();
    assert(dofmap);

    //deactivate all dofs in component sub-space to re-activate some of them below
    for(std::size_t c=0;c<num_cells;c++)
    {
      std::span<const std::int32_t> dofs_xi = dofmap_xi->cell_dofs(c);

      //this is necessary for mixed spaces
      for (std::size_t i = 0; i < dofs_xi.size(); ++i)
        for (int k = 0; k < bs_dof_xi; ++k)
            xi_values[bs_dof_xi*dofs_xi[i]+k]=0.0;
    }

    for(std::size_t c=0;c<num_cells;c++)
    {
      int cell_index = c;
      std::span<const std::int32_t> dofs = dofmap->cell_dofs(cell_index);
      std::vector<T> ls_vals(dofs.size());

      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        ls_vals[i] = ls_values[dofs[i]];
      }

      cutcells::cell::domain marker_type = cutcells::cell::classify_cell_domain<T>(std::span(ls_vals.data(),ls_vals.size()));

      std::span<const std::int32_t> dofs_xi = dofmap_xi->cell_dofs(cell_index);

      if(deactivate_domain=="phi>0")
      {
        // set xi to 1 for inside or intersected cells
        if((marker_type == cutcells::cell::domain::intersected) || (marker_type == cutcells::cell::domain::inside) )
        {
          for (std::size_t i = 0; i < dofs_xi.size(); ++i)
          {
            for (int k = 0; k < bs_dof_xi; ++k)
            {
              xi_values[bs_dof_xi*dofs_xi[i]+k]=1.0;
            }
          }
        }
      }
      else if(deactivate_domain=="phi<0")
      {
        // set xi to 1 for inside or intersected cells
        if((marker_type == cutcells::cell::domain::intersected) || (marker_type == cutcells::cell::domain::outside) )
        {
          for (std::size_t i = 0; i < dofs_xi.size(); ++i)
            for (int k = 0; k < bs_dof_xi; ++k)
              xi_values[bs_dof_xi*dofs_xi[i]+k]=1.0;
        }
      }
      else if(deactivate_domain=="phi!=0")
      {
        // set xi to 1 for inside or intersected cells
        if((marker_type == cutcells::cell::domain::intersected) )
        {
          for (std::size_t i = 0; i < dofs_xi.size(); ++i)
            for (int k = 0; k < bs_dof_xi; ++k)
              xi_values[bs_dof_xi*dofs_xi[i]+k]=1.0;
        }
      }
      else if(deactivate_domain=="all")
      {
        // set xi to 1 for inside or intersected cells
        for (std::size_t i = 0; i < dofs_xi.size(); ++i)
          for (int k = 0; k < bs_dof_xi; ++k)
            xi_values[bs_dof_xi*dofs_xi[i]+k]=0.0;
      }
      else
      {
        throw std::invalid_argument(
          "Invalid string please choose one of the following: phi!=0, phi<0 or phi>0 as string");
      }
    }

    //Visualize xi
    // std::string filename = "xi.pvd";
    // dolfinx::io::VTKFile file(mesh->comm(), filename, "w");
    // file.write<T>({xi}, 0.0);

    //create rows depending on value of xi in each dof
    std::size_t num_dofs_deactivated = 0;
    for(auto &val: xi_values)
    {
      if(val<1e-8)
      {
        num_dofs_deactivated++;
      }
    }

    std::vector<std::int32_t> rows(num_dofs_deactivated);

    int cnt =0;
    for(std::size_t dof = 0; dof<xi_values.size(); dof++)
    {
      if(xi_values[dof]<1e-8)
      {
        rows[cnt] = dof;
        cnt++;
      }
    }

    // std::cout << "number of deactivated dofs=" << cnt << std::endl;

    // std::cout << "deactivated dofs=";
    // for(auto& entry: rows)
    // {
    //   std::cout << entry << ", ";
    // }
    // std::cout << std::endl;

    dolfinx::fem::set_diagonal(set_fn, rows,diagonal);

    return xi;
  }

}
