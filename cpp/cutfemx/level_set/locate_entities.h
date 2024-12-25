// Copyright (c) 2022-2023 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <vector>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>

namespace cutfemx::level_set
{
  //Return entities depending if the entity is inside, outside or intersected by the level set function
  //ls_part can be phi<0,phi>0 or phi=0
  // dim indicates topological dimensions of entities to be classified
  template <std::floating_point T>
  std::vector<int32_t> locate_entities(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                                       const int& dim,
                                       const std::string& ls_part,
                                       bool include_ghost = false);
  template <std::floating_point T>
  std::vector<int32_t> locate_entities(std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                            std::span<const int32_t> entities,
                            const int& dim,
                            const std::string& ls_part);
}