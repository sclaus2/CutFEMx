// Copyright (c) 2022-2023 ONERA 
// Authors: Susanne Claus 
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once 

#include <span>
#include <vector>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/DofMap.h>

namespace cutfemx::fem 
{
    // Create a map for a vector of entities of dimension tdim
    // to dofs
    template <std::floating_point U>
    void create_entity_dofmap(std::span<const int32_t> entities, const int &tdim,
                         std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh,
                         std::shared_ptr<const  dolfinx::fem::DofMap> dofmap,
                         std::vector<std::vector<std::int32_t>>& entity_dofs);
}