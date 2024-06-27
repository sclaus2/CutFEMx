// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <vector>
#include <string>

#include <cutcells/cut_mesh.h>

#include <dolfinx/fem/Function.h>

namespace cutfemx::level_set
{
    // cut entities in mesh from level set function given by a list of indices of entities
    // with their cell type and topological dimension
    // the current implementation only works for cells due to dofmap structure
    // assuming as the level set function represents a geometry and not a field that level set function
    // can only be double or float
    template <std::floating_point U>
    cutcells::mesh::CutCells cut_entities(const std::shared_ptr<const dolfinx::fem::Function<U>> level_set,
                std::span<const int32_t> entities, const int &tdim, const std::string& cut_type);

    //version without computation of dof coordinates
    template <std::floating_point U>
    cutcells::mesh::CutCells cut_entities(const std::shared_ptr<const dolfinx::fem::Function<U>> level_set,
                                                std::span<const U> dof_coordinates,
                                                std::span<const int32_t> entities,
                                                const int& tdim,
                                                const std::string& cut_type);
}