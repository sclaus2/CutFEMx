// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <vector>

#include <dolfinx/mesh/Mesh.h>

namespace cutfemx::mesh
{
  template <std::floating_point T>
  struct CutMesh
  {
    // original background mesh
    std::shared_ptr<dolfinx::mesh::Mesh<T>> _bg_mesh;
    // cut out part of background mesh
    // in parallel vertices on processor boundaries are doubled
    // as cut mesh is only used for visualisation purposes
    std::shared_ptr<dolfinx::mesh::Mesh<T>> _cut_mesh;

    // map of cells between cut mesh and bg_mesh
    std::vector<int32_t> _parent_index;
    // tag if cell in cut_mesh is same cell geometry in background or not
    // if true -> geometry has changed, i.e. original background cell has been cut
    // this is important for interpolation from background mesh to cut mesh
    std::vector<bool> _is_cut_cell;
  };
}