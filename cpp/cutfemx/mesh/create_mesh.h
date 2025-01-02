// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <cutcells/cut_mesh.h>

#include "convert.h"
#include "cut_mesh.h"

namespace cutfemx::mesh
{
  //return Mesh and parent cell map as this can be used for interpolation
  template <std::floating_point T>
  CutMesh<T> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<T>& cut_cells);

  template <std::floating_point T>
  CutMesh<T> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<T>& cut_cells,
                                  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::int32_t> entities);

}