// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <cutcells/cell_types.h>
#include <dolfinx/mesh/cell_types.h>

namespace cutfemx::mesh
{
  static cutcells::cell::type dolfinx_to_cutcells_cell_type(const dolfinx::mesh::CellType& cell_type)
  {
    cutcells::cell::type cutcells_cell_type;
    switch(cell_type)
    {
        case dolfinx::mesh::CellType::point: cutcells_cell_type = cutcells::cell::type::point;
                                                break;
        case dolfinx::mesh::CellType::interval: cutcells_cell_type = cutcells::cell::type::interval;
                                                break;
        case dolfinx::mesh::CellType::triangle: cutcells_cell_type = cutcells::cell::type::triangle;
                                                break;
        case dolfinx::mesh::CellType::tetrahedron: cutcells_cell_type = cutcells::cell::type::tetrahedron;
                                                break;
    }

    return cutcells_cell_type;
  }

  static dolfinx::mesh::CellType cutcells_to_dolfinx_cell_type(const cutcells::cell::type& cell_type)
  {
    dolfinx::mesh::CellType dolfinx_cell_type;
    switch(cell_type)
    {
        case cutcells::cell::type::point: dolfinx_cell_type = dolfinx::mesh::CellType::point;
                                                break;
        case cutcells::cell::type::interval: dolfinx_cell_type = dolfinx::mesh::CellType::interval;
                                                break;
        case cutcells::cell::type::triangle: dolfinx_cell_type = dolfinx::mesh::CellType::triangle;
                                                break;
        case cutcells::cell::type::tetrahedron: dolfinx_cell_type = dolfinx::mesh::CellType::tetrahedron;
                                                break;
    }

    return dolfinx_cell_type;
  }
}