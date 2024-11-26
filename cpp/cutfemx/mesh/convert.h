// Copyright (c) 2022-2023 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include <cutcells/cell_types.h>
#include <dolfinx/mesh/cell_types.h>
#include <basix/cell.h>

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
        case dolfinx::mesh::CellType::quadrilateral: cutcells_cell_type = cutcells::cell::type::quadrilateral;
                                                break;
        case dolfinx::mesh::CellType::hexahedron: cutcells_cell_type = cutcells::cell::type::hexahedron;
                                                break;
        case dolfinx::mesh::CellType::prism: cutcells_cell_type = cutcells::cell::type::prism;
                                                break;
        case dolfinx::mesh::CellType::pyramid: cutcells_cell_type = cutcells::cell::type::pyramid;
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
        case cutcells::cell::type::quadrilateral: dolfinx_cell_type = dolfinx::mesh::CellType::quadrilateral;
                                                break;
        case cutcells::cell::type::hexahedron: dolfinx_cell_type = dolfinx::mesh::CellType::hexahedron;
                                                break;
        case cutcells::cell::type::prism: dolfinx_cell_type = dolfinx::mesh::CellType::prism;
                                                break;
        case cutcells::cell::type::pyramid: dolfinx_cell_type = dolfinx::mesh::CellType::pyramid;
                                                break;
    }

    return dolfinx_cell_type;
  }

  static basix::cell::type cutcells_to_basix_cell_type(const cutcells::cell::type& cell_type)
  {
    basix::cell::type basix_cell_type;
    switch(cell_type)
    {
        case cutcells::cell::type::point: basix_cell_type = basix::cell::type::point;
                                                break;
        case cutcells::cell::type::interval: basix_cell_type = basix::cell::type::interval;
                                                break;
        case cutcells::cell::type::triangle: basix_cell_type = basix::cell::type::triangle;
                                                break;
        case cutcells::cell::type::tetrahedron: basix_cell_type = basix::cell::type::tetrahedron;
                                                break;
        case cutcells::cell::type::quadrilateral: basix_cell_type = basix::cell::type::quadrilateral;
                                                break;
        case cutcells::cell::type::hexahedron: basix_cell_type = basix::cell::type::hexahedron;
                                                break;
        case cutcells::cell::type::prism: basix_cell_type = basix::cell::type::prism;
                                                break;
        case cutcells::cell::type::pyramid: basix_cell_type = basix::cell::type::pyramid;
                                                break;
    }

    return basix_cell_type;
  }
}