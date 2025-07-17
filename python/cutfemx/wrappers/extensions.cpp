// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <array.h>
#include <MPICommWrapper.h>
#include <caster_mpi.h>
#include <numpy_dtype.h>

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/pair.h>
#include <span>

#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/common/types.h>

#include <cutcells/cut_cell.h>
#include <cutcells/cut_mesh.h>

#include <cutfemx/extensions/cell_root_mapping.h>
#include <cutfemx/extensions/dof_root_mapping.h>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_extensions(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  m.def("get_badly_cut_parent_indices", 
        [](const cutcells::mesh::CutCells<T>& cut_cells,
           const T mesh_size_h,
           const T threshold_factor)
        {
          return cutfemx::extensions::get_badly_cut_parent_indices(
              cut_cells, mesh_size_h, threshold_factor);
        },
        "Identify badly cut cells based on volume threshold and return their parent indices. "
        "Uses threshold volume < threshold_factor * mesh_size_h. "
        "The CutCells must be triangulated for accurate volume computation.");

  // Build root-to-cut mapping function
  m.def(("build_root_mapping_" + type).c_str(),
        [](const std::vector<std::int32_t>& badly_cut_parents,
           const std::vector<std::int32_t>& interior_cells,
           const dolfinx::mesh::Mesh<T>& mesh,
           const std::vector<std::int32_t>& cut_cell_parents)
        {
          std::unordered_map<std::int32_t, std::int32_t> root_mapping;
          cutfemx::extensions::build_root_mapping(
              badly_cut_parents, interior_cells, mesh, cut_cell_parents, root_mapping);
          return root_mapping;
        },
        "Build mapping from badly cut parent cells to interior cells using two-phase algorithm. "
        "Phase 1: Direct mapping for badly cut cells sharing facets with interior cells. "
        "Phase 2: Propagation to remaining badly cut cells via connections to already-mapped badly cut cells. "
        "Returns dictionary mapping badly cut parent indices to interior cell indices.");

  // Create inverse mapping function
  m.def(("create_root_to_cell_map_" + type).c_str(),
        [](const std::unordered_map<std::int32_t, std::int32_t>& cell_to_root_mapping)
        {
          return cutfemx::extensions::create_root_to_cell_map(cell_to_root_mapping);
        },
        "Create inverse mapping from interior (root) cells to vectors of cut cells. "
        "Input: mapping from cut cells to interior cells. "
        "Output: mapping from interior cells to vectors of cut cells that map to them.");
        
  // Convenience function for int32 mappings
  m.def("create_root_to_cells_mapping",
        [](const std::unordered_map<std::int32_t, std::int32_t>& cell_to_root_mapping)
        {
          return cutfemx::extensions::create_root_to_cell_map(cell_to_root_mapping);
        },
        "Convenience function to create inverse mapping from root cells to cut cells for int32 types. "
        "Input: mapping from cut cells to interior cells. "
        "Output: mapping from interior cells to vectors of cut cells that map to them.");
        
  // Direct function without type suffix for Python convenience
  m.def("create_root_to_cell_map",
        [](const std::unordered_map<std::int32_t, std::int32_t>& cell_to_root_mapping)
        {
          return cutfemx::extensions::create_root_to_cell_map(cell_to_root_mapping);
        },
        "Create inverse mapping from interior (root) cells to vectors of cut cells. "
        "Input: mapping from cut cells to interior cells. "
        "Output: mapping from interior cells to vectors of cut cells that map to them.");
        
  // Build dof-to-cells mapping function
  m.def(("build_dof_to_cells_mapping_" + type).c_str(),
        [](const std::vector<std::int32_t>& cut_cell_parents,
           const std::unordered_map<std::int32_t, std::int32_t>& root,
           const std::shared_ptr<dolfinx::fem::FunctionSpace<T>> function_space)
        {
          return cutfemx::extensions::build_dof_to_cells_mapping(cut_cell_parents, root, *function_space);
        },
        "Build mapping from DOF indices to vectors of cell indices containing that DOF. "
        "Input: cut cell parents, root mapping, and function space. "
        "Output: mapping from DOF indices to vectors of cut cells that contain each DOF.");

  // Build dof-to-root mapping function
  m.def(("build_dof_to_root_mapping_" + type).c_str(),
        [](const std::unordered_map<std::int32_t, std::vector<std::int32_t>>& dof_to_cells_mapping,
           const std::unordered_map<std::int32_t, std::int32_t>& cut_cell_to_root_mapping,
           const std::vector<T>& dof_coords,
           const std::shared_ptr<dolfinx::fem::FunctionSpace<T>> function_space,
           const dolfinx::mesh::Mesh<T>& mesh)
        {
          return cutfemx::extensions::build_dof_to_root_mapping(
              dof_to_cells_mapping, cut_cell_to_root_mapping, dof_coords, *function_space, mesh);
        },
        "Build mapping from DOF indices to closest root cell indices. "
        "For each DOF, finds all possible root cells and selects the one whose midpoint is closest to the DOF coordinate. "
        "Input: dof-to-cells mapping, cut-cell-to-root mapping, dof coordinates, function space, and mesh. "
        "Output: mapping from DOF indices to closest root cell indices.");
    }
} // namespace

namespace cutfemx_wrappers
{

void extensions(nb::module_& m)
{
  declare_extensions<float>(m, "float32");
  declare_extensions<double>(m, "float64");
  
  // Add direct function binding without type suffix for Python convenience
  m.def("build_dof_to_cells_mapping",
        [](const std::vector<std::int32_t>& cut_cell_parents,
           const std::unordered_map<std::int32_t, std::int32_t>& root,
           const std::shared_ptr<dolfinx::fem::FunctionSpace<double>> function_space)
        {
          return cutfemx::extensions::build_dof_to_cells_mapping(cut_cell_parents, root, *function_space);
        },
        "Build mapping from DOF indices to vectors of cell indices containing that DOF. "
        "Input: cut cell parents, root mapping, and function space. "
        "Output: mapping from DOF indices to vectors of cut cells that contain each DOF.");

  m.def("build_dof_to_root_mapping",
        [](const std::unordered_map<std::int32_t, std::vector<std::int32_t>>& dof_to_cells_mapping,
           const std::unordered_map<std::int32_t, std::int32_t>& cut_cell_to_root_mapping,
           const std::vector<double>& dof_coords,
           const std::shared_ptr<dolfinx::fem::FunctionSpace<double>> function_space,
           const dolfinx::mesh::Mesh<double>& mesh)
        {
          return cutfemx::extensions::build_dof_to_root_mapping(
              dof_to_cells_mapping, cut_cell_to_root_mapping, dof_coords, *function_space, mesh);
        },
        "Build mapping from DOF indices to closest root cell indices. "
        "For each DOF, finds all possible root cells and selects the one whose midpoint is closest to the DOF coordinate. "
        "Input: dof-to-cells mapping, cut-cell-to-root mapping, dof coordinates, function space, and mesh. "
        "Output: mapping from DOF indices to closest root cell indices.");
}

} // end of namespace cutfemx_wrappers
