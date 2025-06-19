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
#include <span>

#include <dolfinx/fem/Function.h>
#include <dolfinx/common/types.h>

#include <cutcells/cut_cell.h>
#include <cutcells/cut_mesh.h>

#include <cutfemx/extensions/discrete_extension.h>

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
  m.def(("build_root_to_cut_mapping_" + type).c_str(),
        [](const std::vector<std::int32_t>& badly_cut_parents,
           const std::vector<std::int32_t>& interior_cells,
           const dolfinx::mesh::Mesh<T>& mesh,
           const cutcells::mesh::CutCells<T>& cut_cells)
        {
          std::vector<std::int32_t> root_to_cut_mapping;
          cutfemx::extensions::build_root_to_cut_mapping(
              badly_cut_parents, interior_cells, mesh, cut_cells, root_to_cut_mapping);
          return root_to_cut_mapping;
        },
        "Build mapping from badly cut parent cells to interior cells. "
        "Returns flat vector [badly_cut_0, interior_0, badly_cut_1, interior_1, ...]");

  // AveragingProjector class (commented out - moved to free functions)
  // nb::class_<cutfemx::extensions::AveragingProjector<T>>(m, 
  //     ("AveragingProjector_" + type).c_str())
  //     .def(nb::init<std::shared_ptr<const dolfinx::fem::FunctionSpace<T>>,
  //                   const cutfemx::extensions::DiscreteExtensionOperator<T>&>(),
  //          "Create averaging projector")
  //     .def("apply", &cutfemx::extensions::AveragingProjector<T>::apply,
  //          "Apply averaging projection")
  //     .def("embed", &cutfemx::extensions::AveragingProjector<T>::embed,
  //          "Construct full embedding from interior function to extended function")
  //     .def_property_readonly("embedding_matrix", 
  //          &cutfemx::extensions::AveragingProjector<T>::embedding_matrix,
  //          "Get the full embedding matrix");

  // PatchExtensionSolver class (commented out - moved to free functions)
  // nb::class_<cutfemx::extensions::PatchExtensionSolver<T>>(m, 
  //     ("PatchExtensionSolver_" + type).c_str())
  //     .def(nb::init<std::shared_ptr<const dolfinx::fem::FunctionSpace<T>>,
  //                   const cutcells::mesh::CutCells<T>&,
  //                   const std::vector<std::int32_t>&,
  //                   const std::vector<std::int32_t>&,
  //                   int>(),
  //          "Create patch extension solver", nb::arg("function_space"), 
  //          nb::arg("cut_cells"), nb::arg("badly_cut_parents"), 
  //          nb::arg("well_cut_parents"), nb::arg("patch_radius") = 1)
  //     .def("solve", &cutfemx::extensions::PatchExtensionSolver<T>::solve,
  //          "Solve local patch problems for extension")
  //     .def("build_extension_matrix", 
  //          &cutfemx::extensions::PatchExtensionSolver<T>::build_extension_matrix,
  //          "Build extension matrix by solving all local problems")
  //     .def_property_readonly("extension_matrix", 
  //          &cutfemx::extensions::PatchExtensionSolver<T>::extension_matrix,
  //          "Get precomputed extension matrix");
}

} // namespace

namespace cutfemx_wrappers
{

void extensions(nb::module_& m)
{
  declare_extensions<float>(m, "float32");
  declare_extensions<double>(m, "float64");
}

} // end of namespace cutfemx_wrappers
