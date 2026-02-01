// Copyright (c) 2022 ONERA
// Authors: Susanne Claus
// This file is part of CutCells
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

#include <cutfemx/mesh/cut_mesh.h>
#include <cutfemx/mesh/create_mesh.h>
#include <cutfemx/mesh/stl/mesh_adapt_stl.h>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_mesh(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  std::string pyclass_mesh_name = std::string("CutMesh_") + type;
  nb::class_<cutfemx::mesh::CutMesh<T>>(m, pyclass_mesh_name.c_str(),
                                         "CutMesh object")
      .def(nb::init<>())
      .def_prop_ro(
          "bg_mesh",
          [](cutfemx::mesh::CutMesh<T>& self)
          {
            return self._bg_mesh;
          },
          nb::rv_policy::reference_internal,
          " Return background mesh.")
      .def_prop_ro(
          "cut_mesh",
          [](cutfemx::mesh::CutMesh<T>& self)
          {
            return self._cut_mesh;
          },
          nb::rv_policy::reference_internal,
          " Return dolfinx version of cut mesh.")
      .def_prop_ro(
          "parent_index",
          [](cutfemx::mesh::CutMesh<T>& self)
          {
            return nb::ndarray<int32_t, nb::numpy>(
                self._parent_index.data(), {self._parent_index.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal,
          " Return parent index.");

  m.def("create_cut_cells_mesh", [](dolfinx_wrappers::MPICommWrapper comm, cutcells::mesh::CutCells<T>& cut_cells)
            {
              return cutfemx::mesh::create_cut_mesh(comm.get(),cut_cells);
            }
            , "create dolfinx mesh from cut cells alone");

  m.def("create_cut_mesh", [](dolfinx_wrappers::MPICommWrapper comm, cutcells::mesh::CutCells<T>& cut_cells,
                          const dolfinx::mesh::Mesh<T>& mesh,
                          nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
            {
              return cutfemx::mesh::create_cut_mesh(comm.get(),cut_cells,mesh,std::span(entities.data(),entities.size()));
            }
            , "create dolfinx mesh from cut cells and background entities");

  m.def("refinement_edges_from_stl", [](dolfinx_wrappers::MPICommWrapper comm,
                                        dolfinx::mesh::Mesh<T>& mesh,
                                        std::string stl_path,
                                        double padding,
                                        int k_ring)
        {
          // We use the mesh communicator, ignoring passed comm for now as they should match
          // or we could use comm.get() if we updated the C++ API.
          auto edges = cutfemx::mesh::stl::refinement_edges_from_stl(mesh, stl_path, padding, k_ring);
          
          // Zero-copy mechanism for vector -> numpy array? 
          // We need to transfer ownership or copy. Copy is safer.
          size_t size = edges.size();
          int32_t* data = new int32_t[size];
          std::copy(edges.begin(), edges.end(), data);
          nb::capsule owner(data, [](void *p) noexcept { delete[] (int32_t *) p; });
          return nb::ndarray<int32_t, nb::numpy>(data, {size}, owner);
        }, "Get refinement edges from STL");
}

} // namespace

namespace cutfemx_wrappers
{

  void mesh(nb::module_& m)
  {
    declare_mesh<float>(m, "float32");
    declare_mesh<double>(m, "float64");
  }
} // end of namespace cutfemx_wrappers