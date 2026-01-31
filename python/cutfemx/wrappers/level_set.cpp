// Copyright (c) 2022 ONERA
// Authors: Susanne Claus
// This file is part of CutCells
//
// SPDX-License-Identifier:    MIT
#include <array.h>
#include <caster_mpi.h>
#include <numpy_dtype.h>

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <span>

#include <dolfinx/fem/Function.h>
#include <dolfinx/common/types.h>

#include <cutcells/cut_cell.h>
#include <cutcells/cut_mesh.h>

#include <cutfemx/level_set/cut_entities.h>
#include <cutfemx/level_set/locate_entities.h>
#include <cutfemx/level_set/ghost_penalty_facets.h>
#include <cutfemx/level_set/compute_normal.h>
#include <cutfemx/mesh/stl_surface.h>
#include <cutfemx/mesh/stl_reader.h>
#include <cutfemx/mesh/distribute_stl.h>
#include <cutfemx/mesh/cell_triangle_map.h>
#include <cutfemx/level_set/reinit.h>
#include <cutfemx/level_set/sign.h>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_level_set(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  m.def("locate_entities", [](std::shared_ptr<const dolfinx::fem::Function<T,U>> level_set,
                                       const int& dim,
                                       const std::string& ls_part, 
                                       bool include_ghost)
            {
              return dolfinx_wrappers::as_nbarray(cutfemx::level_set::locate_entities(level_set,
                                       dim,
                                      ls_part,
                                        include_ghost));

             }
             , "locate entities");

  m.def("locate_entities_part", [](std::shared_ptr<const dolfinx::fem::Function<T,U>> level_set,
                            nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
                            const int& dim,
                            const std::string& ls_part)
          {
            return dolfinx_wrappers::as_nbarray(cutfemx::level_set::locate_entities(level_set,
                            std::span(entities.data(), entities.size()),
                            dim,
                            ls_part));

            }
            , "locate entities_part");

  m.def("cut_entities", [](const std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                          nb::ndarray<const T, nb::ndim<2>, nb::c_contig> dof_coordinates,
                          nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
                          const int& tdim,
                          const std::string& cut_type)
                          {
                            return cutfemx::level_set::cut_entities(level_set,
                                              std::span(dof_coordinates.data(),dof_coordinates.size()),
                                              std::span(entities.data(),entities.size()),
                                              tdim,
                                              cut_type);
                          }, "cut entities");

  m.def("ghost_penalty_facets", []
  (std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                                            const std::string& ls_type)
  {
    std::vector<int32_t> facet_ids;
    cutfemx::level_set::ghost_penalty_facets(level_set, ls_type, facet_ids);
    return dolfinx_wrappers::as_nbarray(facet_ids);
  }, "ghost penalty facets");

  m.def("interior_facets", []
  (std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
  nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
                                            const std::string& ls_type)
  {
    std::vector<int32_t> facet_ids;
    cutfemx::level_set::interior_facets(level_set, std::span(cells.data(), cells.size()), ls_type, facet_ids);
    return dolfinx_wrappers::as_nbarray(facet_ids);
  }, "interior facets");

  m.def("facet_topology", [](std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh,
                             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facet_ids)
  {
    std::vector<std::int32_t> facet_topology;
    cutfemx::level_set::facet_topology(mesh, std::span(facet_ids.data(),facet_ids.size()), facet_topology);
    return dolfinx_wrappers::as_nbarray(facet_topology);
  }, "obtain facet topology for interior facets");

  m.def(
      "compute_normal",
      [](std::shared_ptr<dolfinx::fem::Function<T>> normal, 
      std::shared_ptr<const dolfinx::fem::Function<T>> level_set, 
      nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        cutfemx::level_set::compute_normal(normal, level_set, std::span(entities.data(), entities.size())); return ;
      });

  m.def("reinitialize", 
      [](std::shared_ptr<dolfinx::fem::Function<T>> phi,
         int max_iter,
         double tol) {
          cutfemx::level_set::FMMOptions opt;
          opt.max_iter = max_iter;
          opt.tol = tol;
          cutfemx::level_set::reinitialize<T>(*phi, opt);
      },
      nb::arg("phi"), nb::arg("max_iter") = 1000, nb::arg("tol") = 1e-10,
      "Reinitialize level set function to signed distance");

}

/// Declare STL/surface mesh bindings
template <typename Real>
void declare_stl_bindings(nb::module_& m, std::string type)
{
  using TriSoup = cutfemx::mesh::TriSoup<Real>;
  using CellTriangleMap = cutfemx::mesh::CellTriangleMap;
  
  // TriSoup class - holds the STL surface mesh
  std::string trisoup_name = "TriSoup_" + type;
  nb::class_<TriSoup>(m, trisoup_name.c_str())
      .def(nb::init<>())
      .def_prop_ro("num_vertices", &TriSoup::nV, "Number of vertices")
      .def_prop_ro("num_triangles", &TriSoup::nT, "Number of triangles")
      .def_prop_ro("vertices", [](const TriSoup& soup) {
          // Return a view into the vertex data (no copy)
          return nb::ndarray<nb::numpy, const Real>(
              soup.X.data(), 
              {static_cast<size_t>(soup.nV()), 3ul},
              nb::handle()
          );
      }, nb::rv_policy::reference_internal, "Vertex coordinates (nV x 3)")
      .def_prop_ro("triangles", [](const TriSoup& soup) {
          // Return a view into triangle connectivity (no copy)
          return nb::ndarray<nb::numpy, const std::int32_t>(
              soup.tri.data(),
              {static_cast<size_t>(soup.nT()), 3ul},
              nb::handle()
          );
      }, nb::rv_policy::reference_internal, "Triangle connectivity (nT x 3)")
      .def("bounding_box", [](const TriSoup& soup) {
          return cutfemx::mesh::bounding_box(soup);
      }, "Compute axis-aligned bounding box, returns (min_corner, max_corner)");

  m.def("compute_stl_bbox", [](const std::string& path) {
    return cutfemx::mesh::compute_stl_bbox<Real>(path);
  }, "Compute axis-aligned bounding box of an STL file without loading it completely.");

  // CellTriangleMap class
  std::string ctm_name = "CellTriangleMap";
  if (type == "float64") {  // Only define once
      nb::class_<CellTriangleMap>(m, ctm_name.c_str())
          .def(nb::init<>())
          .def_prop_ro("num_cells", &CellTriangleMap::num_cells)
          .def("triangles", [](const CellTriangleMap& map, std::int32_t cell) {
              auto tris = map.triangles(cell);
              return dolfinx_wrappers::as_nbarray(std::vector<std::int32_t>(tris.begin(), tris.end()));
          }, "Get triangle indices for a cell");
  }

  // distribute_stl_facets - reads and distributes STL across MPI ranks
  m.def("distribute_stl_facets", 
      [](dolfinx_wrappers::MPICommWrapper comm, 
         std::shared_ptr<dolfinx::mesh::Mesh<Real>> mesh,
         const std::string& stl_path,
         double aabb_padding) {
          cutfemx::mesh::DistributeSTLOptions opt;
          opt.aabb_padding = aabb_padding;
          return std::make_shared<TriSoup>(
              cutfemx::mesh::distribute_stl_facets<Real>(comm.get(), *mesh, stl_path, opt)
          );
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("stl_path"), nb::arg("aabb_padding") = 0.05,
      "Read and distribute STL file across MPI ranks based on mesh partitioning");

  // build_cell_triangle_map
  m.def("build_cell_triangle_map",
      [](std::shared_ptr<const dolfinx::mesh::Mesh<Real>> mesh,
         std::shared_ptr<const TriSoup> soup,
         double aabb_padding) {
          cutfemx::mesh::CellTriangleMapOptions opt;
          opt.aabb_padding = aabb_padding;
          return std::make_shared<CellTriangleMap>(
              cutfemx::mesh::build_cell_triangle_map<Real>(*mesh, *soup, opt)
          );
      },
      nb::arg("mesh"), nb::arg("soup"), nb::arg("aabb_padding") = 0.0,
      "Build cell-to-triangle map for FMM initialization");

  // compute_unsigned_distance - FMM distance computation
  m.def("compute_unsigned_distance",
      [](std::shared_ptr<dolfinx::fem::Function<Real>> dist_func,
         std::shared_ptr<const TriSoup> soup,
         std::shared_ptr<const CellTriangleMap> cell_map,
         int max_iter,
         double tol) {
          cutfemx::level_set::FMMOptions opt;
          opt.max_iter = max_iter;
          opt.tol = tol;
          cutfemx::level_set::compute_unsigned_distance<Real>(*dist_func, *soup, *cell_map, opt);
      },
      nb::arg("dist_func"), nb::arg("soup"), nb::arg("cell_map"),
      nb::arg("max_iter") = 1000, nb::arg("tol") = 1e-10,
      "Compute unsigned distance field using FMM");



  // compute_signed_distance
  m.def("compute_signed_distance",
      [](std::shared_ptr<dolfinx::fem::Function<Real>> dist_func,
         std::shared_ptr<const TriSoup> soup,
         std::shared_ptr<const CellTriangleMap> cell_map,
         cutfemx::level_set::SignMode sign_mode,
         int max_iter,
         double tol) {
          // 1. Compute unsigned distance
          cutfemx::level_set::FMMOptions fmm_opt;
          fmm_opt.max_iter = max_iter;
          fmm_opt.tol = tol;
          
          // We need closest_tri out for Method 1
          std::vector<std::int32_t> closest_tri;
          
          cutfemx::level_set::compute_unsigned_distance<Real>(*dist_func, *soup, *cell_map, fmm_opt, &closest_tri);
          
          // 2. Apply sign
          cutfemx::level_set::SignOptions sign_opt;
          sign_opt.mode = sign_mode;
          
          cutfemx::level_set::apply_sign<Real>(*dist_func, *soup, *cell_map, closest_tri, sign_opt);
      },
      nb::arg("dist_func"), nb::arg("soup"), nb::arg("cell_map"),
      nb::arg("sign_mode") = cutfemx::level_set::SignMode::ComponentAnchor,
      nb::arg("max_iter") = 1000, nb::arg("tol") = 1e-10,
      "Compute signed distance field using FMM and specified sign method");
}

} // namespace

namespace cutfemx_wrappers
{

  void level_set(nb::module_& m)
  {
    // SignMode enum (shared across types)
    nb::enum_<cutfemx::level_set::SignMode>(m, "SignMode")
        .value("LocalNormalBand", cutfemx::level_set::SignMode::LocalNormalBand)
        .value("ComponentAnchor", cutfemx::level_set::SignMode::ComponentAnchor)
        .value("WindingNumber", cutfemx::level_set::SignMode::WindingNumber)
        .export_values();

    declare_level_set<float>(m, "float32");
    declare_level_set<double>(m, "float64");
    
    // STL and FMM bindings
    declare_stl_bindings<float>(m, "float32");
    declare_stl_bindings<double>(m, "float64");
  }
} // end of namespace cutfemx_wrappers