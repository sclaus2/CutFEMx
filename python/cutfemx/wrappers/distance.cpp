// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <MPICommWrapper.h>
#include <array.h>
#include <caster_mpi.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include <cutfemx/distance/fast_iterative.h>
#include <cutfemx/distance/normal_extension.h>
#include <cutfemx/distance/reinitialize.h>
#include <cutfemx/distance/sign.h>
#include <cutfemx/distance/stl/cell_triangle_map.h>
#include <cutfemx/distance/stl/distribute.h>
#include <cutfemx/distance/stl/mesh_adapt.h>
#include <cutfemx/distance/stl/reader.h>
#include <cutfemx/distance/stl/surface.h>
#include <cutfemx/mesh/cut_mesh.h>

namespace nb = nanobind;

namespace
{
template <typename Real>
void declare_distance(nb::module_& m, std::string type)
{
  using TriSoup = cutfemx::distance::TriSoup<Real>;
  using CellTriangleMap = cutfemx::distance::CellTriangleMap;

  const std::string trisoup_name = "TriSoup_" + type;
  nb::class_<TriSoup>(m, trisoup_name.c_str(), "Distributed STL triangle soup")
      .def(nb::init<>())
      .def_prop_ro("num_vertices", &TriSoup::nV)
      .def_prop_ro("num_triangles", &TriSoup::nT)
      .def_prop_ro(
          "vertices",
          [](const TriSoup& soup)
          {
            return nb::ndarray<nb::numpy, const Real>(
                soup.X.data(), {static_cast<std::size_t>(soup.nV()), 3ul},
                nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "triangles",
          [](const TriSoup& soup)
          {
            return nb::ndarray<nb::numpy, const std::int32_t>(
                soup.tri.data(),
                {static_cast<std::size_t>(soup.nT()), 3ul}, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def("bounding_box",
           [](const TriSoup& soup)
           { return cutfemx::distance::bounding_box(soup); });

  if (type == "float64")
  {
    nb::class_<CellTriangleMap>(m, "CellTriangleMap",
                                "CSR map from local cells to local STL triangles")
        .def(nb::init<>())
        .def_prop_ro("num_cells", &CellTriangleMap::num_cells)
        .def("triangles",
             [](const CellTriangleMap& map, std::int32_t cell)
             {
               auto tris = map.triangles(cell);
               return dolfinx_wrappers::as_nbarray(
                   std::vector<std::int32_t>(tris.begin(), tris.end()));
             });
  }

  m.def("compute_stl_bbox",
        [](const std::string& path)
        { return cutfemx::distance::compute_stl_bbox<Real>(path); },
        nb::arg("path"));

  m.def(
      "distribute_stl",
      [](dolfinx_wrappers::MPICommWrapper comm,
         std::shared_ptr<dolfinx::mesh::Mesh<Real>> mesh,
         const std::string& stl_path, double aabb_padding)
      {
        cutfemx::distance::DistributeSTLOptions opt;
        opt.aabb_padding = aabb_padding;
        return std::make_shared<TriSoup>(
            cutfemx::distance::distribute_stl_facets<Real>(
                comm.get(), *mesh, stl_path, opt));
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("stl_path"),
      nb::arg("aabb_padding") = 0.05);

  m.def(
      "build_cell_triangle_map",
      [](std::shared_ptr<const dolfinx::mesh::Mesh<Real>> mesh,
         std::shared_ptr<const TriSoup> soup, double aabb_padding)
      {
        cutfemx::distance::CellTriangleMapOptions opt;
        opt.aabb_padding = aabb_padding;
        return std::make_shared<CellTriangleMap>(
            cutfemx::distance::build_cell_triangle_map<Real>(*mesh, *soup, opt));
      },
      nb::arg("mesh"), nb::arg("soup"), nb::arg("aabb_padding") = 0.0);

  m.def(
      "compute_unsigned_distance",
      [](std::shared_ptr<dolfinx::fem::Function<Real>> dist_func,
         std::shared_ptr<const TriSoup> soup,
         std::shared_ptr<const CellTriangleMap> cell_map, int max_iter,
         double tol)
      {
        cutfemx::distance::FMMOptions opt;
        opt.max_iter = max_iter;
        opt.tol = tol;
        cutfemx::distance::compute_unsigned_distance<Real>(
            *dist_func, *soup, *cell_map, opt);
      },
      nb::arg("dist_func"), nb::arg("soup"), nb::arg("cell_map"),
      nb::arg("max_iter") = 1000, nb::arg("tol") = 1e-10);

  m.def(
      "compute_signed_distance",
      [](std::shared_ptr<dolfinx::fem::Function<Real>> dist_func,
         std::shared_ptr<const TriSoup> soup,
         std::shared_ptr<const CellTriangleMap> cell_map,
         cutfemx::distance::SignMode sign_mode, int max_iter, double tol)
      {
        cutfemx::distance::FMMOptions fmm_opt;
        fmm_opt.max_iter = max_iter;
        fmm_opt.tol = tol;

        std::vector<std::int32_t> closest_tri;
        cutfemx::distance::compute_unsigned_distance<Real>(
            *dist_func, *soup, *cell_map, fmm_opt, &closest_tri);

        cutfemx::distance::SignOptions sign_opt;
        sign_opt.mode = sign_mode;
        cutfemx::distance::apply_sign<Real>(
            *dist_func, *soup, *cell_map, closest_tri, sign_opt);
      },
      nb::arg("dist_func"), nb::arg("soup"), nb::arg("cell_map"),
      nb::arg("sign_mode") = cutfemx::distance::SignMode::ComponentAnchor,
      nb::arg("max_iter") = 1000, nb::arg("tol") = 1e-10);

  m.def(
      "reinitialize",
      [](std::shared_ptr<dolfinx::fem::Function<Real>> phi, int max_iter,
         double tol)
      {
        cutfemx::distance::FMMOptions opt;
        opt.max_iter = max_iter;
        opt.tol = tol;
        cutfemx::distance::reinitialize<Real>(*phi, opt);
      },
      nb::arg("phi"), nb::arg("max_iter") = 1000, nb::arg("tol") = 1e-10);

  m.def(
      "reinitialize_from_facets",
      [](std::shared_ptr<dolfinx::fem::Function<Real>> phi,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facets,
         int max_iter, double tol)
      {
        cutfemx::distance::FMMOptions opt;
        opt.max_iter = max_iter;
        opt.tol = tol;
        cutfemx::distance::reinitialize_from_facets<Real>(
            *phi, std::span<const std::int32_t>(facets.data(), facets.size()),
            opt);
      },
      nb::arg("phi"), nb::arg("facets"), nb::arg("max_iter") = 1000,
      nb::arg("tol") = 1e-10);

  m.def(
      "extend_normal_velocity",
      [](std::shared_ptr<const dolfinx::fem::Function<Real>> phi,
         std::shared_ptr<const dolfinx::fem::Function<Real>> interface_speed,
         const cutfemx::mesh::CutMesh<Real>& interface_cut_mesh,
         std::shared_ptr<dolfinx::fem::Function<Real>> speed,
         std::shared_ptr<dolfinx::fem::Function<Real>> velocity,
         std::shared_ptr<dolfinx::fem::Function<Real>> signed_distance,
         int k_ring, double normal_sign, int max_iter, double tol)
      {
        cutfemx::distance::FMMOptions opt;
        opt.max_iter = max_iter;
        opt.tol = tol;
        cutfemx::distance::extend_normal_velocity<Real>(
            std::move(phi), *interface_speed, interface_cut_mesh, *speed,
            *velocity, *signed_distance, k_ring, normal_sign, opt);
      },
      nb::arg("phi"), nb::arg("interface_speed"),
      nb::arg("interface_cut_mesh"), nb::arg("speed"), nb::arg("velocity"),
      nb::arg("signed_distance"), nb::arg("k_ring") = 1,
      nb::arg("normal_sign") = 1.0, nb::arg("max_iter") = 1000,
      nb::arg("tol") = 1.0e-10);

  m.def(
      "refinement_edges_from_stl",
      [](dolfinx_wrappers::MPICommWrapper,
         dolfinx::mesh::Mesh<Real>& mesh, const std::string& stl_path,
         double aabb_padding, int k_ring)
      {
        auto edges = cutfemx::distance::stl::refinement_edges_from_stl(
            mesh, stl_path, aabb_padding, k_ring);

        auto* data = new std::int32_t[edges.size()];
        std::copy(edges.begin(), edges.end(), data);
        nb::capsule owner(data,
                          [](void* p) noexcept
                          { delete[] static_cast<std::int32_t*>(p); });
        return nb::ndarray<std::int32_t, nb::numpy>(
            data, {edges.size()}, owner);
      },
      nb::arg("comm"), nb::arg("mesh"), nb::arg("stl_path"),
      nb::arg("aabb_padding") = 0.0, nb::arg("k_ring") = 1);
}
} // namespace

namespace cutfemx_wrappers
{
void distance(nb::module_& m)
{
  nb::enum_<cutfemx::distance::SignMode>(m, "SignMode")
      .value("LocalNormalBand", cutfemx::distance::SignMode::LocalNormalBand)
      .value("ComponentAnchor", cutfemx::distance::SignMode::ComponentAnchor)
      .value("WindingNumber", cutfemx::distance::SignMode::WindingNumber)
      .export_values();

  declare_distance<float>(m, "float32");
  declare_distance<double>(m, "float64");
}
} // namespace cutfemx_wrappers
