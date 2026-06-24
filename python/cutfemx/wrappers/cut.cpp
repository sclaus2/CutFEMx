// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <array.h>

#include <algorithm>
#include <stdexcept>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <span>

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include <cutfemx/mesh/cut_mesh.h>
#include <cutfemx/cut/cut.h>
#include <cutfemx/cut/runtime_quadrature.h>

namespace nb = nanobind;

namespace
{
template <typename Options>
constexpr bool has_cut_refinement_options
    = requires(Options& options, int value) {
        options.max_refinement_iterations = value;
        options.edge_max_depth = value;
      };

template <typename T>
int local_facet_index(std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
                      std::int32_t cell, std::int32_t facet)
{
  const int tdim = mesh->topology()->dim();
  auto c_to_f = mesh->topology()->connectivity(tdim, tdim - 1);
  if (!c_to_f)
    throw std::runtime_error("Cell-to-facet connectivity is unavailable.");

  auto facets = c_to_f->links(cell);
  auto it = std::find(facets.begin(), facets.end(), facet);
  if (it == facets.end())
    throw std::runtime_error("Could not resolve local facet index.");
  return static_cast<int>(std::distance(facets.begin(), it));
}

template <typename T>
std::vector<std::int32_t> facet_integration_rows(
    std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
    std::span<const std::int32_t> facet_ids, const std::string& integral_type)
{
  if (!mesh)
    throw std::runtime_error("Cannot build facet rows without a mesh.");

  const int tdim = mesh->topology()->dim();
  if (tdim < 1)
    throw std::runtime_error("Facet integration requires mesh tdim >= 1.");
  const int fdim = tdim - 1;

  mesh->topology_mutable()->create_entities(fdim);
  mesh->topology_mutable()->create_connectivity(fdim, tdim);
  mesh->topology_mutable()->create_connectivity(tdim, fdim);

  auto f_to_c = mesh->topology()->connectivity(fdim, tdim);
  auto c_to_f = mesh->topology()->connectivity(tdim, fdim);
  if (!f_to_c || !c_to_f)
    throw std::runtime_error("Facet-cell connectivity is unavailable.");

  const bool exterior = integral_type == "exterior_facet";
  const bool interior = integral_type == "interior_facet";
  if (!exterior && !interior)
  {
    throw std::runtime_error(
        "facet_integration_rows expects 'exterior_facet' or 'interior_facet'.");
  }

  std::vector<std::int32_t> rows;
  rows.reserve(facet_ids.size() * (exterior ? 2 : 4));
  for (std::int32_t facet : facet_ids)
  {
    auto cells = f_to_c->links(facet);
    if (exterior)
    {
      if (cells.size() != 1)
      {
        throw std::runtime_error(
            "Exterior facet domain contains a non-boundary facet.");
      }
      const std::int32_t cell = cells[0];
      rows.push_back(cell);
      rows.push_back(local_facet_index(mesh, cell, facet));
    }
    else
    {
      if (cells.size() != 2)
      {
        throw std::runtime_error(
            "Interior facet domain contains a facet without two adjacent cells.");
      }
      for (std::int32_t cell : cells)
      {
        rows.push_back(cell);
        rows.push_back(local_facet_index(mesh, cell, facet));
      }
    }
  }
  return rows;
}

inline cutcells::CutOptions make_cut_options(
    std::string cut_approximation, int cut_approximation_order,
    int max_refinement_iterations, int edge_max_depth)
{
  cutcells::CutOptions options;
  options.triangulate_cut_parts = true;
  options.triangulation_strategy
      = cutcells::cell::TriangulationStrategy::classical;
  options.cut_approximation = std::move(cut_approximation);
  options.cut_approximation_order = cut_approximation_order;
  if constexpr (has_cut_refinement_options<cutcells::CutOptions>)
  {
    options.max_refinement_iterations = max_refinement_iterations;
    options.edge_max_depth = edge_max_depth;
  }
  else if (max_refinement_iterations != 8 || edge_max_depth != 20)
  {
    throw std::invalid_argument(
        "This CutCells build does not support max_refinement_iterations or "
        "edge_max_depth. Rebuild CutCells from a version that exposes adaptive "
        "cut refinement options, or use the default values.");
  }
  return options;
}

template <typename T>
void declare_cut_api(nb::module_& m, std::string type)
{
  using CutData = cutfemx::CutData<T>;
  using CutMesh = cutfemx::mesh::CutMesh<T>;
  using RuntimeQuadrature = cutfemx::RuntimeQuadrature<T>;

  std::string cut_mesh_name = "CutMesh_" + type;
  nb::class_<CutMesh>(m, cut_mesh_name.c_str(), "CutFEMx cut mesh")
      .def_prop_ro(
          "background_mesh",
          [](CutMesh& self) { return self._bg_mesh; },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "cut_mesh",
          [](CutMesh& self) { return self._cut_mesh; },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "parent_index",
          [](CutMesh& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self._parent_index.data(), {self._parent_index.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "is_cut_cell",
          [](CutMesh& self)
          {
            return nb::ndarray<const std::int8_t, nb::numpy>(
                self._is_cut_cell.data(), {self._is_cut_cell.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal);

  std::string quadrature_name = "RuntimeQuadrature_" + type;
  nb::class_<RuntimeQuadrature>(m, quadrature_name.c_str(),
                                "CutFEMx runtime quadrature rules")
      .def_prop_ro("tdim",
                   [](RuntimeQuadrature& self)
                   { return self.cutcells_rules._tdim; })
      .def_prop_ro("gdim", [](RuntimeQuadrature& self) { return self.gdim(); })
      .def_prop_ro(
          "points",
          [](RuntimeQuadrature& self)
          {
            const std::size_t npoints = self.cutcells_rules._weights.size();
            return nb::ndarray<const T, nb::numpy>(
                self.cutcells_rules._points.data(),
                {npoints,
                 static_cast<std::size_t>(self.cutcells_rules._tdim)},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "weights",
          [](RuntimeQuadrature& self)
          {
            return nb::ndarray<const T, nb::numpy>(
                self.cutcells_rules._weights.data(),
                {self.cutcells_rules._weights.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "offsets",
          [](RuntimeQuadrature& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.cutcells_rules._offset.data(),
                {self.cutcells_rules._offset.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "parent_map",
          [](RuntimeQuadrature& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.cutcells_rules._parent_map.data(),
                {self.cutcells_rules._parent_map.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "physical_points",
          [](RuntimeQuadrature& self)
          {
            const std::vector<T>& points = self.physical_points();
            return nb::ndarray<const T, nb::numpy>(
                points.data(),
                {static_cast<std::size_t>(self.gdim()),
                 self.cutcells_rules._weights.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("kind",
                   [](RuntimeQuadrature&) { return std::string("per_entity"); });

  std::string pyclass_name = "CutData_" + type;
  nb::class_<CutData>(m, pyclass_name.c_str(), "CutFEMx cut data")
      .def("update", [](CutData& self) { cutfemx::update(self); })
      .def_prop_ro("num_local_cells",
                   [](const CutData& self) { return self.num_local_cells; })
      .def_prop_ro("tdim", [](const CutData& self) { return self.tdim; })
      .def_prop_ro("gdim", [](const CutData& self) { return self.gdim; })
      .def_prop_ro("level_set_names",
                   [](const CutData& self) { return self.level_set_names; });

  m.def(
      "cut",
      [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
         const std::string& cut_approximation, int cut_approximation_order,
         int max_refinement_iterations, int edge_max_depth)
      {
        return std::make_shared<CutData>(cutfemx::cut(
            std::move(level_set),
            make_cut_options(cut_approximation, cut_approximation_order,
                             max_refinement_iterations, edge_max_depth)));
      },
      nb::arg("level_set"), nb::arg("cut_approximation") = "auto",
      nb::arg("cut_approximation_order") = 1,
      nb::arg("max_refinement_iterations") = 8,
      nb::arg("edge_max_depth") = 20);

  m.def(
      "cut_entities",
      [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         int entity_dim, const std::string& cut_approximation,
         int cut_approximation_order, int max_refinement_iterations,
         int edge_max_depth)
      {
        return std::make_shared<CutData>(cutfemx::cut(
            std::move(level_set),
            std::span<const std::int32_t>(entities.data(), entities.size()),
            entity_dim,
            make_cut_options(cut_approximation, cut_approximation_order,
                             max_refinement_iterations, edge_max_depth)));
      },
      nb::arg("level_set"), nb::arg("entities"), nb::arg("entity_dim"),
      nb::arg("cut_approximation") = "auto",
      nb::arg("cut_approximation_order") = 1,
      nb::arg("max_refinement_iterations") = 8,
      nb::arg("edge_max_depth") = 20);

  m.def(
      "cut_multi",
      [](std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets,
         const std::string& cut_approximation, int cut_approximation_order,
         int max_refinement_iterations, int edge_max_depth)
      {
        return std::make_shared<CutData>(cutfemx::cut<T>(
            std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>(
                level_sets.data(), level_sets.size()),
            make_cut_options(cut_approximation, cut_approximation_order,
                             max_refinement_iterations, edge_max_depth)));
      },
      nb::arg("level_sets"), nb::arg("cut_approximation") = "auto",
      nb::arg("cut_approximation_order") = 1,
      nb::arg("max_refinement_iterations") = 8,
      nb::arg("edge_max_depth") = 20);

  m.def(
      "cut_multi_entities",
      [](std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         int entity_dim, const std::string& cut_approximation,
         int cut_approximation_order, int max_refinement_iterations,
         int edge_max_depth)
      {
        return std::make_shared<CutData>(cutfemx::cut<T>(
            std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>(
                level_sets.data(), level_sets.size()),
            std::span<const std::int32_t>(entities.data(), entities.size()),
            entity_dim,
            make_cut_options(cut_approximation, cut_approximation_order,
                             max_refinement_iterations, edge_max_depth)));
      },
      nb::arg("level_sets"), nb::arg("entities"), nb::arg("entity_dim"),
      nb::arg("cut_approximation") = "auto",
      nb::arg("cut_approximation_order") = 1,
      nb::arg("max_refinement_iterations") = 8,
      nb::arg("edge_max_depth") = 20);

  m.def(
      "update_cut",
      [](CutData& cut_data) { cutfemx::update(cut_data); },
      nb::arg("cut"));

  m.def(
      "locate_entities",
      [](const CutData& cut_data, const std::string& ls_part)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::locate_entities(cut_data, ls_part));
      },
      nb::arg("cut"), nb::arg("ls_part"));

  m.def(
      ("interior_facets_for_cells_" + type).c_str(),
      [](std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cells,
         bool include_ghosts)
      {
        return dolfinx_wrappers::as_nbarray(
            cutfemx::interior_facets_for_cells<T>(
                std::move(mesh),
                std::span<const std::int32_t>(cells.data(), cells.size()),
                include_ghosts));
      },
      nb::arg("mesh"), nb::arg("cells"), nb::arg("include_ghosts") = false,
      "Return raw local interior facet ids touched by cells.");

  m.def(
      "create_cut_mesh",
      [](const CutData& cut_data, const std::string& ls_part,
         const std::string& mode)
      { return cutfemx::create_cut_mesh(cut_data, ls_part, mode); },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("mode"));

  m.def(
      "runtime_quadrature",
      [](const CutData& cut_data, const std::string& ls_part, int order,
         const std::string& backend)
      { return cutfemx::runtime_quadrature(cut_data, ls_part, order, backend); },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("order"),
      nb::arg("backend") = "straight");

  m.def(
      "runtime_quadratures",
      [](const CutData& cut_data, std::vector<std::string> ls_parts, int order,
         const std::string& backend)
      {
        return cutfemx::runtime_quadratures<T>(
            cut_data,
            std::span<const std::string>(ls_parts.data(), ls_parts.size()),
            order, backend);
      },
      nb::arg("cut"), nb::arg("ls_parts"), nb::arg("order"),
      nb::arg("backend") = "straight");

  m.def(
      ("facet_integration_rows_" + type).c_str(),
      [](std::shared_ptr<const dolfinx::mesh::Mesh<T>> mesh,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> facets,
         const std::string& integral_type)
      {
        return dolfinx_wrappers::as_nbarray(facet_integration_rows<T>(
            std::move(mesh),
            std::span<const std::int32_t>(facets.data(), facets.size()),
            integral_type));
      },
      nb::arg("mesh"), nb::arg("facets"), nb::arg("integral_type"),
      "Convert raw local facet ids to DOLFINx integration rows.");
}
} // namespace

namespace cutfemx_wrappers
{
void cut(nb::module_& m)
{
  declare_cut_api<float>(m, "float32");
  declare_cut_api<double>(m, "float64");
}
} // namespace cutfemx_wrappers
