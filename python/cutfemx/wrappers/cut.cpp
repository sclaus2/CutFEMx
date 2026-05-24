// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <array.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <span>

#include <dolfinx/fem/Function.h>

#include <cutfemx/mesh/cut_mesh.h>
#include <cutfemx/cut/cut.h>
#include <cutfemx/cut/runtime_quadrature.h>

namespace nb = nanobind;

namespace
{
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
      [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set)
      { return std::make_shared<CutData>(cutfemx::cut(std::move(level_set))); },
      nb::arg("level_set"));

  m.def(
      "cut_multi",
      [](std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> level_sets)
      {
        return std::make_shared<CutData>(cutfemx::cut<T>(
            std::span<const std::shared_ptr<const dolfinx::fem::Function<T>>>(
                level_sets.data(), level_sets.size())));
      },
      nb::arg("level_sets"));

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
      "locate_entities_subset",
      [](const CutData& cut_data, const std::string& ls_part,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         int entity_dim)
      {
        return dolfinx_wrappers::as_nbarray(cutfemx::locate_entities(
            cut_data,
            std::span<const std::int32_t>(entities.data(), entities.size()),
            entity_dim, ls_part));
      },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("entities"),
      nb::arg("entity_dim"));

  m.def(
      "create_cut_mesh",
      [](const CutData& cut_data, const std::string& ls_part,
         const std::string& mode)
      { return cutfemx::create_cut_mesh(cut_data, ls_part, mode); },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("mode"));

  m.def(
      "create_cut_mesh_subset",
      [](const CutData& cut_data, const std::string& ls_part,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         int entity_dim, const std::string& mode)
      {
        return cutfemx::create_cut_mesh(
            cut_data,
            std::span<const std::int32_t>(entities.data(), entities.size()),
            entity_dim, ls_part, mode);
      },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("entities"),
      nb::arg("entity_dim"), nb::arg("mode"));

  m.def(
      "runtime_quadrature",
      [](const CutData& cut_data, const std::string& ls_part, int order)
      { return cutfemx::runtime_quadrature(cut_data, ls_part, order); },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("order"));

  m.def(
      "runtime_quadrature_subset",
      [](const CutData& cut_data, const std::string& ls_part, int order,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
         int entity_dim)
      {
        return cutfemx::runtime_quadrature(
            cut_data,
            std::span<const std::int32_t>(entities.data(), entities.size()),
            entity_dim, ls_part, order);
      },
      nb::arg("cut"), nb::arg("ls_part"), nb::arg("order"),
      nb::arg("entities"), nb::arg("entity_dim"));
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
