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
#include <nanobind/stl/pair.h>
#include <span>

#include <cutcells/cut_mesh.h>

#include <cutfemx/quadrature/quadrature.h>
#include <cutfemx/quadrature/generation.h>
#include <cutfemx/quadrature/physical_points.h>

#include <dolfinx/mesh/Mesh.h>


namespace nb = nanobind;

namespace
{
template <typename T>
void declare_quadrature(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  std::string pyclass_quadrature_name = std::string("QuadratureRule_") + type;
  nb::class_<cutfemx::quadrature::QuadratureRule<T>>(m, pyclass_quadrature_name.c_str(),
                                         "QuadratureRule object")
      .def(
          "__init__",
          [](cutfemx::quadrature::QuadratureRule<T>* self,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> points,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> weights)
          {
            // Explicitly call constructor via placement new since nanobind allocated raw memory
            new (self) cutfemx::quadrature::QuadratureRule<T>();

            // Copy data from numpy to std::vector
            self->_points.assign(points.data(), points.data() + points.size());
            self->_weights.assign(weights.data(), weights.data() + weights.size());
          },
          nb::arg("points"), nb::arg("weights"))
      .def_prop_ro(
          "points",
          [](cutfemx::quadrature::QuadratureRule<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(
                self._points.data(), {self._weights.size(), self._points.size()/self._weights.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal,
          " Return quadrature points in reference space.")
      .def_prop_ro(
          "weights",
          [](cutfemx::quadrature::QuadratureRule<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(
                self._weights.data(), {self._weights.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal,
          " Return weights in reference space.");

  std::string pyclass_quadratures_name = std::string("QuadratureRules_") + type;
  nb::class_<cutfemx::quadrature::QuadratureRules<T>>(m, pyclass_quadratures_name.c_str(),
                                         "QuadratureRules object")
      .def(
          "__init__",
          [](cutfemx::quadrature::QuadratureRules<T>* self,
             std::vector<cutfemx::quadrature::QuadratureRule<T>>& rules,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> parent_map)
          {
            new (self) cutfemx::quadrature::QuadratureRules<T>();

            self->_quadrature_rules = rules;

            self->_parent_map.assign(parent_map.data(), parent_map.data() + parent_map.size());
          },
          nb::arg("quadrature rules"), nb::arg("parent map"))
      .def_prop_ro(
          "quadrature_rules",
          [](cutfemx::quadrature::QuadratureRules<T>& self)
          {
            return self._quadrature_rules;
          },
          nb::rv_policy::reference_internal,
          "Return quadrature points in reference space.")
      .def_prop_ro(
          "parent_map",
          [](cutfemx::quadrature::QuadratureRules<T>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(self._parent_map.data(),{self._parent_map.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal,
          " Return map of quadrature rules to parent mesh cell.");

  m.def("runtime_quadrature", [](std::shared_ptr<const dolfinx::fem::Function<T>> level_set,
                     const std::string& ls_part,
                     const int& order)
            {
              cutfemx::quadrature::QuadratureRules<T> runtime_rules;
              cutfemx::quadrature::runtime_quadrature(level_set, ls_part, order, runtime_rules);

              return runtime_rules;
             },
             "generate runtime quadrature rules");

  m.def("physical_points", [](const cutfemx::quadrature::QuadratureRules<T>& runtime_rules, const dolfinx::mesh::Mesh<T>& mesh)
          {
            const std::size_t gdim = mesh.geometry().dim();
            auto points = cutfemx::quadrature::physical_points(runtime_rules, mesh);

            std::size_t num_points = points.size()/gdim;

            T *points_3d = new T[num_points * 3];

            for(std::size_t i=0;i<num_points;i++)
            {
              for(std::size_t j=0;j<gdim;j++)
              {
                points_3d[i*3+j] = points[i*gdim+j];
              }
            }

            if(gdim==2)
            {
              for(std::size_t i=0;i<num_points;i++)
              {
                  points_3d[i*3+2] = 0.0;
              }
            }

            // Delete 'data' when the 'owner' capsule expires
            nb::capsule owner(points_3d, [](void *p) noexcept {
              delete[] (T *) p;
            });

            return nb::ndarray<nb::numpy, T, nb::shape<-1, 3>, nb::c_contig>(
              /* data = */ points_3d,
              /* shape = */ { num_points, 3 },
              /* owner = */ owner
              );
          }
        , "quadrature points in physical space in background mesh");
}

} // namespace

namespace cutfemx_wrappers
{
  void quadrature(nb::module_& m)
  {
    declare_quadrature<float>(m, "float32");
    declare_quadrature<double>(m, "float64");
  }
} // end of namespace cutfemx_wrappers