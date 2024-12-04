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
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <span>

#include <cutfemx/fem/CutForm.h>
#include <cutfemx/fem/interpolate.h>
#include <cutfemx/quadrature/quadrature.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Form.h>
#include <ufcx.h>


namespace nb = nanobind;

namespace
{
template <typename T>
void declare_fem(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  std::string pyclass_cutform_name = std::string("CutForm_") + type;
  nb::class_<cutfemx::fem::CutForm<T,U>>(m, pyclass_cutform_name.c_str(),
                                         "CutForm object")
      .def(
          "__init__",
          [](cutfemx::fem::CutForm<T, U>* fp, std::uintptr_t ufcform)
            //std::shared_ptr<const dolfinx::fem::Form<T, U>> form) 
            // const std::map<dolfinx::fem::IntegralType,
            // std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>>>&
            // subdomains)
          {
            std::cout << "coucoucou" << std::endl;
            // ufcx_form* p = reinterpret_cast<ufcx_form*>(ufcform);
            // new (fp)
            //     cutfemx::fem::CutForm<T, U>(cutfemx::fem::create_cut_form_factory<T>(
            //         *p, form, subdomains));
          }
          , "Create a Cutform from a pointer to a ufcx_form")
      .def(
          "integral_ids",
          [](const cutfemx::fem::CutForm<T, U>& self,
             dolfinx::fem::IntegralType type)
          {
            auto ids = self.integral_ids(type);
            return dolfinx_wrappers::as_nbarray(std::move(ids));
          },
          nb::arg("type"))
      .def_prop_ro("integral_types", &cutfemx::fem::CutForm<T, U>::integral_types)
      .def(
          "quadrature_rules",
          [](const cutfemx::fem::CutForm<T, U>& self,
             dolfinx::fem::IntegralType type, int i)
          {
            return self.quadrature_rules(type, i);
          },
          nb::rv_policy::reference_internal, nb::arg("type"), nb::arg("i"));;

  m.def("create_cut_function", [](const dolfinx::fem::Function<T,U>& u, const cutfemx::mesh::CutMesh<U>& sub_mesh)
          {
            return cutfemx::fem::create_cut_function(u, sub_mesh);
          }
        , "create function over cut mesh");

}

} // namespace

namespace cutfemx_wrappers
{
  void fem(nb::module_& m)
  {
    declare_fem<float>(m, "float32");
    declare_fem<double>(m, "float64");
  }
} // end of namespace cutfemx_wrappers