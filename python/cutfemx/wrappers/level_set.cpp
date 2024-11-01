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
#include <span>

#include <dolfinx/fem/Function.h>
#include <dolfinx/common/types.h>

#include <cutcells/cut_cell.h>

#include <cutfemx/level_set/cut_entities.h>
#include <cutfemx/level_set/locate_entities.h>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_level_set(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_type_t<T>;

  m.def("locate_entities", [](std::shared_ptr<const dolfinx::fem::Function<T,U>> level_set,
                                       const int& dim,
                                       const std::string& ls_part)
            {
              bool include_ghost = true;
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
}

} // namespace

namespace cutfemx_wrappers
{

  void level_set(nb::module_& m)
  {
    declare_level_set<float>(m, "float32");
    declare_level_set<double>(m, "float64");
  }
} // end of namespace cutfemx_wrappers