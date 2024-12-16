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
#include <span>

#include <dolfinx/fem/Function.h>
#include <dolfinx/common/types.h>

#include <cutcells/cut_cell.h>
#include <cutcells/cut_mesh.h>

#include <cutfemx/level_set/cut_entities.h>
#include <cutfemx/level_set/locate_entities.h>
#include <cutfemx/level_set/ghost_penalty_facets.h>
#include <cutfemx/level_set/compute_normal.h>

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