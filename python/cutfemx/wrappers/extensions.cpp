// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#include <array.h>

#include <complex>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <string>

#include <cutfemx/cut/cut.h>
#include <cutfemx/extensions/cell_aggregation.h>
#include <cutfemx/extensions/extension_penalty.h>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/MatrixCSR.h>

namespace nb = nanobind;

namespace
{
template <typename T>
void declare_extensions(nb::module_& m, const std::string& type)
{
  using CellAggregation = cutfemx::extensions::CellAggregation<T>;
  using ExtensionQuadrature = cutfemx::extensions::ExtensionQuadrature<T>;
  using CutData = cutfemx::CutData<T>;
  using FunctionSpace = dolfinx::fem::FunctionSpace<T>;
  using MatrixCSR = dolfinx::la::MatrixCSR<T>;

  std::string aggregation_name = "CellAggregation_" + type;
  nb::class_<CellAggregation>(m, aggregation_name.c_str(),
                              "CutFEMx cell aggregation data")
      .def_prop_ro(
          "active_cells",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.active_cells.data(), {self.active_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "cut_cells",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.cut_cells.data(), {self.cut_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "interior_cells",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.interior_cells.data(), {self.interior_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "well_posed_cells",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.well_posed_cells.data(), {self.well_posed_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "ill_posed_cells",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.ill_posed_cells.data(), {self.ill_posed_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "root_cell",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.root_cell.data(), {self.root_cell.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "aggregate_id",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.aggregate_id.data(), {self.aggregate_id.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "propagation_depth",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.propagation_depth.data(), {self.propagation_depth.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "rootless_cells",
          [](CellAggregation& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.rootless_cells.data(), {self.rootless_cells.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "cut_volume_fraction",
          [](CellAggregation& self)
          {
            return nb::ndarray<const T, nb::numpy>(
                self.cut_volume_fraction.data(),
                {self.cut_volume_fraction.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal);

  std::string quadrature_name = "ExtensionQuadrature_" + type;
  nb::class_<ExtensionQuadrature>(m, quadrature_name.c_str(),
                                  "Full-cell bad/root extension quadrature")
      .def_prop_ro(
          "bad_cells",
          [](ExtensionQuadrature& self)
          {
            std::vector<std::int32_t> cells;
            cells.reserve(self.pairs.size());
            for (const auto& pair : self.pairs)
              cells.push_back(pair.bad_cell);
            return cells;
          })
      .def_prop_ro(
          "root_cells",
          [](ExtensionQuadrature& self)
          {
            std::vector<std::int32_t> cells;
            cells.reserve(self.pairs.size());
            for (const auto& pair : self.pairs)
              cells.push_back(pair.root_cell);
            return cells;
          })
      .def_prop_ro(
          "offsets",
          [](ExtensionQuadrature& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.offsets.data(), {self.offsets.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "bad_reference_points",
          [](ExtensionQuadrature& self)
          {
            return nb::ndarray<const T, nb::numpy>(
                self.bad_reference_points.data(),
                {self.bad_reference_points.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "root_reference_points",
          [](ExtensionQuadrature& self)
          {
            return nb::ndarray<const T, nb::numpy>(
                self.root_reference_points.data(),
                {self.root_reference_points.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "weights",
          [](ExtensionQuadrature& self)
          {
            return nb::ndarray<const T, nb::numpy>(
                self.weights.data(), {self.weights.size()},
                nb::cast(self, nb::rv_policy::reference));
          },
          nb::rv_policy::reference_internal)
      .def_ro("tdim", &ExtensionQuadrature::tdim);

  m.def(
      ("create_cell_aggregation_" + type).c_str(),
      [](const CutData& cut_data, const std::string& selector,
         T volume_fraction_threshold, const std::string& root_policy,
         int max_iterations, bool allow_rootless)
      {
        return cutfemx::extensions::create_cell_aggregation(
            cut_data, selector, volume_fraction_threshold,
            cutfemx::extensions::root_policy_from_string(root_policy),
            max_iterations, allow_rootless);
      },
      nb::arg("cut_data"), nb::arg("selector"),
      nb::arg("volume_fraction_threshold"),
      nb::arg("root_policy") = "interior_or_well_cut",
      nb::arg("max_iterations") = -1, nb::arg("allow_rootless") = false);

  m.def(
      ("extension_quadrature_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation, int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::extension_quadrature(
            *V, cut_data, aggregation, quadrature_degree);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      nb::arg("quadrature_degree"),
      "Build standard full-cell quadrature on bad cells, pulled back to roots.");

  m.def(
      ("create_extension_penalty_matrix_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::create_extension_penalty_matrix<T, T>(
            *V, cut_data, aggregation);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      "Create a MatrixCSR with sparsity for extension-penalty bad/root pairs.");

  m.def(
      ("assemble_extension_penalty_scalar_" + type).c_str(),
      [](MatrixCSR& A, std::shared_ptr<const FunctionSpace> V,
         const CutData& cut_data, const CellAggregation& aggregation, T beta,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        cutfemx::extensions::assemble_extension_penalty(
            A, *V, cut_data, aggregation, beta, quadrature_degree);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("cut_data"),
      nb::arg("aggregation"), nb::arg("beta"),
      nb::arg("quadrature_degree"),
      "Assemble scalar-coefficient extension penalty into an existing MatrixCSR.");

  m.def(
      ("assemble_extension_penalty_cellwise_" + type).c_str(),
      [](MatrixCSR& A, std::shared_ptr<const FunctionSpace> V,
         const CutData& cut_data, const CellAggregation& aggregation,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> beta_cell_values,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        cutfemx::extensions::assemble_extension_penalty(
            A, *V, cut_data, aggregation,
            std::span<const T>(beta_cell_values.data(),
                               beta_cell_values.size()),
            quadrature_degree);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("cut_data"),
      nb::arg("aggregation"), nb::arg("beta_cell_values"),
      nb::arg("quadrature_degree"),
      "Assemble cellwise-coefficient extension penalty into an existing MatrixCSR.");

  m.def(
      ("extension_penalty_matrix_scalar_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation, T beta, int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::extension_penalty_matrix(
            *V, cut_data, aggregation, beta, quadrature_degree);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      nb::arg("beta"), nb::arg("quadrature_degree"),
      "Create and assemble a scalar-coefficient extension penalty MatrixCSR.");

  m.def(
      ("extension_penalty_matrix_cellwise_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> beta_cell_values,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::extension_penalty_matrix(
            *V, cut_data, aggregation,
            std::span<const T>(beta_cell_values.data(),
                               beta_cell_values.size()),
            quadrature_degree);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      nb::arg("beta_cell_values"),
      nb::arg("quadrature_degree"),
      "Create and assemble a cellwise-coefficient extension penalty MatrixCSR.");
}

template <typename Scalar, typename Geometry>
void declare_extension_matrices(nb::module_& m, const std::string& type)
{
  using CellAggregation = cutfemx::extensions::CellAggregation<Geometry>;
  using CutData = cutfemx::CutData<Geometry>;
  using FunctionSpace = dolfinx::fem::FunctionSpace<Geometry>;
  using MatrixCSR = dolfinx::la::MatrixCSR<Scalar>;

  m.def(
      ("create_extension_penalty_matrix_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::create_extension_penalty_matrix<Scalar, Geometry>(
            *V, cut_data, aggregation);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      "Create a MatrixCSR with extension-penalty sparsity.");

  m.def(
      ("assemble_extension_penalty_scalar_" + type).c_str(),
      [](MatrixCSR& A, std::shared_ptr<const FunctionSpace> V,
         const CutData& cut_data, const CellAggregation& aggregation,
         Scalar beta, int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        cutfemx::extensions::assemble_extension_penalty(
            A, *V, cut_data, aggregation, beta, quadrature_degree);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("cut_data"),
      nb::arg("aggregation"), nb::arg("beta"),
      nb::arg("quadrature_degree"),
      "Assemble scalar-coefficient extension penalty into MatrixCSR.");

  m.def(
      ("assemble_extension_penalty_cellwise_" + type).c_str(),
      [](MatrixCSR& A, std::shared_ptr<const FunctionSpace> V,
         const CutData& cut_data, const CellAggregation& aggregation,
         nb::ndarray<const Scalar, nb::ndim<1>, nb::c_contig> beta_cell_values,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        cutfemx::extensions::assemble_extension_penalty(
            A, *V, cut_data, aggregation,
            std::span<const Scalar>(beta_cell_values.data(),
                                    beta_cell_values.size()),
            quadrature_degree);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("cut_data"),
      nb::arg("aggregation"), nb::arg("beta_cell_values"),
      nb::arg("quadrature_degree"),
      "Assemble cellwise-coefficient extension penalty into MatrixCSR.");

  m.def(
      ("extension_penalty_matrix_scalar_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation, Scalar beta, int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::extension_penalty_matrix(
            *V, cut_data, aggregation, beta, quadrature_degree);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      nb::arg("beta"), nb::arg("quadrature_degree"),
      "Create and assemble a scalar-coefficient MatrixCSR.");

  m.def(
      ("extension_penalty_matrix_cellwise_" + type).c_str(),
      [](std::shared_ptr<const FunctionSpace> V, const CutData& cut_data,
         const CellAggregation& aggregation,
         nb::ndarray<const Scalar, nb::ndim<1>, nb::c_contig> beta_cell_values,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        return cutfemx::extensions::extension_penalty_matrix(
            *V, cut_data, aggregation,
            std::span<const Scalar>(beta_cell_values.data(),
                                    beta_cell_values.size()),
            quadrature_degree);
      },
      nb::arg("V"), nb::arg("cut_data"), nb::arg("aggregation"),
      nb::arg("beta_cell_values"),
      nb::arg("quadrature_degree"),
      "Create and assemble a cellwise-coefficient MatrixCSR.");
}
} // namespace

namespace cutfemx_wrappers
{
void extensions(nb::module_& m)
{
  declare_extensions<float>(m, "float32");
  declare_extensions<double>(m, "float64");
  declare_extension_matrices<float, double>(m, "float32_float64");
  declare_extension_matrices<double, float>(m, "float64_float32");
  declare_extension_matrices<std::complex<float>, float>(
      m, "complex64_float32");
  declare_extension_matrices<std::complex<float>, double>(
      m, "complex64_float64");
  declare_extension_matrices<std::complex<double>, float>(
      m, "complex128_float32");
  declare_extension_matrices<std::complex<double>, double>(
      m, "complex128_float64");
}
} // namespace cutfemx_wrappers
