// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

#include <array.h>
#include <caster_petsc.h>

#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>

#include <dolfinx_custom_data/fem/assembler.h>

#include <cutfemx/fem/deactivate.h>
#include <cutfemx/cut/cut.h>
#include <cutfemx/extensions/cell_aggregation.h>
#include <cutfemx/extensions/extension_penalty.h>

#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace
{
template <typename T>
void validate_petsc_matrix_rows(Mat A, std::span<const std::int32_t> rows)
{
  if (A == nullptr)
    throw std::runtime_error("Block deactivation received a null PETSc matrix");

  PetscInt local_rows = 0;
  PetscInt local_cols = 0;
  MatGetLocalSize(A, &local_rows, &local_cols);
  if (!rows.empty() && rows.back() >= local_rows)
  {
    throw std::runtime_error(
        "ActiveDomain inactive rows are incompatible with the PETSc matrix row map");
  }
}

template <typename T, typename U>
void validate_petsc_block_deactivation_inputs(
    const std::vector<std::vector<Mat>>& A_blocks,
    const std::vector<cutfemx::fem::ActiveDomain<T, U>*>& active_domains)
{
  if (A_blocks.empty())
    throw std::runtime_error("Block deactivation requires at least one block row");
  if (A_blocks.size() != active_domains.size())
  {
    throw std::runtime_error(
        "Block deactivation requires one ActiveDomain per block row");
  }

  const std::size_t num_blocks = A_blocks.size();
  for (std::size_t i = 0; i < num_blocks; ++i)
  {
    if (A_blocks[i].size() != num_blocks)
    {
      throw std::runtime_error(
          "Block deactivation requires a square block matrix");
    }
    if (active_domains[i] == nullptr)
    {
      throw std::runtime_error(
          "Block deactivation received a null ActiveDomain");
    }
    validate_petsc_matrix_rows<T>(A_blocks[i][i],
                                  active_domains[i]->inactive_dofs);
  }
}

template <typename T, typename U>
void deactivate_outside_petsc_blocks(
    const std::vector<std::vector<Mat>>& A_blocks,
    const std::vector<cutfemx::fem::ActiveDomain<T, U>*>& active_domains,
    T diagonal)
{
  validate_petsc_block_deactivation_inputs<T, U>(A_blocks, active_domains);
  for (std::size_t i = 0; i < A_blocks.size(); ++i)
  {
    Mat Aii = A_blocks[i][i];
    MatAssemblyBegin(Aii, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(Aii, MAT_FLUSH_ASSEMBLY);

    cutfemx::fem::deactivate_outside(
        dolfinx::la::petsc::Matrix::set_fn(Aii, INSERT_VALUES),
        *active_domains[i], diagonal);
  }
}

template <typename T, typename U>
void deactivate_outside_petsc_blocks(
    const std::vector<std::vector<Mat>>& A_blocks,
    const std::vector<Vec>& b_blocks,
    const std::vector<cutfemx::fem::ActiveDomain<T, U>*>& active_domains,
    T diagonal, T rhs_value)
{
  validate_petsc_block_deactivation_inputs<T, U>(A_blocks, active_domains);
  if (b_blocks.size() != A_blocks.size())
  {
    throw std::runtime_error(
        "Block deactivation requires one RHS vector per block row");
  }

  for (std::size_t i = 0; i < A_blocks.size(); ++i)
  {
    Mat Aii = A_blocks[i][i];
    MatAssemblyBegin(Aii, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(Aii, MAT_FLUSH_ASSEMBLY);

    Vec b = b_blocks[i];
    if (b == nullptr)
      throw std::runtime_error("Block deactivation received a null RHS vector");

    Vec b_local;
    VecGhostGetLocalForm(b, &b_local);
    PetscInt n = 0;
    VecGetLocalSize(b_local, &n);
    PetscScalar* array = nullptr;
    VecGetArray(b_local, &array);

    try
    {
      cutfemx::fem::deactivate_outside(
          dolfinx::la::petsc::Matrix::set_fn(Aii, INSERT_VALUES),
          std::span<T>(reinterpret_cast<T*>(array), n), *active_domains[i],
          diagonal, rhs_value);
    }
    catch (...)
    {
      VecRestoreArray(b_local, &array);
      VecGhostRestoreLocalForm(b, &b_local);
      throw;
    }

    VecRestoreArray(b_local, &array);
    VecGhostRestoreLocalForm(b, &b_local);
  }
}

template <typename T>
bool petsc_row_has_nonzero(const std::vector<Mat>& row_blocks,
                           PetscInt global_row, double tol)
{
  for (Mat A : row_blocks)
  {
    if (A == nullptr)
      continue;

    PetscInt ncols = 0;
    const PetscInt* cols = nullptr;
    const PetscScalar* values = nullptr;
    MatGetRow(A, global_row, &ncols, &cols, &values);

    bool nonzero = false;
    for (PetscInt k = 0; k < ncols; ++k)
    {
      if (std::abs(static_cast<T>(values[k])) > tol)
      {
        nonzero = true;
        break;
      }
    }

    MatRestoreRow(A, global_row, &ncols, &cols, &values);
    if (nonzero)
      return true;
  }
  return false;
}

template <typename T>
std::vector<std::int32_t> zero_petsc_rows(Mat A, double tol)
{
  if (A == nullptr)
    throw std::runtime_error("Zero-row scan received a null PETSc matrix");

  MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

  PetscInt rstart = 0;
  PetscInt rend = 0;
  MatGetOwnershipRange(A, &rstart, &rend);

  std::vector<std::int32_t> rows;
  rows.reserve(static_cast<std::size_t>(rend - rstart));
  std::vector<Mat> row_blocks = {A};
  for (PetscInt row = rstart; row < rend; ++row)
    if (!petsc_row_has_nonzero<T>(row_blocks, row, tol))
      rows.push_back(static_cast<std::int32_t>(row - rstart));
  return rows;
}

template <typename T>
std::vector<std::vector<std::int32_t>> zero_petsc_block_rows(
    const std::vector<std::vector<Mat>>& A_blocks, double tol)
{
  if (A_blocks.empty())
    throw std::runtime_error("Zero-row scan requires at least one block row");

  const std::size_t num_blocks = A_blocks.size();
  std::vector<std::vector<std::int32_t>> rows(num_blocks);
  for (std::size_t i = 0; i < num_blocks; ++i)
  {
    if (A_blocks[i].size() != num_blocks)
      throw std::runtime_error("Zero-row scan requires a square block matrix");
    if (A_blocks[i][i] == nullptr)
      throw std::runtime_error("Zero-row scan requires every diagonal matrix block");

    PetscInt local_rows = 0;
    PetscInt local_cols = 0;
    MatGetLocalSize(A_blocks[i][i], &local_rows, &local_cols);
    for (std::size_t j = 0; j < num_blocks; ++j)
    {
      if (A_blocks[i][j] == nullptr)
        continue;
      MatAssemblyBegin(A_blocks[i][j], MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A_blocks[i][j], MAT_FLUSH_ASSEMBLY);

      PetscInt block_rows = 0;
      PetscInt block_cols = 0;
      MatGetLocalSize(A_blocks[i][j], &block_rows, &block_cols);
      if (block_rows != local_rows)
      {
        throw std::runtime_error(
            "Zero-row scan found incompatible row maps in a block row");
      }
    }

    PetscInt rstart = 0;
    PetscInt rend = 0;
    MatGetOwnershipRange(A_blocks[i][i], &rstart, &rend);
    rows[i].reserve(static_cast<std::size_t>(rend - rstart));
    for (PetscInt row = rstart; row < rend; ++row)
    {
      if (!petsc_row_has_nonzero<T>(A_blocks[i], row, tol))
        rows[i].push_back(static_cast<std::int32_t>(row - rstart));
    }
  }
  return rows;
}

template <typename T>
void declare_runtime_petsc(nb::module_& m, std::string type)
{
  using U = typename dolfinx::scalar_value_t<T>;
  using Form = dolfinx_custom_data::fem::Form<T, U>;
  using FunctionSpace = dolfinx::fem::FunctionSpace<U>;
  using DirichletBC = dolfinx::fem::DirichletBC<T, U>;
  using ActiveDomain = cutfemx::fem::ActiveDomain<T, U>;
  using CutData = cutfemx::CutData<U>;
  using CellAggregation = cutfemx::extensions::CellAggregation<U>;

  m.def(
      ("create_matrix_" + type).c_str(),
      [](const Form& form, std::optional<std::string> mat_type)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot create PETSc matrix. Form is not bilinear.");
        }

        dolfinx::la::SparsityPattern sp
            = dolfinx_custom_data::fem::create_sparsity_pattern(form);
        sp.finalize();
        return dolfinx::la::petsc::create_matrix(form.mesh()->comm(), sp,
                                                 mat_type);
      },
      nb::rv_policy::take_ownership, nb::arg("form"),
      nb::arg("type") = nb::none(),
      "Create a PETSc Mat compatible with a CutFEMx runtime form.");

  m.def(
      ("create_matrix_with_extension_sparsity_" + type).c_str(),
      [](const Form& form,
         const std::vector<std::shared_ptr<const FunctionSpace>>& spaces,
         const std::vector<const CellAggregation*>& aggregations,
         std::optional<std::string> mat_type)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot create PETSc matrix. Form is not bilinear.");
        }
        if (spaces.size() != aggregations.size())
        {
          throw std::runtime_error(
              "Extension sparsity spaces and aggregations have different sizes.");
        }

        dolfinx::la::SparsityPattern sp
            = dolfinx_custom_data::fem::create_sparsity_pattern(form);
        for (std::size_t i = 0; i < spaces.size(); ++i)
        {
          if (!spaces[i])
            throw std::runtime_error("Received a null extension function space.");
          if (aggregations[i] == nullptr)
            throw std::runtime_error("Received a null extension aggregation.");
          cutfemx::extensions::insert_extension_penalty_sparsity(
              sp, *spaces[i], *aggregations[i]);
        }
        sp.finalize();
        return dolfinx::la::petsc::create_matrix(form.mesh()->comm(), sp,
                                                 mat_type);
      },
      nb::rv_policy::take_ownership, nb::arg("form"), nb::arg("spaces"),
      nb::arg("aggregations"), nb::arg("type") = nb::none(),
      "Create a PETSc Mat with CutFEMx runtime form and extension sparsity.");

  m.def(
      ("assemble_vector_" + type).c_str(),
      [](Vec b, const Form& form)
      {
        if (form.rank() != 1)
        {
          throw std::runtime_error(
              "Cannot assemble PETSc vector. Form is not linear.");
        }

        Vec b_local;
        VecGhostGetLocalForm(b, &b_local);
        PetscInt n = 0;
        VecGetSize(b_local, &n);
        PetscScalar* array = nullptr;
        VecGetArray(b_local, &array);
        std::span<T> values(reinterpret_cast<T*>(array), n);
        dolfinx_custom_data::fem::assemble_vector(values, form);
        VecRestoreArray(b_local, &array);
        VecGhostRestoreLocalForm(b, &b_local);
      },
      nb::arg("b"), nb::arg("form"),
      "Assemble a linear CutFEMx runtime form into an existing PETSc Vec.");

  m.def(
      ("assemble_matrix_" + type).c_str(),
      [](Mat A, const Form& form, const std::vector<const DirichletBC*>& bcs)
      {
        if (form.rank() != 2)
        {
          throw std::runtime_error(
              "Cannot assemble PETSc matrix. Form is not bilinear.");
        }

        std::vector<std::reference_wrapper<const DirichletBC>> _bcs;
        for (const DirichletBC* bc : bcs)
        {
          assert(bc);
          _bcs.push_back(*bc);
        }

        const std::array<int, 2> data_bs
            = {form.function_spaces().at(0)->dofmaps(0)->index_map_bs(),
               form.function_spaces().at(1)->dofmaps(0)->index_map_bs()};

        if (data_bs[0] == data_bs[1])
        {
          dolfinx_custom_data::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), form,
              _bcs);
        }
        else
        {
          dolfinx_custom_data::fem::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_expand_fn(
                  A, data_bs[0], data_bs[1], ADD_VALUES),
              form, _bcs);
        }
      },
      nb::arg("A"), nb::arg("form"), nb::arg("bcs"),
      "Assemble a bilinear CutFEMx runtime form into an existing PETSc Mat.");

  m.def(
      ("assemble_extension_penalty_scalar_" + type).c_str(),
      [](Mat A, std::shared_ptr<const FunctionSpace> V,
         const CutData& cut_data, const CellAggregation& aggregation, T beta,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          std::span<const U>)>
            mat_add = dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES);
        cutfemx::extensions::assemble_extension_penalty(
            mat_add, *V, cut_data, aggregation, beta, quadrature_degree);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("cut_data"),
      nb::arg("aggregation"), nb::arg("beta"),
      nb::arg("quadrature_degree"),
      "Assemble a scalar-coefficient extension penalty into a PETSc Mat.");

  m.def(
      ("assemble_extension_penalty_cellwise_" + type).c_str(),
      [](Mat A, std::shared_ptr<const FunctionSpace> V,
         const CutData& cut_data, const CellAggregation& aggregation,
         nb::ndarray<const U, nb::ndim<1>, nb::c_contig> beta_cell_values,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          std::span<const U>)>
            mat_add = dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES);
        cutfemx::extensions::assemble_extension_penalty(
            mat_add, *V, cut_data, aggregation,
            std::span<const U>(beta_cell_values.data(),
                               beta_cell_values.size()),
            quadrature_degree);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("cut_data"),
      nb::arg("aggregation"), nb::arg("beta_cell_values"),
      nb::arg("quadrature_degree"),
      "Assemble a cellwise-coefficient extension penalty into a PETSc Mat.");

  m.def(
      ("insert_diagonal_" + type).c_str(),
      [](Mat A, const FunctionSpace& V,
         const std::vector<const DirichletBC*>& bcs, T diagonal)
      {
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

        std::vector<std::reference_wrapper<const DirichletBC>> _bcs;
        for (const DirichletBC* bc : bcs)
        {
          assert(bc);
          _bcs.push_back(*bc);
        }
        dolfinx_custom_data::fem::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES), V, _bcs,
            diagonal);
      },
      nb::arg("A"), nb::arg("V"), nb::arg("bcs"), nb::arg("diagonal"),
      "Insert a diagonal value for constrained PETSc matrix rows.");

  m.def(
      ("deactivate_outside_" + type).c_str(),
      [](Mat A, ActiveDomain& active_domain, T diagonal)
      {
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

        PetscInt local_rows = 0;
        PetscInt local_cols = 0;
        MatGetLocalSize(A, &local_rows, &local_cols);
        if (!active_domain.inactive_dofs.empty()
            && active_domain.inactive_dofs.back() >= local_rows)
        {
          throw std::runtime_error(
              "ActiveDomain inactive rows are incompatible with the PETSc matrix row map");
        }
        cutfemx::fem::deactivate_outside(
            dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
            active_domain, diagonal);
      },
      nb::arg("A"), nb::arg("active_domain"), nb::arg("diagonal"),
      "Deactivate PETSc matrix rows outside a CutFEMx active domain.");

  m.def(
      ("deactivate_outside_matrix_vector_" + type).c_str(),
      [](Mat A, Vec b, ActiveDomain& active_domain, T diagonal, T rhs_value)
      {
        MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

        PetscInt local_rows = 0;
        PetscInt local_cols = 0;
        MatGetLocalSize(A, &local_rows, &local_cols);
        if (!active_domain.inactive_dofs.empty()
            && active_domain.inactive_dofs.back() >= local_rows)
        {
          throw std::runtime_error(
              "ActiveDomain inactive rows are incompatible with the PETSc matrix row map");
        }

        Vec b_local;
        VecGhostGetLocalForm(b, &b_local);
        PetscInt n = 0;
        VecGetLocalSize(b_local, &n);
        PetscScalar* array = nullptr;
        VecGetArray(b_local, &array);

        try
        {
          cutfemx::fem::deactivate_outside(
              dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
              std::span<T>(reinterpret_cast<T*>(array), n),
              active_domain, diagonal, rhs_value);
        }
        catch (...)
        {
          VecRestoreArray(b_local, &array);
          VecGhostRestoreLocalForm(b, &b_local);
          throw;
        }

        VecRestoreArray(b_local, &array);
        VecGhostRestoreLocalForm(b, &b_local);
      },
      nb::arg("A"), nb::arg("b"), nb::arg("active_domain"),
      nb::arg("diagonal"), nb::arg("rhs_value"),
      "Deactivate PETSc matrix rows and matching RHS entries outside a CutFEMx active domain.");

  m.def(
      ("deactivate_outside_blocks_" + type).c_str(),
      [](const std::vector<std::vector<Mat>>& A_blocks,
         const std::vector<ActiveDomain*>& active_domains, T diagonal)
      {
        deactivate_outside_petsc_blocks<T, U>(A_blocks, active_domains,
                                              diagonal);
      },
      nb::arg("A_blocks"), nb::arg("active_domains"),
      nb::arg("diagonal"),
      "Deactivate a PETSc block system from per-row ActiveDomain objects.");

  m.def(
      ("deactivate_outside_blocks_matrix_vector_" + type).c_str(),
      [](const std::vector<std::vector<Mat>>& A_blocks,
         const std::vector<Vec>& b_blocks,
         const std::vector<ActiveDomain*>& active_domains, T diagonal,
         T rhs_value)
      {
        deactivate_outside_petsc_blocks<T, U>(
            A_blocks, b_blocks, active_domains, diagonal, rhs_value);
      },
      nb::arg("A_blocks"), nb::arg("b_blocks"), nb::arg("active_domains"),
      nb::arg("diagonal"), nb::arg("rhs_value"),
      "Deactivate a PETSc block system and matching RHS blocks from per-row ActiveDomain objects.");

  m.def(
      ("zero_rows_" + type).c_str(),
      [](Mat A, double tol) { return zero_petsc_rows<T>(A, tol); },
      nb::arg("A"), nb::arg("tol") = 0.0,
      "Return owned local PETSc rows whose entries are all zero.");

  m.def(
      ("zero_block_rows_" + type).c_str(),
      [](const std::vector<std::vector<Mat>>& A_blocks, double tol)
      { return zero_petsc_block_rows<T>(A_blocks, tol); },
      nb::arg("A_blocks"), nb::arg("tol") = 0.0,
      "Return owned local rows whose entries are zero across each PETSc block row.");
}
} // namespace

namespace cutfemx_wrappers
{
void petsc_runtime(nb::module_& m)
{
  nb::module_ petsc_mod
      = m.def_submodule("petsc", "PETSc-specific runtime FEM module");
  declare_runtime_petsc<double>(petsc_mod, "float64");
}
} // namespace cutfemx_wrappers

#endif
