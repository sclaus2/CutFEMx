// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <petsc.h>
#include <petscmat.h>
#include <petscvec.h>

#include <array.h>
#include <caster_petsc.h>

#include <dolfinx/common/IndexMap.h>
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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace
{
/// Return a DOLFINx PETSc set function that throws when PETSc rejects a
/// value. DOLFINx checks the PETSc error code only in debug builds, and its
/// matrices forbid new nonzeros, so an entry outside the nonzero structure
/// would otherwise be dropped silently.
auto checked_set_fn(Mat A, InsertMode mode)
{
  return [set_fn = dolfinx::la::petsc::Matrix::set_fn(A, mode)](
             std::span<const std::int32_t> rows,
             std::span<const std::int32_t> cols,
             std::span<const PetscScalar> vals) mutable -> int
  {
    if (PetscErrorCode ierr = set_fn(rows, cols, vals); ierr != 0)
    {
      const char* desc = nullptr;
      PetscErrorMessage(ierr, &desc, nullptr);
      throw std::runtime_error(
          "Failed to set PETSc matrix values (PETSc error "
          + std::to_string(ierr) + ": " + (desc ? desc : "unknown")
          + "). Entries set after assembly, such as the diagonal of "
            "deactivated rows, must be in the matrix nonzero structure; create "
            "the matrix with cutfemx.petsc.create_matrix.");
    }
    return 0;
  };
}

/// Write explicit zeros to the owned diagonal reserved by the runtime form
/// sparsity pattern. PETSc drops preallocated entries that were never written
/// at the first MAT_FINAL_ASSEMBLY, and DOLFINx matrices forbid new nonzeros
/// afterwards. Rows outside the active integration domain receive no assembly
/// contribution, so without explicit zeros deactivate_outside could not set
/// their diagonal once the matrix has been assembled.
template <dolfinx::scalar T, std::floating_point U>
void write_reserved_diagonal(Mat A,
                             const dolfinx_custom_data::fem::Form<T, U>& form)
{
  if (!dolfinx_custom_data::fem::reserves_deactivation_diagonal(form))
    return;

  const auto dofmap = form.function_spaces().at(0)->dofmaps().front();
  std::vector<std::int32_t> rows(dofmap->index_map->size_local()
                                 * dofmap->index_map_bs());
  std::iota(rows.begin(), rows.end(), 0);
  dolfinx_custom_data::fem::set_diagonal(checked_set_fn(A, ADD_VALUES),
                                         std::span<const std::int32_t>(rows),
                                         T(0));

  // Reset the insert mode so the next caller may add or insert values.
  MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
}

/// Complete assembly so rows can be read back with MatGetRow, which PETSc
/// rejects for matrices that are only flush-assembled.
void finalize_for_row_access(Mat A)
{
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

void check_petsc(PetscErrorCode ierr, const std::string& what)
{
  if (ierr != 0)
    dolfinx::la::petsc::error(ierr, __FILE__, what);
}

/// Return whether A is a MATSEQAIJ or MATMPIAIJ matrix in host memory, whose
/// CSR arrays AIJBlockAdder writes.
bool is_host_aij(Mat A)
{
  PetscBool aij = PETSC_FALSE;
  check_petsc(PetscObjectTypeCompareAny(reinterpret_cast<PetscObject>(A), &aij,
                                        MATSEQAIJ, MATMPIAIJ, ""),
              "PetscObjectTypeCompareAny");
  return aij == PETSC_TRUE;
}

/// Insert explicit zeros for every entry of a finalized sparsity pattern
/// and complete the assembly of A.
///
/// The nonzero structure of A then holds the whole pattern, including the
/// deactivation diagonal, so assemblies can add into the CSR arrays of A
/// (AIJBlockAdder) instead of searching for, and inserting, every scalar
/// entry with MatSetValues.
void insert_sparsity_pattern(Mat A, const dolfinx::la::SparsityPattern& sp)
{
  const std::array bs = {sp.block_size(0), sp.block_size(1)};
  const std::int32_t num_owned_rows = sp.index_map(0)->size_local();
  const std::int64_t row_offset = sp.index_map(0)->local_range()[0];
  const std::vector<std::int64_t> global_cols = sp.column_indices();
  const auto [cols, offsets] = sp.graph();

  std::vector<PetscInt> rows(bs[0]);
  std::vector<PetscInt> row_cols;
  std::vector<PetscScalar> zeros;
  for (std::int32_t r = 0; r < num_owned_rows; ++r)
  {
    auto row = cols.subspan(offsets[r], offsets[r + 1] - offsets[r]);
    for (int i = 0; i < bs[0]; ++i)
      rows[i] = bs[0] * (row_offset + r) + i;
    row_cols.resize(row.size() * bs[1]);
    for (std::size_t k = 0; k < row.size(); ++k)
      for (int j = 0; j < bs[1]; ++j)
        row_cols[k * bs[1] + j] = bs[1] * global_cols[row[k]] + j;
    zeros.resize(rows.size() * row_cols.size(), 0);
    check_petsc(MatSetValues(A, rows.size(), rows.data(), row_cols.size(),
                             row_cols.data(), zeros.data(), INSERT_VALUES),
                "MatSetValues");
  }
  finalize_for_row_access(A);
}

/// Adds element matrices into the CSR arrays of an assembled MATSEQAIJ or
/// MATMPIAIJ matrix.
///
/// MatSetValues searches the row for every scalar entry. Here one search
/// locates a bs0 x bs1 block: the bs1 columns of a block are contiguous in a
/// row, and the bs0 rows of a block row share their columns. Rows owned by
/// other processes go through `fallback`. So does everything from the first
/// block outside the nonzero structure on, after the arrays are released,
/// since inserting new nonzeros reallocates them.
template <typename T>
class AIJBlockAdder
{
public:
  using set_fn_t = std::function<int(std::span<const std::int32_t>,
                                     std::span<const std::int32_t>,
                                     std::span<const T>)>;

  AIJBlockAdder(Mat A, std::array<int, 2> bs,
                const dolfinx::common::IndexMap& row_map,
                const dolfinx::common::IndexMap& col_map, set_fn_t fallback)
      : _bs(bs), _num_owned_rows(row_map.size_local()),
        _num_owned_cols(col_map.size_local()), _fallback(std::move(fallback))
  {
    PetscBool mpi = PETSC_FALSE;
    check_petsc(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(A),
                                       MATMPIAIJ, &mpi),
                "PetscObjectTypeCompare");
    Mat diag = A;
    Mat off_diag = nullptr;
    const PetscInt* garray = nullptr;
    if (mpi)
    {
      check_petsc(MatMPIAIJGetSeqAIJ(A, &diag, &off_diag, &garray),
                  "MatMPIAIJGetSeqAIJ");
    }

    _parts[0] = open(diag);
    _active = true;
    if (!off_diag)
      return;

    try
    {
      _parts[1] = open(off_diag);
      PetscInt m = 0;
      PetscInt num_cols = 0;
      check_petsc(MatGetSize(off_diag, &m, &num_cols), "MatGetSize");

      // Compressed off-diagonal column of the first component of each ghost
      // column block, or -1 if no owned row has entries in it.
      std::span<const std::int64_t> ghosts = col_map.ghosts();
      _ghost_cols.resize(ghosts.size(), -1);
      for (std::size_t k = 0; k < ghosts.size(); ++k)
      {
        const PetscInt col = _bs[1] * ghosts[k];
        const PetscInt* it = std::lower_bound(garray, garray + num_cols, col);
        if (it != garray + num_cols and *it == col)
          _ghost_cols[k] = it - garray;
      }
    }
    catch (...)
    {
      release();
      throw;
    }
  }

  AIJBlockAdder(const AIJBlockAdder&) = delete;
  AIJBlockAdder& operator=(const AIJBlockAdder&) = delete;

  ~AIJBlockAdder() { release(); }

  /// Add the element matrix `vals` for block rows `rows` and block columns
  /// `cols` (local indices), row-major of shape (bs0 rows, bs1 cols).
  int add(std::span<const std::int32_t> rows,
          std::span<const std::int32_t> cols, std::span<const T> vals)
  {
    if (!_active)
      return _fallback(rows, cols, vals);

    const int bs0 = _bs[0];
    const int bs1 = _bs[1];
    const std::size_t row_size = cols.size() * bs1;

    // Part (diagonal or off-diagonal block of A) and target column of the
    // first component of each column block, searched in increasing order.
    _col_part.resize(cols.size());
    _col_target.resize(cols.size());
    _order.resize(cols.size());
    for (std::size_t c = 0; c < cols.size(); ++c)
    {
      _order[c] = c;
      if (cols[c] < _num_owned_cols)
      {
        _col_part[c] = 0;
        _col_target[c] = bs1 * cols[c];
      }
      else if (std::size_t k = cols[c] - _num_owned_cols;
               k < _ghost_cols.size() and _ghost_cols[k] >= 0)
      {
        _col_part[c] = 1;
        _col_target[c] = _ghost_cols[k];
      }
      else
        return switch_to_fallback(rows, cols, vals);
    }
    std::ranges::sort(_order,
                      [this](std::size_t a, std::size_t b)
                      {
                        return std::pair(_col_part[a], _col_target[a])
                               < std::pair(_col_part[b], _col_target[b]);
                      });

    _pos.resize(bs0 * cols.size());
    for (std::size_t r = 0; r < rows.size(); ++r)
    {
      auto block_rows = rows.subspan(r);
      auto block_vals = vals.subspan(r * bs0 * row_size);
      if (rows[r] >= _num_owned_rows)
      {
        _fallback(block_rows.first(1), cols,
                  block_vals.first(bs0 * row_size));
        continue;
      }

      if (!locate(bs0 * rows[r]))
        return switch_to_fallback(block_rows, cols, block_vals);

      for (int i = 0; i < bs0; ++i)
      {
        const T* v = block_vals.data() + i * row_size;
        for (std::size_t c = 0; c < cols.size(); ++c)
        {
          T* a = _parts[_col_part[c]].values + _pos[i * cols.size() + c];
          for (int j = 0; j < bs1; ++j)
            a[j] += v[c * bs1 + j];
        }
      }
    }
    return 0;
  }

  /// Return the arrays to PETSc. Later values go through the fallback.
  ///
  /// MatSeqAIJRestoreArray marks the blocks as changed; the final assembly
  /// of A that callers perform after assembling does the same for A.
  void release()
  {
    if (!_active)
      return;
    _active = false;
    for (Part& part : _parts)
    {
      if (!part.mat)
        continue;
      PetscScalar* values = reinterpret_cast<PetscScalar*>(part.values);
      MatSeqAIJRestoreArray(part.mat, &values);
      PetscInt n = 0;
      PetscBool done = PETSC_FALSE;
      MatRestoreRowIJ(part.mat, 0, PETSC_FALSE, PETSC_FALSE, &n, &part.ia,
                      &part.ja, &done);
    }
  }

private:
  /// CSR arrays of the diagonal or off-diagonal SeqAIJ block of A.
  struct Part
  {
    Mat mat = nullptr;
    const PetscInt* ia = nullptr;
    const PetscInt* ja = nullptr;
    T* values = nullptr;
  };

  static Part open(Mat mat)
  {
    Part part{mat};
    PetscInt n = 0;
    PetscBool done = PETSC_FALSE;
    check_petsc(MatGetRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &n, &part.ia,
                            &part.ja, &done),
                "MatGetRowIJ");
    if (!done)
      throw std::runtime_error("PETSc did not provide the AIJ row structure");
    PetscScalar* values = nullptr;
    if (PetscErrorCode ierr = MatSeqAIJGetArray(mat, &values); ierr != 0)
    {
      MatRestoreRowIJ(mat, 0, PETSC_FALSE, PETSC_FALSE, &n, &part.ia, &part.ja,
                      &done);
      check_petsc(ierr, "MatSeqAIJGetArray");
    }
    part.values = reinterpret_cast<T*>(values);
    return part;
  }

  /// Fill _pos with the value offsets of all blocks of the block row whose
  /// first scalar row is `row`. Return false if a block is missing.
  bool locate(std::int32_t row)
  {
    const int bs0 = _bs[0];
    const int bs1 = _bs[1];
    const std::size_t num_cols = _order.size();
    std::array<PetscInt, 2> lower = {_parts[0].ia ? _parts[0].ia[row] : 0,
                                     _parts[1].ia ? _parts[1].ia[row] : 0};
    for (std::size_t c : _order)
    {
      const Part& part = _parts[_col_part[c]];
      const PetscInt target = _col_target[c];
      const PetscInt begin = part.ia[row];
      const PetscInt end = part.ia[row + 1];
      PetscInt& lo = lower[_col_part[c]];
      const PetscInt* it = std::lower_bound(part.ja + lo, part.ja + end, target);
      const PetscInt pos = it - part.ja;
      if (pos + bs1 > end or part.ja[pos] != target
          or part.ja[pos + bs1 - 1] != target + bs1 - 1)
      {
        return false;
      }
      // Not past the block: interior facet dofmaps list shared dofs twice.
      lo = pos;

      // The other rows of the block row have the same columns.
      for (int i = 0; i < bs0; ++i)
      {
        PetscInt p = part.ia[row + i] + (pos - begin);
        if (i > 0
            and (p + bs1 > part.ia[row + i + 1] or part.ja[p] != target
                 or part.ja[p + bs1 - 1] != target + bs1 - 1))
        {
          const PetscInt* row_end = part.ja + part.ia[row + i + 1];
          const PetscInt* jt
              = std::lower_bound(part.ja + part.ia[row + i], row_end, target);
          p = jt - part.ja;
          if (p + bs1 > part.ia[row + i + 1] or part.ja[p] != target
              or part.ja[p + bs1 - 1] != target + bs1 - 1)
          {
            return false;
          }
        }
        _pos[i * num_cols + c] = p;
      }
    }
    return true;
  }

  int switch_to_fallback(std::span<const std::int32_t> rows,
                         std::span<const std::int32_t> cols,
                         std::span<const T> vals)
  {
    release();
    return _fallback(rows, cols, vals);
  }

  std::array<int, 2> _bs;
  std::int32_t _num_owned_rows;
  std::int32_t _num_owned_cols;
  set_fn_t _fallback;
  std::array<Part, 2> _parts;
  std::vector<PetscInt> _ghost_cols;
  bool _active = false;

  // Per-element scratch
  std::vector<int> _col_part;
  std::vector<PetscInt> _col_target;
  std::vector<std::size_t> _order;
  std::vector<PetscInt> _pos;
};

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
        checked_set_fn(Aii, INSERT_VALUES),
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
          checked_set_fn(Aii, INSERT_VALUES),
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
    if (PetscErrorCode ierr = MatGetRow(A, global_row, &ncols, &cols, &values);
        ierr != 0)
    {
      dolfinx::la::petsc::error(ierr, __FILE__, "MatGetRow");
    }

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

  finalize_for_row_access(A);

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
      finalize_for_row_access(A_blocks[i][j]);

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

template <typename T, std::floating_point U>
void declare_runtime_petsc(nb::module_& m, std::string type)
{
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
        Mat A = dolfinx::la::petsc::create_matrix(form.mesh()->comm(), sp,
                                                  mat_type);
        if (is_host_aij(A))
          insert_sparsity_pattern(A, sp);
        else
          write_reserved_diagonal(A, form);
        return A;
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
        Mat A = dolfinx::la::petsc::create_matrix(form.mesh()->comm(), sp,
                                                  mat_type);
        if (is_host_aij(A))
          insert_sparsity_pattern(A, sp);
        else
          write_reserved_diagonal(A, form);
        return A;
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
            = {form.function_spaces().at(0)->dofmaps().front()->index_map_bs(),
               form.function_spaces().at(1)->dofmaps().front()->index_map_bs()};

        typename AIJBlockAdder<T>::set_fn_t set_fn;
        if (data_bs[0] == data_bs[1])
          set_fn = dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES);
        else
        {
          set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, data_bs[0], data_bs[1], ADD_VALUES);
        }

        // Add into the CSR arrays of AIJ matrices whose nonzero structure is
        // complete (see cutfemx.petsc.create_matrix).
        PetscBool assembled = PETSC_FALSE;
        check_petsc(MatAssembled(A, &assembled), "MatAssembled");
        const auto dofmap0 = form.function_spaces().at(0)->dofmaps().front();
        const auto dofmap1 = form.function_spaces().at(1)->dofmaps().front();
        if (assembled and is_host_aij(A) and dofmap0->bs() == data_bs[0]
            and dofmap1->bs() == data_bs[1])
        {
          AIJBlockAdder<T> adder(A, data_bs, *dofmap0->index_map,
                                 *dofmap1->index_map, std::move(set_fn));
          dolfinx_custom_data::fem::assemble_matrix(
              [&adder](std::span<const std::int32_t> rows,
                       std::span<const std::int32_t> cols,
                       std::span<const T> vals)
              { return adder.add(rows, cols, vals); },
              form, _bcs);
        }
        else
          dolfinx_custom_data::fem::assemble_matrix(set_fn, form, _bcs);
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
                          std::span<const T>)>
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
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> beta_cell_values,
         int quadrature_degree)
      {
        if (!V)
          throw std::runtime_error("Received a null function space.");
        std::function<int(std::span<const std::int32_t>,
                          std::span<const std::int32_t>,
                          std::span<const T>)>
            mat_add = dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES);
        cutfemx::extensions::assemble_extension_penalty(
            mat_add, *V, cut_data, aggregation,
            std::span<const T>(beta_cell_values.data(),
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
            checked_set_fn(A, INSERT_VALUES), V, _bcs,
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
            checked_set_fn(A, INSERT_VALUES),
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
              checked_set_fn(A, INSERT_VALUES),
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
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  declare_runtime_petsc<std::complex<float>, float>(petsc_mod, "complex64");
  declare_runtime_petsc<std::complex<float>, double>(petsc_mod,
                                                     "complex64_float64");
#else
  declare_runtime_petsc<std::complex<double>, float>(petsc_mod,
                                                     "complex128_float32");
  declare_runtime_petsc<std::complex<double>, double>(petsc_mod, "complex128");
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  declare_runtime_petsc<float, float>(petsc_mod, "float32");
  declare_runtime_petsc<float, double>(petsc_mod, "float32_float64");
#else
  declare_runtime_petsc<double, float>(petsc_mod, "float64_float32");
  declare_runtime_petsc<double, double>(petsc_mod, "float64");
#endif
#endif
}
} // namespace cutfemx_wrappers

#endif
