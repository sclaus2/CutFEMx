// Copyright (c) 2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

#pragma once

#ifdef HAS_PETSC

#include "CutForm.h"
#include "assembler.h"

#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>

#include <concepts>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscvec.h>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace cutfemx::fem
{
namespace petsc
{

  template <std::floating_point T>
  Mat create_matrix(const CutForm<PetscScalar, T>& a,
                    std::string type = std::string())
  {
    dolfinx::la::SparsityPattern pattern = cutfemx::fem::create_sparsity_pattern(a);
    pattern.finalize();
    return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
  }

  template <std::floating_point T>
  void assemble_vector(
      Vec b, const CutForm<PetscScalar, T>& L,
      std::span<const PetscScalar> constants,
      const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                    std::pair<std::span<const PetscScalar>, int>>& coeffs,
      const std::map<std::pair<cutfemx::fem::IntegralType, int>,
                    std::pair<std::span<const PetscScalar>, int>>& coeffs_rt)
  {
    Vec b_local;
    VecGhostGetLocalForm(b, &b_local);
    PetscInt n = 0;
    VecGetSize(b_local, &n);
    PetscScalar* array = nullptr;
    VecGetArray(b_local, &array);
    std::span<PetscScalar> _b(array, n);
    assemble_vector(_b, L, constants, coeffs, coeffs_rt);
    VecRestoreArray(b_local, &array);
    VecGhostRestoreLocalForm(b, &b_local);
  }


  template <std::floating_point T>
  void assemble_vector(Vec b, const CutForm<PetscScalar, T>& L)
  {
    Vec b_local;
    VecGhostGetLocalForm(b, &b_local);
    PetscInt n = 0;
    VecGetSize(b_local, &n);
    PetscScalar* array = nullptr;
    VecGetArray(b_local, &array);
    std::span<PetscScalar> _b(array, n);
    assemble_vector(_b, L);
    VecRestoreArray(b_local, &array);
    VecGhostRestoreLocalForm(b, &b_local);
  }

} // namespace petsc
} // namespace cutfemx::fem

#endif
