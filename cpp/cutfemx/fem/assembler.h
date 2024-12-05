// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once

#include "../quadrature/quadrature.h"
#include "CutForm.h"

#include "typedefs.h"
#include "assemble_scalar.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>

#include <basix/finite-element.h>

#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Constant.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace cutfemx::fem
{
    template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  T assemble_scalar(
      const CutForm<T, U>& M, std::span<const T> constants,
      const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                    std::pair<std::span<const T>, int>>& coefficients)
  {
    std::shared_ptr<const dolfinx::mesh::Mesh<U>> mesh = M._form->mesh();
    assert(mesh);

    T val = dolfinx::fem::impl::assemble_scalar<T,U>(*M._form, mesh->geometry().dofmap(),
                                  mesh->geometry().x(), constants, coefficients);
    T val_cut = assemble_scalar(M, mesh->geometry().dofmap(),
                                  mesh->geometry().x(), constants, coefficients);
    val += val_cut;
    return val;
  }

    template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
  T assemble_scalar(const CutForm<T, U>& M)
  {
    const std::vector<T> constants = dolfinx::fem::pack_constants(*M._form);
    auto coefficients = dolfinx::fem::allocate_coefficient_storage(*M._form);
    dolfinx::fem::pack_coefficients(*M._form, coefficients);

    T val = assemble_scalar(M, std::span(constants),
                          dolfinx::fem::make_coefficients_span(coefficients));

    return val;
  }

} // namespace cutfemx::fem
