# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing
import collections
import functools
from itertools import chain

from petsc4py import PETSc

from cutfemx import cutfemx_cpp as _cpp
from cutfemx.fem import _assemble_vector_array
from cutfemx.fem import CutForm

import dolfinx.cpp as dcpp
from mpi4py import MPI

from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem import Function

from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants

from dolfinx.la import create_petsc_vector

__all__ = [
  "create_matrix",
  "assemble_vector",
  "assemble_matrix",
  "deactivate"
]

def create_matrix(a: CutForm, mat_type=None) -> PETSc.Mat:
    if mat_type is None:
        return _cpp.fem.petsc.create_matrix(a._cpp_object)
    else:
        return _cpp.fem.petsc.create_matrix(a._cpp_object, mat_type)

@functools.singledispatch
def assemble_vector(L: typing.Any, constants=None, coeffs=None, coeffs_rt=None) -> PETSc.Vec:
    b = create_petsc_vector(
        L.function_spaces[0].dofmap.index_map, L.function_spaces[0].dofmap.index_map_bs
    )
    with b.localForm() as b_local:
        _assemble_vector_array(b_local.array_w, L, constants, coeffs, coeffs_rt)
    return b

@functools.singledispatch
def assemble_matrix(
    a: typing.Any, bcs: list[DirichletBC] = [], diagonal: float = 1.0, constants=None, coeffs=None, coeffs_rt=None
):
    A = _cpp.fem.petsc.create_matrix(a._cpp_object)
    assemble_matrix_mat(A, a, bcs, diagonal, constants, coeffs, coeffs_rt)
    return A


@assemble_matrix.register
def assemble_matrix_mat(
    A: PETSc.Mat,
    a: CutForm,
    bcs: list[DirichletBC] = [],
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
    coeffs_rt=None,
) -> PETSc.Mat:
    constants = _pack_constants(a._form._cpp_object) if constants is None else constants
    coeffs = _pack_coefficients(a._form._cpp_object) if coeffs is None else coeffs
    coeffs_rt = _cpp.fem.pack_coefficients(a._cpp_object) if coeffs_rt is None else coeffs_rt
    _bcs = [bc._cpp_object for bc in bcs]
    _cpp.fem.petsc.assemble_matrix(A, a._cpp_object, constants, coeffs, coeffs_rt, _bcs)
    if a.function_spaces[0] is a.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dcpp.fem.petsc.insert_diagonal(A, a.function_spaces[0], _bcs, diagonal)
    return A

def deactivate(A: PETSc.Mat, deactivate_domain: str, level_set: Function , V: typing.Any, diagonal: float = 1.0) -> Function:
      if(len(V)==1):
        component = [-1]
        xi = _cpp.fem.petsc.deactivate(A, deactivate_domain, level_set._cpp_object, V[0]._cpp_object, component, diagonal)
        return xi
      else:
        component = [V[1]]
        xi = _cpp.fem.petsc.deactivate(A, deactivate_domain, level_set._cpp_object, V[0]._cpp_object, component, diagonal)
        return xi