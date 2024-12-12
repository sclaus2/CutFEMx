# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import numpy as np
import numpy.typing as npt
import typing
import collections
from itertools import chain

from cutfemx import cutfemx_cpp as _cpp
from cutfemx.mesh import CutMesh

import ufl
from dolfinx import default_scalar_type, jit
from dolfinx.fem import form_cpp_class, Form, form, FunctionSpace
from dolfinx.fem import IntegralType

import dolfinx.cpp as dcpp
from mpi4py import MPI

from dolfinx.fem import function
from dolfinx.mesh import Mesh, MeshTags

from dolfinx.mesh import Mesh
from dolfinx.fem import Function
from dolfinx.cpp.fem import pack_coefficients as _pack_coefficients
from dolfinx.cpp.fem import pack_constants as _pack_constants

__all__ = [
  "CutForm",
  "get_integration_domains",
  "cut_form_cpp_class",
  "cut_form",
  "cut_function",
  "assemble_scalar"
]

_ufl_to_dolfinx_domain = {
    "cell": IntegralType.cell,
    "exterior_facet": IntegralType.exterior_facet,
    "interior_facet": IntegralType.interior_facet,
    "vertex": IntegralType.vertex,
}

class CutForm:
    _cpp_object: typing.Union[
        _cpp.fem.CutForm_float32,
        _cpp.fem.CutForm_float64,
    ]

    def __init__(
        self,
        cut_form: typing.Union[
            _cpp.fem.CutForm_float32,
            _cpp.fem.CutForm_float64],
        form: Form
    ):
        self._cpp_object = cut_form
        self._form = form

    @property
    def rank(self) -> int:
        return self._form._cpp_object.rank  # type: ignore

    @property
    def function_spaces(self) -> list[FunctionSpace]:
        """Function spaces on which this form is defined."""
        return self._form._cpp_object.function_spaces  # type: ignore

    @property
    def dtype(self) -> np.dtype:
        """Scalar type of this form."""
        return np.dtype(self._form._cpp_object.dtype)

    @property
    def mesh(self) -> typing.Union[dcpp.mesh.Mesh_float32, dcpp.mesh.Mesh_float64]:
        """Mesh on which this form is defined."""
        return self._form._cpp_object.mesh

    @property
    def integral_types(self):
        """Integral types in the form."""
        return self._cpp_object.integral_types

def get_integration_domains(
    integral_type: IntegralType,
    subdomain: typing.Optional[typing.Union[MeshTags, list[tuple[int, np.ndarray]]]],
    subdomain_ids: list[int],
) -> list[tuple[int, np.ndarray]]:
    if subdomain is None:
        return []
    else:
        domains = []
        try:
            if integral_type in (IntegralType.exterior_facet, IntegralType.interior_facet):
                tdim = subdomain.topology.dim  # type: ignore
                subdomain._cpp_object.topology.create_connectivity(tdim - 1, tdim)  # type: ignore
                subdomain._cpp_object.topology.create_connectivity(tdim, tdim - 1)  # type: ignore
            # Compute integration domains only for each subdomain id in
            # the integrals
            # If a process has no integral entities, insert an empty
            # array
            for id in subdomain_ids:
                integration_entities = dcpp.fem.compute_integration_domains(
                    integral_type,
                    subdomain._cpp_object.topology,  # type: ignore
                    subdomain.find(id),  # type: ignore
                    subdomain.dim,  # type: ignore
                )
                domains.append((id, integration_entities))
            return [(s[0], np.array(s[1])) for s in domains]
        except AttributeError:
            return [(s[0], np.array(s[1])) for s in subdomain]  # type: ignore

def cut_form_cpp_class(
    dtype: npt.DTypeLike,
) -> typing.Union[
    _cpp.fem.CutForm_float32, _cpp.fem.CutForm_float64
]:
    if np.issubdtype(dtype, np.float32):
        return _cpp.fem.CutForm_float32
    elif np.issubdtype(dtype, np.float64):
        return _cpp.fem.CutForm_float64
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

def cut_form(
    form: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    dtype: npt.DTypeLike = default_scalar_type,
    form_compiler_options: typing.Optional[dict] = None,
    jit_options: typing.Optional[dict] = None,
    entity_maps: typing.Optional[dict[Mesh, np.typing.NDArray[np.int32]]] = None,
):
    if form_compiler_options is None:
        form_compiler_options = dict()

    form_compiler_options["scalar_type"] = dtype
    ftype = form_cpp_class(dtype)
    cut_ftype = cut_form_cpp_class(dtype)

    def _cut_form(form):
        """Compile a single UFL form."""
        # Extract subdomain data from UFL form
        integrals = form.integrals()
        domain = form.ufl_domain()  # Assuming single domain

        runtime_sd = {}
        sd = {}

        for integral in integrals:
            scheme = integral.metadata().get("quadrature_rule")
            if scheme == "runtime":
              runtime_sd[_ufl_to_dolfinx_domain[integral.integral_type()]] = []
            else:
              sd[_ufl_to_dolfinx_domain[integral.integral_type()]] = []

        #split integrals into runtime and standard
        for integral in integrals:
            scheme = integral.metadata().get("quadrature_rule")
            if scheme == "runtime":
              runtime_sd[_ufl_to_dolfinx_domain[integral.integral_type()]].append(integral.subdomain_data())
            else:
                sd[_ufl_to_dolfinx_domain[integral.integral_type()]].append(integral.subdomain_data())

        # print("runtime_sd=", runtime_sd)
        # print("sd=", sd)
        # print("domain=", domain)

        mesh = domain.ufl_cargo()
        if mesh is None:
            raise RuntimeError("Expecting to find a Mesh in the form.")
        ufcx_form, module, code = jit.ffcx_jit(
            mesh.comm, form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )

        # For each argument in form extract its function space
        V = [arg.ufl_function_space()._cpp_object for arg in form.arguments()]

        # Prepare coefficients data. For every coefficient in form take
        # its C++ object.
        original_coeffs = form.coefficients()
        coeffs = [
            original_coeffs[ufcx_form.original_coefficient_positions[i]]._cpp_object
            for i in range(ufcx_form.num_coefficients)
        ]
        constants = [c._cpp_object for c in form.constants()]

        # Make map from integral_type to subdomain id
        subdomain_ids = {}
        subdomain_ids_runtime = {}
        for integral in integrals:
            scheme = integral.metadata().get("quadrature_rule")
            if scheme == "runtime":
              subdomain_ids_runtime[_ufl_to_dolfinx_domain[integral.integral_type()]] = []
            else:
              subdomain_ids[_ufl_to_dolfinx_domain[integral.integral_type()]] = []

        for integral in form.integrals():
            if integral.subdomain_data() is not None:
                # Subdomain ids can be strings, its or tuples with
                # strings and ints
                if integral.subdomain_id() != "everywhere":
                    try:
                        ids = [sid for sid in integral.subdomain_id() if sid != "everywhere"]
                    except TypeError:
                        # If not tuple, but single integer id
                        ids = [integral.subdomain_id()]
                else:
                    ids = []

                scheme = integral.metadata().get("quadrature_rule")
                if scheme == "runtime":
                  subdomain_ids_runtime[_ufl_to_dolfinx_domain[integral.integral_type()]].append(ids)
                else:
                  subdomain_ids[_ufl_to_dolfinx_domain[integral.integral_type()]].append(ids)

        # Chain and sort subdomain ids
        # used to extract entities from meshtags
        for itg_type, marker_ids in subdomain_ids.items():
            flattened_ids = list(chain.from_iterable(marker_ids))
            flattened_ids.sort()
            subdomain_ids[itg_type] = flattened_ids

        for itg_type, marker_ids in subdomain_ids_runtime.items():
            flattened_ids = list(chain.from_iterable(marker_ids))
            flattened_ids.sort()
            subdomain_ids_runtime[itg_type] = flattened_ids

        print("subdomain ids runtime=", subdomain_ids_runtime)

        # Subdomain markers (possibly empty list for some integral
        # types)
        subdomains = {}
        runtime_subdomains = {}

        for itg_type, ids in subdomain_ids.items():
            subdomains[itg_type] = get_integration_domains(
                itg_type, sd[itg_type][0], ids)

        #order runtime subdomains
        for itg_type, ids in subdomain_ids_runtime.items():
            runtime_subdomains[itg_type] = []
            for id in ids:
              for s in runtime_sd[itg_type][0]:
                if s[0] == id:
                    runtime_subdomains[itg_type].append((s[0], s[1]))

        print("runtime_subdomains=", runtime_subdomains)
        print("subdomains=", subdomains)

        if entity_maps is None:
            _entity_maps = dict()
        else:
            _entity_maps = {msh._cpp_object: emap for (msh, emap) in entity_maps.items()}

        f = ftype(
            module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form)),
            V,
            coeffs,
            constants,
            subdomains,
            _entity_maps,
            mesh,
        )
        form_py = Form(f, ufcx_form, code, module)

        ufcx_address = module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form))
        cutf = cut_ftype(ufcx_address,
                         f,
                         runtime_subdomains)

        return CutForm(cutf, form_py)


    def _create_cut_form(form):
        if isinstance(form, ufl.Form):
            if form.empty():
                return None
            else:
                return _cut_form(form)
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _create_cut_form(sub_form), form))
        else:
            return form

    return _create_cut_form(form)

def cut_function(
        u: Function,
        sub_mesh: CutMesh)->Function:
      return _cpp.fem.create_cut_function(u._cpp_object,sub_mesh._cpp_object)

def assemble_scalar(M: CutForm, constants=None, coeffs=None, coeffs_rt=None):
    constants = constants or _pack_constants(M._form._cpp_object)
    coeffs = coeffs or _pack_coefficients(M._form._cpp_object)
    coeffs_rt = coeffs_rt or _cpp.fem.pack_coefficients(M._cpp_object)

    return _cpp.fem.assemble_scalar(M._cpp_object, constants, coeffs, coeffs_rt)


