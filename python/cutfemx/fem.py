# Copyright (c) 2024 ONERA
# Authors: Susanne Claus
# This file is part of CutFEMx
#
# SPDX-License-Identifier:    MIT
import sys
import numpy as np
import numpy.typing as npt
import typing
import collections
import functools
from itertools import chain

from cutfemx import cutfemx_cpp as _cpp
from cutfemx.mesh import CutMesh

import ufl
from dolfinx import default_scalar_type, jit
from dolfinx.fem import form_cpp_class, Form, form, FunctionSpace
from dolfinx.fem import IntegralType as dIntegralType

import dolfinx.cpp as dcpp
from mpi4py import MPI

from dolfinx.fem.assemble import create_vector
from dolfinx import la
from dolfinx.mesh import Mesh, MeshTags
from dolfinx.fem.bcs import DirichletBC

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
  "assemble_scalar",
  "assemble_vector",
  "assemble_matrix",
  "create_sparsity_pattern",
  "create_matrix",
  "deactivate"

]

_ufl_to_dolfinx_domain = {
    "cell": dIntegralType.cell,
    "exterior_facet": dIntegralType.exterior_facet,
    "interior_facet": dIntegralType.interior_facet,
    "vertex": dIntegralType.vertex,
}

_ufl_to_cutfemx_domain = {
    "cutcell": _cpp.fem.IntegralType.cutcell,
    "interface": _cpp.fem.IntegralType.interface,
}

class ElementSourceType:
    """Type of element source for tabulation."""
    ARGUMENT = 0
    COEFFICIENT = 1
    GEOMETRY = 2

def extract_integral_element_sources(ufcx_form, ufl_form, module):
    """
    Extract per-integral element sources from UFCX.
    
    For each runtime integral, identifies which elements are needed for
    tabulation and where they come from (argument, coefficient, geometry).
    
    Returns:
        dict: {(integral_type, integral_id): [(source_type, index, component, deriv_order), ...]}
    """
    result = {}
    
    # UFCX integral type offsets:
    # 0=cell, 1=exterior_facet, 2=interior_facet, 3=cutcell (custom/vertex slot), 4=interface
    # NOTE: Standard FFCx might place custom integrals in slot 3 or 4 depending on version.
    # Observed trace shows cutcell at slot 3.
    CUTCELL_OFFSET = 3
    INTERFACE_OFFSET = 4
    
    # Get integral counts
    offsets = [ufcx_form.form_integral_offsets[i] for i in range(7)]
    num_cutcell = offsets[CUTCELL_OFFSET + 1] - offsets[CUTCELL_OFFSET]
    num_interface = offsets[INTERFACE_OFFSET + 1] - offsets[INTERFACE_OFFSET]
    
    # Build hash-to-source lookup table
    hash_to_source = _build_hash_to_source_map(ufl_form)
    
    # Process cutcell integrals
    for i in range(num_cutcell):
        idx = offsets[CUTCELL_OFFSET] + i
        integral = ufcx_form.form_integrals[idx]
        integral_id = ufcx_form.form_integral_ids[idx]
        
        sources = _extract_sources_from_integral(integral, hash_to_source)
        result[(_cpp.fem.IntegralType.cutcell, integral_id)] = sources
    

    # Process interface integrals (legacy expect at 5)
    for i in range(num_interface):
        idx = offsets[INTERFACE_OFFSET] + i
        integral = ufcx_form.form_integrals[idx]
        integral_id = ufcx_form.form_integral_ids[idx]
        
        sources = _extract_sources_from_integral(integral, hash_to_source)
        result[(_cpp.fem.IntegralType.interface, integral_id)] = sources
    
    return result

def _extract_sources_from_integral(integral, hash_to_source):
    """Extract element sources for a single integral."""
    sources = []
    seen = set()
    
    # Debug prints
    
    
    for j in range(integral.num_fe):
        hash_val = integral.finite_element_hashes[j]
        deriv_order = integral.finite_element_deriv_order[j]
        
        if hash_val in hash_to_source:
            src_type, idx, comp = hash_to_source[hash_val]
            key = (src_type, idx, comp, deriv_order)
            if key not in seen:
                 sources.append(key)
                 seen.add(key)
        else:

            # Fallback logic:
            # - deriv_order=0 and not found likely means geometry/coordinate element
            # - Otherwise, use first argument as fallback
            if deriv_order == 0:
                # Likely a geometry/coordinate element
                key = (ElementSourceType.GEOMETRY, 0, -1, deriv_order)
            else:
                # Fallback to first argument
                key = (ElementSourceType.ARGUMENT, 0, -1, deriv_order)
            if key not in seen:
                 sources.append(key)
                 seen.add(key)
    sys.stdout.flush()
    return sources

def _build_hash_to_source_map(ufl_form):
    """Build a mapping from element hash to (source_type, index, component)."""
    hash_map = {}
    
    
    
    # Arguments (function spaces)
    for i, arg in enumerate(ufl_form.arguments()):
        element = arg.ufl_function_space().ufl_element()

        _add_element_hashes(hash_map, element, ElementSourceType.ARGUMENT, i)
    
    # Coefficients
    for i, coeff in enumerate(ufl_form.coefficients()):
        element = coeff.ufl_function_space().ufl_element()
        _add_element_hashes(hash_map, element, ElementSourceType.COEFFICIENT, i)
    
    # Geometry (coordinate element from domain)
    domain = ufl_form.ufl_domain()
    if domain is not None:
        coord_elem = domain.ufl_coordinate_element()
        if coord_elem is not None:

             # Force component -1 for GEOMETRY to ensure full block extraction
             _add_element_hashes(hash_map, coord_elem, ElementSourceType.GEOMETRY, 0, component=-1)

    

    return hash_map

def _add_element_hashes(hash_map, element, source_type, index, component=-1):
    """Add element and its sub-elements to the hash map recursively."""
    from basix.ufl import _BasixElement
    
    # Helper to extract hash
    def get_hash(el):
        if hasattr(el, 'basix_hash'):
            # Check if it's callable
            if callable(el.basix_hash):
                return el.basix_hash()
            return el.basix_hash
        if hasattr(el, 'basix_element'):
             return el.basix_element.hash()
        if hasattr(el, '_basix_element'):
             return el._basix_element.hash()
        return None

    # Get the basix element if available
    h = get_hash(element)
    if h is not None:
        hash_map[h] = (source_type, index, component)
    
    # Handle elements with sub-elements (mixed, blocked, etc.)
    if hasattr(element, 'sub_elements') and len(element.sub_elements) > 0:
        for comp, sub_el in enumerate(element.sub_elements):
            # Recursively process sub-element
            # For top-level mixed elements, use comp as the component
            # For nested elements (like BlockedElement), inherit the parent component
                new_comp = comp if component == -1 else component
                _add_element_hashes(hash_map, sub_el, source_type, index, new_comp)

def _compute_basix_hash(cmap):
    """Compute hash for coordinate element."""
    try:
        import basix
        from basix import create_element
        
        # Dolfinx CoordinateElement (from cpp) exposes these properties
        degree = cmap.degree
        cell_shape = cmap.cell_shape
        variant = cmap.variant
        
        # Create the basix element with the same properties
        e = create_element(
            basix.ElementFamily.P,
            cell_shape,
            degree,
            variant,
            basix.DPCVariant.unset,
            False,
            basix.element.dtype.float64
        )
        return e.hash()
    except Exception as ex:
        import sys
        print(f"DEBUG: Failed to compute coord hash: {ex}", file=sys.stdout)
        return None


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

    def update_integration_domains(
      self,
      subdomains: dict[str, list[tuple[int, np.ndarray]]],
      entity_maps: typing.Optional[dict[Mesh, np.typing.NDArray[np.int32]]] = None
  ):
      sd = {}

      for itg_type, domains in subdomains.items():
          sd[_ufl_to_dolfinx_domain[itg_type]] = []
          for id, entities in domains:
              sd[_ufl_to_dolfinx_domain[itg_type]].append((id,entities))

      if entity_maps is None:
          _entity_maps = dict()
      else:
          _entity_maps = {msh._cpp_object: emap for (msh, emap) in entity_maps.items()}

      #subdomains : ufl typed subdomain entity data
      self._cpp_object.update_integration_domains(sd,_entity_maps)

      #update cpp object in python form
      self._form._cpp_object = self._cpp_object.form

    def update_runtime_domains(
      self,
      subdomains: dict[str, list[tuple[int, typing.Union[_cpp.quadrature.QuadratureRules_float32, _cpp.quadrature.QuadratureRules_float64]]]]
    ):

      sd = {}

      for itg_type, domains in subdomains.items():
          sd[_ufl_to_cutfemx_domain[itg_type]] = []
          for id, quad_rule in domains:
              sd[_ufl_to_cutfemx_domain[itg_type]].append((id,quad_rule))

      self._cpp_object.update_runtime_domains(sd)

def get_integration_domains(
    integral_type: dIntegralType,
    subdomain: typing.Optional[typing.Union[MeshTags, list[tuple[int, np.ndarray]]]],
    subdomain_ids: list[int],
) -> list[tuple[int, np.ndarray]]:
    if subdomain is None:
        return []
    else:
        domains = []
        try:
            if integral_type in (dIntegralType.exterior_facet, dIntegralType.interior_facet):
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

        runtime_types = ["cutcell", "interface"]

        for integral in integrals:
            if integral.integral_type() in runtime_types:
              runtime_sd[_ufl_to_cutfemx_domain[integral.integral_type()]] = []
            else:
              sd[_ufl_to_dolfinx_domain[integral.integral_type()]] = []

        #split integrals into runtime and standard
        for integral in integrals:
            if integral.integral_type() in runtime_types:
              runtime_sd[_ufl_to_cutfemx_domain[integral.integral_type()]].append(integral.subdomain_data())
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
            if integral.integral_type() in runtime_types:
              subdomain_ids_runtime[_ufl_to_cutfemx_domain[integral.integral_type()]] = []
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

                if integral.integral_type() in runtime_types:
                  subdomain_ids_runtime[_ufl_to_cutfemx_domain[integral.integral_type()]].append(ids)
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

        # print("subdomain ids runtime=", subdomain_ids_runtime)

        # Subdomain markers (possibly empty list for some integral
        # types)
        subdomains = {}
        runtime_subdomains = {}
        treated_ids = set()

        for itg_type, ids in subdomain_ids.items():
            subdomains[itg_type] = get_integration_domains(
                itg_type, sd[itg_type][0], ids)

        #order runtime subdomains
        for itg_type, ids in subdomain_ids_runtime.items():
            runtime_subdomains[itg_type] = []
            for id in ids:
              for s in runtime_sd[itg_type][0]:
                if s[0] == id:
                    if id not in treated_ids:
                      treated_ids.add(id)
                      runtime_subdomains[itg_type].append((s[0], s[1]))

        # print("runtime_subdomains=", runtime_subdomains)
        # print("subdomains=", subdomains)

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

        
        # Extract element sources from UFCX for runtime integrals
        element_sources_py = extract_integral_element_sources(ufcx_form, form, module)

        
        # Convert to C++ ElementSource objects
        # Convert to C++ ElementSource objects
        element_sources_cpp = {}
        for (itg_type, itg_id), sources in element_sources_py.items():

            cpp_sources = []
            for (src_type, idx, comp, deriv) in sources:

                cpp_src = _cpp.fem.ElementSource(
                    _cpp.fem.ElementSourceType(src_type),
                    idx, comp, deriv
                )
                cpp_sources.append(cpp_src)
            element_sources_cpp[(itg_type, itg_id)] = cpp_sources
        


        ufcx_address = module.ffi.cast("uintptr_t", module.ffi.addressof(ufcx_form))
        cutf = cut_ftype(ufcx_address,
                         f,
                         runtime_subdomains,
                         element_sources_cpp)

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

def assemble_scalar(M: CutForm, constants=None, coeffs=None, coeffs_rt=None):
    constants = constants or _pack_constants(M._form._cpp_object)
    coeffs = coeffs or _pack_coefficients(M._form._cpp_object)
    coeffs_rt = coeffs_rt or _cpp.fem.pack_coefficients(M._cpp_object)

    return _cpp.fem.assemble_scalar(M._cpp_object, constants, coeffs, coeffs_rt)


@functools.singledispatch
def assemble_vector(L: typing.Any, constants=None, coeffs=None, coeffs_rt=None):
    return _assemble_vector_form(L, constants, coeffs, coeffs_rt)


@assemble_vector.register(CutForm)
def _assemble_vector_form(L: CutForm, constants=None, coeffs=None, coeffs_rt=None) -> la.Vector:
    b = create_vector(L._form)
    b.array[:] = 0
    constants = constants or _pack_constants(L._form._cpp_object)
    coeffs = coeffs or _pack_coefficients(L._form._cpp_object)
    coeffs_rt = coeffs_rt or _cpp.fem.pack_coefficients(L._cpp_object)
    _assemble_vector_array(b.array, L, constants, coeffs, coeffs_rt)
    return b

@assemble_vector.register(np.ndarray)
def _assemble_vector_array(b: np.ndarray, L: CutForm, constants=None, coeffs=None, coeffs_rt=None):
    constants = _pack_constants(L._form._cpp_object) if constants is None else constants
    coeffs = _pack_coefficients(L._form._cpp_object) if coeffs is None else coeffs
    coeffs_rt = _cpp.fem.pack_coefficients(L._cpp_object) if coeffs_rt is None else coeffs_rt
    _cpp.fem.assemble_vector(b, L._cpp_object, constants, coeffs, coeffs_rt)
    return b

def create_sparsity_pattern(a: CutForm):
    return _cpp.fem.create_sparsity_pattern(a._cpp_object)

def create_matrix(a: CutForm, block_mode: typing.Optional[la.BlockMode] = None) -> la.MatrixCSR:
    sp = create_sparsity_pattern(a)
    sp.finalize()
    if block_mode is not None:
        return la.matrix_csr(sp, block_mode=block_mode, dtype=a.dtype)
    else:
        return la.matrix_csr(sp, dtype=a.dtype)

@functools.singledispatch
def assemble_matrix(
    a: typing.Any,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
    coeffs_rt=None,
    block_mode: typing.Optional[la.BlockMode] = None,
):
    bcs = [] if bcs is None else bcs
    A: la.MatrixCSR = create_matrix(a, block_mode)
    _assemble_matrix_csr(A, a, bcs, diagonal, constants, coeffs, coeffs_rt)
    return A


@assemble_matrix.register
def _assemble_matrix_csr(
    A: la.MatrixCSR,
    a: CutForm,
    bcs: typing.Optional[list[DirichletBC]] = None,
    diagonal: float = 1.0,
    constants=None,
    coeffs=None,
    coeffs_rt=None
) -> la.MatrixCSR:
    bcs = [] if bcs is None else [bc._cpp_object for bc in bcs]
    constants = _pack_constants(a._form._cpp_object) if constants is None else constants
    coeffs = _pack_coefficients(a._form._cpp_object) if coeffs is None else coeffs
    coeffs_rt = _cpp.fem.pack_coefficients(a._cpp_object) if coeffs_rt is None else coeffs_rt

    _cpp.fem.assemble_matrix(A._cpp_object, a._cpp_object, constants, coeffs, coeffs_rt, bcs)

    # If matrix is a 'diagonal'block, set diagonal entry for constrained
    # dofs
    if a._form.function_spaces[0] is a._form.function_spaces[1]:
        dcpp.fem.insert_diagonal(A._cpp_object, a._form.function_spaces[0], bcs, diagonal)
    return A

def deactivate(A: la.MatrixCSR, deactivate_domain: str, level_set: Function , V: typing.Any, diagonal: float = 1.0) -> Function:
      if(len(V)==1):
        component = [-1]
        xi = _cpp.fem.deactivate(A._cpp_object, deactivate_domain, level_set._cpp_object, V[0]._cpp_object, component, diagonal)
        return xi
      else:
        component = [V[1]]
        xi = _cpp.fem.deactivate(A._cpp_object, deactivate_domain, level_set._cpp_object, V[0]._cpp_object, component, diagonal)
        return xi