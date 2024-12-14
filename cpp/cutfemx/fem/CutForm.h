// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#pragma once


#include "../quadrature/quadrature.h"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <ufcx.h>

#include <basix/finite-element.h>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/traits.h>

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_map>

namespace cutfemx::fem
{
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
struct runtime_integral_data
{
  template <typename K, typename W>
    requires std::is_convertible_v<
                 std::remove_cvref_t<K>,
                 std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const uint8_t*,
                                    const int*, const U*, const U*,
                                    const T*, const size_t*)>>
                 and std::is_convertible_v<std::remove_cvref_t<W>,
                                           std::vector<int>>
  runtime_integral_data(int id, K&& kernel, std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules,
                std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>> elements,
                W&& coeffs)
      : id(id), kernel(std::forward<K>(kernel)),
        quadrature_rules(quadrature_rules),
        elements(elements),
        coeffs(std::forward<W>(coeffs))
  {
  }

  /// @brief Integral ID.
  int id;

  /// @brief The integration kernel.
  std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const uint8_t*,
                                    const int*, const U*, const U*,
                                    const T*, const size_t*)>
      kernel;

  /// @brief The quadrature rule to integrate over (contains parent map to mesh).
  std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules;

  /// @brief basix elements used in integral and their maximum derivative
  /// this is used to tabulate values in the assembler over runtime integrals
  std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>> elements;

  /// @brief Indices of coefficients (from the form) that are in this
  /// integral.
  std::vector<int> coeffs;
};

template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class CutForm
{
public:
  /// Scalar type
  using scalar_type = T;

  /// Geometry type
  using geometry_type = U;

  /// @brief Create a cut finite element form from standard form and runtime integral data.
  template <typename X>
    requires std::is_convertible_v<
                 std::remove_cvref_t<X>,
                 std::map<dolfinx::fem::IntegralType, std::vector<runtime_integral_data<
                                            scalar_type, geometry_type>>>>
  CutForm(std::shared_ptr<const dolfinx::fem::Form<scalar_type, geometry_type>> form,
      X&& integrals)
  {
    _form = form;

    // Store kernels, looping over integrals by domain type (dimension)
    for (auto&& [domain_type, data] : integrals)
    {
      if (!std::ranges::is_sorted(data,
                                  [](auto& a, auto& b) { return a.id < b.id; }))
      {
        throw std::runtime_error("Integral IDs not sorted");
      }

      std::vector<runtime_integral_data<scalar_type, geometry_type>>& itg
          = _integrals[static_cast<std::size_t>(domain_type)];
      for (auto&& [id, kern, q, e, c] : data)
        itg.emplace_back(id, kern, q, std::move(e), std::move(c));
    }
  }

  /// Copy constructor
  CutForm(const CutForm& cut_form) = delete;

  /// Move constructor
  CutForm(CutForm&& cut_form) = default;

  /// Destructor
  virtual ~CutForm() = default;

  /// @brief Rank of the form.
  ///
  /// bilinear form = 2, linear form = 1, functional = 0, etc.
  ///
  /// @return The rank of the form
  int rank() const { return _form->function_spaces().size(); }

  /// @brief Extract common mesh for the form.
  /// @return The mesh.
  std::shared_ptr<const dolfinx::mesh::Mesh<geometry_type>> mesh() const
  {
    return _form->mesh();
  }

  /// @brief Function spaces for all arguments.
  /// @return Function spaces.
  const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<geometry_type>>>&
  function_spaces() const
  {
    return _form->function_spaces();
  }

  /// @brief Get the kernel function for integral `i` on given domain
  /// type.
  /// @param[in] type Integral type.
  /// @param[in] i Domain identifier (index).
  /// @return Function to call for `tabulate_tensor_runtime`.
  std::function<void(scalar_type*, const scalar_type*, const scalar_type*, const geometry_type*,
                     const int*, const uint8_t*,
                     const int*, const geometry_type*, const geometry_type*,
                     const scalar_type*, const size_t*)>
  kernel(dolfinx::fem::IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::ranges::lower_bound(integrals, i, std::less<>{},
                                       [](const auto& a) { return a.id; });
    if (it != integrals.end() and it->id == i)
      return it->kernel;
    else
      throw std::runtime_error("No kernel for requested domain index.");
  }

  std::set<dolfinx::fem::IntegralType> integral_types() const
  {
    std::set<dolfinx::fem::IntegralType> set;
    for (std::size_t i = 0; i < _integrals.size(); ++i)
    {
      if (!_integrals[i].empty())
        set.insert(static_cast<dolfinx::fem::IntegralType>(i));
    }

    return set;
  }

  int num_integrals(dolfinx::fem::IntegralType type) const
  {
    return _integrals[static_cast<std::size_t>(type)].size();
  }

  std::vector<int> active_coeffs(dolfinx::fem::IntegralType type, std::size_t i) const
  {
    assert(i < _integrals[static_cast<std::size_t>(type)].size());
    return _integrals[static_cast<std::size_t>(type)][i].coeffs;
  }

  std::vector<int> integral_ids(dolfinx::fem::IntegralType type) const
  {
    std::vector<int> ids;
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    std::ranges::transform(integrals, std::back_inserter(ids),
                           [](auto& integral) { return integral.id; });
    return ids;
  }

  std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules(dolfinx::fem::IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::ranges::lower_bound(integrals, i, std::less<>{},
                                       [](const auto& a) { return a.id; });
    if (it != integrals.end() and it->id == i)
      return it->quadrature_rules;
    else
      throw std::runtime_error("No mesh entities for requested domain index.");
  }

  std::vector<std::pair<std::shared_ptr<basix::FiniteElement<T>>, std::int32_t>> elements(dolfinx::fem::IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::ranges::lower_bound(integrals, i, std::less<>{},
                                       [](const auto& a) { return a.id; });
    if (it != integrals.end() and it->id == i)
      return it->elements;
    else
      throw std::runtime_error("No mesh entities for requested domain index.");
  }

  std::shared_ptr<const dolfinx::fem::Form<scalar_type, geometry_type>> _form;

  std::array<std::vector<runtime_integral_data<scalar_type, geometry_type>>, 4>
      _integrals;

};

template <std::floating_point T>
void find_element_leaves(std::shared_ptr<const dolfinx::fem::FiniteElement<T>> element,
                          std::vector<std::shared_ptr<const dolfinx::fem::FiniteElement<T>>>& leaves)
{
    if (!element) return;

    //std::cout << "N sub elements=" << element->num_sub_elements() << std::endl;

    // If the node has no children, it's a leaf
    if (element->num_sub_elements()== 0) {
        leaves.push_back(element);
        //std::cout << "Add element:" << leaves.size() << std::endl;
    }

    // Recursively check all children of the current node
    for (auto child : element->sub_elements()) {
        find_element_leaves(child, leaves);
    }
}

  template <std::floating_point T>
  void map_hash_element(const std::vector<std::shared_ptr<basix::FiniteElement<T>>>& elements,
                      std::unordered_map<std::size_t,std::shared_ptr<basix::FiniteElement<T>>>& hash_el)
  {
    for(auto& basix_element : elements)
    {
      auto hash = basix_element->hash();

      if(hash_el.find(hash) != hash_el.end())
      {

      }
      else{
        hash_el.insert({hash,basix_element}); 
      }
    }
  }

  // Compile list of all basix element in flattened components that are used in form.
  // These can come from function spaces, the mesh or coefficients
  template <dolfinx::scalar T,
            std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void form_elements(std::shared_ptr<const dolfinx::fem::Form<T,U>> form,
    std::vector<std::shared_ptr<basix::FiniteElement<U>>>& elements)
  {
    //Start with finding elements in functionspaces
    const std::vector<std::shared_ptr<const dolfinx::fem::FunctionSpace<U>>>&
    function_spaces = form->function_spaces();

    int num_spaces = function_spaces.size();

    std::vector<std::shared_ptr<const dolfinx::fem::FiniteElement<U>>> elements_function_spaces;

    for(std::size_t i=0;i<num_spaces;i++)
    {
      auto element = function_spaces[i]->element();
      assert(element);

      //extract subspaces
      if(element->is_mixed())
      {
        std::cout << "Warning implementation still to be tested for mixed element" << std::endl;
      }

      find_element_leaves<U>(element,elements_function_spaces);
    }

    //get basix element for all components of found dolfinx elements
    for(auto& element : elements_function_spaces)
    {
      auto basix_element = std::make_shared<basix::FiniteElement<U>>(element->basix_element());
      assert(basix_element); // should all exist after flattening
      elements.push_back(basix_element);
    }

    //std::cout << "Number of component elements identified from function spaces=" << elements.size() << std::endl;

    //Now Mesh elements
    auto mesh = form->mesh();
    const dolfinx::fem::CoordinateElement<U>& coord_el = mesh->geometry().cmap();

    auto basix_element = std::make_shared<basix::FiniteElement<U>>(basix::create_element<T>(basix::element::family::P,
                                    dolfinx::mesh::cell_type_to_basix_type(coord_el.cell_shape()),
                                    coord_el.degree(), coord_el.variant(),
                                    basix::element::dpc_variant::unset, false));

    elements.push_back(basix_element);

    //std::cout << "Number of component elements identified from mesh=" << elements.size() << std::endl;

    //Now find elements in coefficients
    auto coefficients = form->coefficients();
    std::vector<std::shared_ptr<const dolfinx::fem::FiniteElement<U>>> elements_coefficients;
    int element_coeffs_cnt = 0;

    for(auto& coefficient: coefficients)
    {
      auto element = coefficient->function_space()->element();
      assert(element);

      //extract subspaces
      if(element->is_mixed())
      {
        std::cout << "Warning implementation still to be tested for mixed element" << std::endl;
      }

      find_element_leaves<U>(element,elements_coefficients);
    }

    //get basix element for all components of found dolfinx elements
    for(auto& element : elements_coefficients)
    {
      auto basix_element = std::make_shared<basix::FiniteElement<U>>(element->basix_element());
      assert(basix_element); // should all exist after flattening
      elements.push_back(basix_element);
    }

    //std::cout << "Number of component elements identified from coefficients=" << elements_coefficients.size() << std::endl;
  }


// obtain basix elements that are used in integral and put them in the correct order
// these elements are then used during assembly to tabulate basis function values at 
// arbitrary quadrature points at runtime
// find elements corresponding to the right finite element hashes
// note these are potentially components of elements
// elements can come from functionspaces, from meshes or coefficients
/// @todo: see if this cannot be recovered during code generation process
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void pack_elements(ufcx_integral* integral,
                   std::unordered_map<std::size_t,std::shared_ptr<basix::FiniteElement<T>>>& hash_el,
                   std::vector<std::pair<std::shared_ptr<basix::FiniteElement<U>>, std::int32_t>>& basix_elements)
{
  int num_elements = integral->num_fe;

  if(num_elements==0)
    throw std::runtime_error("Number of elements in generated code is zero");

  std::vector<std::uint64_t> element_hashes(num_elements);
  std::vector<int> element_deriv_order(num_elements);
  basix_elements.resize(num_elements);

  for(std::size_t i=0;i<num_elements;i++)
  {
    element_hashes[i] = integral->finite_element_hashes[i];
    element_deriv_order[i] = integral->finite_element_deriv_order[i];
  }

  //next identify element associated with mesh
  for(std::size_t i=0;i<num_elements;i++)
  {
    if (hash_el.find(element_hashes[i]) != hash_el.end())
    {
      basix_elements[i] = std::make_pair(hash_el[element_hashes[i]],element_deriv_order[i]);
    }
    else{
       throw std::runtime_error("Element hash not found in form elements");
    }
  }
}

/// @brief Create a Form from UFCx input with coefficients and constants
/// passed in the required order.
///
/// Use fem::create_form to create a fem::Form with coefficients and
/// constants associated with the name/string.
///
/// @param[in] ufcx_form The UFCx form.
/// @param[in] spaces Vector of function spaces. The number of spaces is
/// equal to the rank of the form.
/// @param[in] coefficients Coefficient fields in the form.
/// @param[in] constants Spatial constants in the form.
/// @param[in] subdomains Subdomain markers.
/// @param[in] entity_maps The entity maps for the form. Empty for
/// single domain problems.
/// @param[in] mesh The mesh of the domain.
///
/// @pre Each value in `subdomains` must be sorted by domain id.
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
CutForm<T, U> create_cut_form_factory(
    const ufcx_form& ufcx_form,
    std::shared_ptr<const dolfinx::fem::Form<T,U>> form,
    const std::map<
        dolfinx::fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>>>&
        subdomains)
{
  auto spaces = form->function_spaces();
  auto mesh = form->mesh();

  auto topology = mesh->topology();
  assert(topology);
  const int tdim = topology->dim();

  const int* integral_offsets = ufcx_form.form_integral_offsets;
  std::vector<int> num_integrals_type(3);
  for (int i = 0; i < 3; ++i)
    num_integrals_type[i] = integral_offsets[i + 1] - integral_offsets[i];

  // Create facets, if required
  if (num_integrals_type[exterior_facet] > 0
      or num_integrals_type[interior_facet] > 0)
  {
    mesh->topology_mutable()->create_entities(tdim - 1);
    mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable()->create_connectivity(tdim, tdim - 1);
  }

  // Get list of integral IDs, and load tabulate tensor into memory for
  // each
  using kern_t = std::function<void(T*, const T*, const T*, const U*,
                                    const int*, const uint8_t*,
                                    const int*, const U*, const U*,
                                    const T*, const size_t*)>;
  std::map<dolfinx::fem::IntegralType, std::vector<runtime_integral_data<T, U>>> integrals;


  //Get all elements in form
  std::vector<std::shared_ptr<basix::FiniteElement<U>>> all_elements;
  form_elements(form,all_elements);

  std::unordered_map<std::size_t,std::shared_ptr<basix::FiniteElement<T>>> hash_el;
  map_hash_element(all_elements, hash_el);

  // Attach cell kernels
  {
    std::span<const int> ids(ufcx_form.form_integral_ids
                                 + integral_offsets[cell],
                             num_integrals_type[cell]);
    auto itg = integrals.insert({dolfinx::fem::IntegralType::cell, {}});
    auto sd = subdomains.find(dolfinx::fem::IntegralType::cell);
    for (int i = 0; i < num_integrals_type[cell]; ++i)
    {
      const int id = ids[i];
      //find if id exists
      for(auto it = sd->second.begin(); it!=sd->second.end(); it++ )
      {
        if(it->first == id)
        {
          ufcx_integral* integral
              = ufcx_form.form_integrals[integral_offsets[cell] + i];
          assert(integral);

          // Build list of active coefficients
          std::vector<int> active_coeffs;
          for (int j = 0; j < ufcx_form.num_coefficients; ++j)
          {
            if (integral->enabled_coefficients[j])
              active_coeffs.push_back(j);
          }

          //add vector of elements used in integral
          std::vector<std::pair<std::shared_ptr<basix::FiniteElement<U>>, std::int32_t>> basix_elements;
          pack_elements(integral, hash_el, basix_elements);

          kern_t k = nullptr;
          if constexpr (std::is_same_v<T, float>)
            k = integral->tabulate_tensor_runtime_float32;
          else if constexpr (std::is_same_v<T, double>)
            k = integral->tabulate_tensor_runtime_float64;

          itg.first->second.emplace_back(id, k, it->second, basix_elements, active_coeffs);
        }
      }
    }
  }

  // Attach exterior facet kernels
  // {
  //   std::span<const int> ids(ufcx_form.form_integral_ids
  //                                + integral_offsets[exterior_facet],
  //                            num_integrals_type[exterior_facet]);
  //   auto itg = integrals.insert({dolfinx::fem::IntegralType::exterior_facet, {}});
  //   auto sd = subdomains.find(dolfinx::fem::IntegralType::exterior_facet);
  //   for (int i = 0; i < num_integrals_type[exterior_facet]; ++i)
  //   {
  //     const int id = ids[i];
  //     ufcx_integral* integral
  //         = ufcx_form.form_integrals[integral_offsets[exterior_facet] + i];
  //     assert(integral);
  //     std::vector<int> active_coeffs;
  //     for (int j = 0; j < ufcx_form.num_coefficients; ++j)
  //     {
  //       if (integral->enabled_coefficients[j])
  //         active_coeffs.push_back(j);
  //     }

  //     kern_t k = nullptr;
  //     if constexpr (std::is_same_v<T, float>)
  //       k = integral->tabulate_tensor_runtime_float32;
  //     else if constexpr (std::is_same_v<T, double>)
  //       k = integral->tabulate_tensor_runtime_float64;

  //     // Build list of entities to assembler over
  //     const std::vector bfacets = mesh::exterior_facet_indices(*topology);
  //     auto f_to_c = topology->connectivity(tdim - 1, tdim);
  //     assert(f_to_c);
  //     auto c_to_f = topology->connectivity(tdim, tdim - 1);
  //     assert(c_to_f);

  //     if (sd != subdomains.end())
  //     {
  //       // NOTE: This requires that pairs are sorted
  //       auto it
  //           = std::ranges::lower_bound(sd->second, id, std::less<>{},
  //                                      [](const auto& a) { return a.first; });
  //       if (it != sd->second.end() and it->first == id)
  //         itg.first->second.emplace_back(id, k, it->second, active_coeffs);
  //     }
  //   }
  // }

  // // Attach interior facet kernels
  // {
  //   std::span<const int> ids(ufcx_form.form_integral_ids
  //                                + integral_offsets[interior_facet],
  //                            num_integrals_type[interior_facet]);
  //   auto itg = integrals.insert({dolfinx::fem::IntegralType::interior_facet, {}});
  //   auto sd = subdomains.find(dolfinx::fem::IntegralType::interior_facet);

  //   // Create indicator for interprocess facets
  //   std::vector<std::int8_t> interprocess_marker;
  //   if (num_integrals_type[interior_facet] > 0)
  //   {
  //     assert(topology->index_map(tdim - 1));
  //     const std::vector<std::int32_t>& interprocess_facets
  //         = topology->interprocess_facets();
  //     std::int32_t num_facets = topology->index_map(tdim - 1)->size_local()
  //                               + topology->index_map(tdim - 1)->num_ghosts();
  //     interprocess_marker.resize(num_facets, 0);
  //     std::ranges::for_each(interprocess_facets, [&interprocess_marker](auto f)
  //                           { interprocess_marker[f] = 1; });
  //   }

  //   for (int i = 0; i < num_integrals_type[interior_facet]; ++i)
  //   {
  //     const int id = ids[i];
  //     ufcx_integral* integral
  //         = ufcx_form.form_integrals[integral_offsets[interior_facet] + i];
  //     assert(integral);
  //     std::vector<int> active_coeffs;
  //     for (int j = 0; j < ufcx_form.num_coefficients; ++j)
  //     {
  //       if (integral->enabled_coefficients[j])
  //         active_coeffs.push_back(j);
  //     }

  //     kern_t k = nullptr;
  //     if constexpr (std::is_same_v<T, float>)
  //       k = integral->tabulate_tensor_runtime_float32;
  //     else if constexpr (std::is_same_v<T, double>)
  //       k = integral->tabulate_tensor_runtime_float64;

  //     // Build list of entities to assembler over
  //     auto f_to_c = topology->connectivity(tdim - 1, tdim);
  //     assert(f_to_c);
  //     auto c_to_f = topology->connectivity(tdim, tdim - 1);
  //     assert(c_to_f);

  //     if (sd != subdomains.end())
  //     {
  //       auto it
  //           = std::ranges::lower_bound(sd->second, id, std::less<>{},
  //                                      [](const auto& a) { return a.first; });
  //       if (it != sd->second.end() and it->first == id)
  //         itg.first->second.emplace_back(id, k, it->second, active_coeffs);
  //     }
  //   }
  // }

  return CutForm<T, U>(form, integrals);
}


/// @brief Create a Form using a factory function that returns a pointer
/// to a `ufcx_form`.
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
CutForm<T, U> create_form(
    ufcx_form* (*fptr)(),
    std::shared_ptr<const dolfinx::fem::Form<T,U>> form,
    const std::map<
        dolfinx::fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>>>&
        subdomains
)
{
  ufcx_form* ufcform = fptr();
  CutForm<T, U> L = create_cut_form_factory<T, U>(*ufcform, form, subdomains);
  std::free(form);
  return L;
}

} // namespace cutfemx::fem