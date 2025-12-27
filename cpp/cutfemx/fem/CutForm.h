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
#include <cstdio>
#include <ufcx.h>
#include <spdlog/spdlog.h>

#include <basix/finite-element.h>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/utils.h>
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

/// @brief Type of integral
enum class IntegralType : std::int8_t
{
  cutcell = 0,           ///< run time integration one parent cell -> dC
  interface = 1,         ///< run time integration coupling two parent cells -> dI
};

/// @brief Source type for element metadata (passed from Python)
enum class ElementSourceType : std::int8_t
{
  Argument = 0,     ///< Element from form argument (trial/test function space)
  Coefficient = 1,  ///< Element from coefficient function space
  Geometry = 2,     ///< Coordinate element from mesh geometry
};

/// @brief Describes where to find an element for tabulation
/// Passed from Python based on FFCx IR analysis
struct ElementSource
{
  ElementSourceType type;
  int index;       ///< Argument or coefficient index
  int component;   ///< Sub-element component (-1 = whole element)
  int deriv_order; ///< Maximum derivative order needed
};


static std::string integralTypeToString(dolfinx::fem::IntegralType type)
{
  switch (type)
  {
  case dolfinx::fem::IntegralType::cell:
    return "cell";
  case dolfinx::fem::IntegralType::exterior_facet:
    return "exterior_facet";
  case dolfinx::fem::IntegralType::interior_facet:
    return "interior_facet";
  default:
    return "unknown";
  }
}

static std::string rtIntegralTypeToString(cutfemx::fem::IntegralType type)
{
  switch (type)
  {
  case cutfemx::fem::IntegralType::cutcell:
    return "cutcell";
  case cutfemx::fem::IntegralType::interface:
    return "interface";
  default:
    return "unknown";
  }
}


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
  runtime_integral_data(int id, K&& kernel, 
                std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules,
                std::vector<ElementSource> element_sources,
                W&& coeffs)
      : id(id), kernel(std::forward<K>(kernel)),
        quadrature_rules(quadrature_rules),
        element_sources(std::move(element_sources)),
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

  /// @brief Element sources describing where to get elements for tabulation
  /// (from Python based on FFCx IR analysis)
  std::vector<ElementSource> element_sources;

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
                 std::map<cutfemx::fem::IntegralType, std::vector<runtime_integral_data<
                                            scalar_type, geometry_type>>>>
  CutForm(std::shared_ptr<const dolfinx::fem::Form<scalar_type, geometry_type>> form,
      X&& integrals)
  {
    spdlog::info("Creating CutForm with {} integrals", integrals.size());
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

  std::shared_ptr<const dolfinx::fem::Form<scalar_type, geometry_type>> form() const
  {
    return _form;
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
  kernel(cutfemx::fem::IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::ranges::lower_bound(integrals, i, std::less<>{},
                                       [](const auto& a) { return a.id; });
    if (it != integrals.end() and it->id == i)
      return it->kernel;
    else
      throw std::runtime_error("No kernel for requested domain index.");
  }

  std::set<cutfemx::fem::IntegralType> integral_types() const
  {
    std::set<cutfemx::fem::IntegralType> set;
    for (std::size_t i = 0; i < _integrals.size(); ++i)
    {
      if (!_integrals[i].empty())
        set.insert(static_cast<cutfemx::fem::IntegralType>(i));
    }

    return set;
  }

  int num_integrals(cutfemx::fem::IntegralType type) const
  {
    return _integrals[static_cast<std::size_t>(type)].size();
  }

  std::vector<int> active_coeffs(cutfemx::fem::IntegralType type, std::size_t i) const
  {
    assert(i < _integrals[static_cast<std::size_t>(type)].size());
    return _integrals[static_cast<std::size_t>(type)][i].coeffs;
  }

  std::vector<int> integral_ids(cutfemx::fem::IntegralType type) const
  {
    std::vector<int> ids;
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    std::ranges::transform(integrals, std::back_inserter(ids),
                           [](auto& integral) { return integral.id; });
    return ids;
  }

  std::shared_ptr<const cutfemx::quadrature::QuadratureRules<U>> quadrature_rules(cutfemx::fem::IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::ranges::lower_bound(integrals, i, std::less<>{},
                                       [](const auto& a) { return a.id; });
    if (it != integrals.end() and it->id == i)
      return it->quadrature_rules;
    else
      throw std::runtime_error("No mesh entities for requested domain index.");
  }

  std::vector<ElementSource> element_sources(cutfemx::fem::IntegralType type, int i) const
  {
    const auto& integrals = _integrals[static_cast<std::size_t>(type)];
    auto it = std::ranges::lower_bound(integrals, i, std::less<>{},
                                       [](const auto& a) { return a.id; });
    if (it != integrals.end() and it->id == i)
      return it->element_sources;
    else
      throw std::runtime_error("No mesh entities for requested domain index.");
  }

  

  void update_integration_domains(const std::map<
        dolfinx::fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>&
        subdomains,
        const std::map<std::shared_ptr<const dolfinx::mesh::Mesh<geometry_type>>,
                     std::span<const std::int32_t>>& entity_maps = {})
  {
    spdlog::info("Updating standard integration domains");
    std::map<dolfinx::fem::IntegralType, std::vector<dolfinx::fem::integral_data<T, U>>> integrals;

    std::vector<dolfinx::fem::IntegralType> itg_types = {dolfinx::fem::IntegralType::cell,
                                                         dolfinx::fem::IntegralType::exterior_facet,
                                                         dolfinx::fem::IntegralType::interior_facet};

    for(auto& itg_type : itg_types)
    {
      auto ids = _form->integral_ids(itg_type);
      auto itg = integrals.insert({itg_type, {}});
      auto sd = subdomains.find(itg_type);

      if(sd==subdomains.end())
      {
        spdlog::info("No subdomains to update for integral type: " + integralTypeToString(itg_type));
        continue;
      }

      //Check if ids given in update exit in integral_ids
      for(auto& [id, subdomain] : sd->second)
      {
        auto it = std::ranges::find(ids, id);
        if(it==ids.end())
        {
          throw std::runtime_error("Integral " + integralTypeToString(itg_type) + " with id " + std::to_string(id) + " to update not found in form with rank: " + std::to_string(_form->rank()) + ". Please recompile form.");
        }
      }

      for(auto& id : ids)
      {
          // see if id is in subdomains to update
          auto it = std::ranges::find(sd->second, id,
                            &std::pair<std::int32_t, std::span<const std::int32_t>>::first);

          if(it!=sd->second.end())
          {
            spdlog::info("Updating standard integral " + integralTypeToString(itg_type) + ": " + std::to_string(id) );
            itg.first->second.emplace_back(id, _form->kernel(itg_type, id), it->second, _form->active_coeffs(itg_type, id));
          }
          else //do not update and leave as is
          {
            spdlog::info("Keeping standard integral " + integralTypeToString(itg_type) + ": " + std::to_string(id) );
            itg.first->second.emplace_back(id, _form->kernel(itg_type, id), _form->domain(itg_type, id), _form->active_coeffs(itg_type, id));
          }
      }
    }

    spdlog::info("Creating new form with updated integration domains");
    std::shared_ptr<dolfinx::fem::Form<T,U>> new_form = std::make_shared<dolfinx::fem::Form<T, U>>(_form->function_spaces(), integrals, _form->coefficients(), _form->constants(),
                    _form->needs_facet_permutations(), entity_maps, _form->mesh());

    ///@todo: make sure previous form is destroyed
    _form = new_form;
  }

  void update_runtime_domains(const std::map<
        cutfemx::fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>>>&
        quaddomains)
  {
    spdlog::info("Updating runtime domains");
    std::vector<cutfemx::fem::IntegralType> itg_types = {cutfemx::fem::IntegralType::cutcell,
                                                      cutfemx::fem::IntegralType::interface};

    for(auto& itg_type : itg_types)
    {
      auto sd = quaddomains.find(itg_type);


      if(sd==quaddomains.end())
      {
        spdlog::info("No quadrature rule to update for integral type: " + rtIntegralTypeToString(itg_type));
        continue;
      }

      auto& integrals = _integrals[static_cast<std::size_t>(itg_type)];
      auto ids = this->integral_ids(itg_type);

      //Check if ids given in update exit in integral_ids
      for(auto& [id, subdomain] : sd->second)
      {
        auto it = std::ranges::find(ids, id);
        if(it==ids.end())
        {
          throw std::runtime_error("Runtime integral " + rtIntegralTypeToString(itg_type) + " with id " + std::to_string(id) + " to update not found in form with rank: " + std::to_string(_form->rank()) + ". Please recompile form.");
        }
      }

      for(auto& [id, quad_rule] : sd->second)
      {
        for(auto& integral: integrals)
        {
          //if id is same as input replace quadrature rule in integral
          if(integral.id == id)
          {
            spdlog::info("Updating runtime integral " + rtIntegralTypeToString(itg_type) + ": " + std::to_string(id) );
            //std::cout << "update rule: " << id << std::endl;
            integral.quadrature_rules = quad_rule;
            spdlog::info("Updated " + rtIntegralTypeToString(itg_type) + ": " + std::to_string(id) );
          }
        }
      }
    }
    spdlog::info("Runtime Integrals have been updated.");
  }

  std::shared_ptr<const dolfinx::fem::Form<scalar_type, geometry_type>> _form;

  std::array<std::vector<runtime_integral_data<scalar_type, geometry_type>>, 2>
      _integrals;

};

template <std::floating_point T>
void collect_all_elements(std::shared_ptr<const dolfinx::fem::FiniteElement<T>> element,
                          std::vector<std::shared_ptr<const dolfinx::fem::FiniteElement<T>>>& elements)
{
    if (!element) return;

    // Add current element (whether mixed or leaf)
    elements.push_back(element);

    // Recursively check all children of the current node
    for (auto child : element->sub_elements()) {
        collect_all_elements(child, elements);
    }
}

  template <std::floating_point T>
  void map_hash_element(const std::vector<std::pair<std::size_t, std::shared_ptr<const dolfinx::fem::FiniteElement<T>>>>& elements,
                      std::unordered_map<std::size_t,std::shared_ptr<const dolfinx::fem::FiniteElement<T>>>& hash_el)
  {
    for(auto& [hash, element] : elements)
    {
      hash_el[hash] = element;
    }
  }

  // Compile list of all basix element in flattened components that are used in form.
  // These can come from function spaces, the mesh or coefficients
  template <dolfinx::scalar T,
            std::floating_point U = dolfinx::scalar_value_type_t<T>>
  void form_elements(std::shared_ptr<const dolfinx::fem::Form<T,U>> form,
                     std::vector<std::pair<std::size_t, std::shared_ptr<const dolfinx::fem::FiniteElement<U>>>>& elements)
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
      collect_all_elements<U>(element,elements_function_spaces);
    }

    //get element for all components of found dolfinx elements
    for(auto& element : elements_function_spaces)
    {
       if (element->num_sub_elements() == 0)
       {
          elements.push_back({element->basix_element().hash(), element});
          // Leaf Element: Map 0 to it as well (FFCx runtime fix for mixed elements)
          elements.push_back({0, element});
       }
       else
       {
          // Mixed Element: Just map 0
          elements.push_back({0, element});
       }
    }

    //std::cout << "Number of component elements identified from function spaces=" << elements.size() << std::endl;

    //Now Mesh elements
    auto mesh = form->mesh();
    const dolfinx::fem::CoordinateElement<U>& coord_el = mesh->geometry().cmap();

    //TODO: This creates a new element just for hash/storage, ideally we should use the one from mesh if possible or just store it.
    // However, coordinate element doesn't inherit from FiniteElement in same way.
    // For now, let's keep the basix creation for coordinate element as it's usually standard.
    // WAIT: We need Dolfinx FiniteElement!
    // Since CoordinateElement is distinct, we might need a wrapper or just rely on Basix for this one specific case?
    // BUT our vector expects Dolfinx FiniteElement.
    // We can construct a dolfinx::fem::FiniteElement from the basix element.
    
    auto basix_element = basix::create_element<T>(basix::element::family::P,
                                    dolfinx::mesh::cell_type_to_basix_type(coord_el.cell_shape()),
                                    coord_el.degree(), coord_el.variant(),
                                    basix::element::dpc_variant::unset, false);
    
    // Create dummy function space to generate dolfinx element wrapper
    auto V_dummy = dolfinx::fem::create_functionspace(std::make_shared<dolfinx::mesh::Mesh<U>>(*mesh), basix_element, {1});
    auto dolfinx_coord_element = V_dummy->element();
    
    elements.push_back({basix_element.hash(), dolfinx_coord_element});

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
      collect_all_elements<U>(element,elements_coefficients);
    }

    //get element for all components of found dolfinx elements
    for(auto& element : elements_coefficients)
    {
       if (element->num_sub_elements() == 0)
       {
          elements.push_back({element->basix_element().hash(), element});
          // Leaf Element
          elements.push_back({0, element});
       }
       else
       {
          elements.push_back({0, element});
       }
    }

    //std::cout << "Number of component elements identified from coefficients=" << elements_coefficients.size() << std::endl;
  }

/// @brief Resolve element sources to basix elements for tabulation
/// Uses the form's function spaces and coefficients to look up the actual elements
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
void resolve_element_sources(
    const std::vector<ElementSource>& sources,
    const dolfinx::fem::Form<T, U>& form,
    std::vector<std::pair<std::shared_ptr<basix::FiniteElement<U>>, std::int32_t>>& basix_elements)
{
  basix_elements.resize(sources.size());
  
  const auto& function_spaces = form.function_spaces();
  const auto& coefficients = form.coefficients();
  auto mesh = form.mesh();
  
  for (std::size_t i = 0; i < sources.size(); ++i)
  {
    const auto& src = sources[i];
    std::shared_ptr<const dolfinx::fem::FiniteElement<U>> element;
    
    switch (src.type)
    {
      case ElementSourceType::Argument:
      {
        if (src.index < 0 || static_cast<std::size_t>(src.index) >= function_spaces.size())
          throw std::runtime_error("Invalid argument index in ElementSource");
        element = function_spaces[src.index]->element();
        break;
      }
      case ElementSourceType::Coefficient:
      {
        if (src.index < 0 || static_cast<std::size_t>(src.index) >= coefficients.size())
          throw std::runtime_error("Invalid coefficient index in ElementSource");
        element = coefficients[src.index]->function_space()->element();
        break;
      }
      case ElementSourceType::Geometry:
      {
        // Create basix element for coordinate element
        const auto& coord_el = mesh->geometry().cmap();
        try {
            auto basix_el = std::make_shared<basix::FiniteElement<U>>(
                basix::create_element<U>(basix::element::family::P,
                    dolfinx::mesh::cell_type_to_basix_type(coord_el.cell_shape()),
                    coord_el.degree(), coord_el.variant(),
                    basix::element::dpc_variant::unset, false));
            basix_elements[i] = {basix_el, src.deriv_order};
        } catch (const std::exception& e) {
             std::cout << "Exception in basix::create_element: " << e.what() << std::endl;
             throw;
        }
        continue;
      }
      default:
        throw std::runtime_error("Unknown ElementSourceType");
    }
    
    // Handle sub-elements for mixed elements BEFORE accessing basix_element
    if (src.component >= 0)
    {
      const auto& sub_elements = element->sub_elements();
      if (static_cast<std::size_t>(src.component) >= sub_elements.size())
        throw std::runtime_error("Invalid component index for mixed element");
      element = sub_elements[src.component];
    }
    
    // Now we can safely access basix_element
    
    // Extract basix element for tabulation
    basix_elements[i] = {
        std::make_shared<basix::FiniteElement<U>>(element->basix_element()),
        src.deriv_order
    };
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
/// @param[in] element_sources_map Element sources per integral from Python.
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
        cutfemx::fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>>>&
        subdomains,
    const std::map<
        std::pair<cutfemx::fem::IntegralType, int>,
        std::vector<ElementSource>>&
        element_sources_map = {})
{
  auto spaces = form->function_spaces();
  auto mesh = form->mesh();

  auto topology = mesh->topology();
  assert(topology);
  const int tdim = topology->dim();

  const int* integral_offsets = ufcx_form.form_integral_offsets;
  std::vector<int> num_integrals_type(5);
  for (int i = 0; i < 5; ++i)
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
  std::map<cutfemx::fem::IntegralType, std::vector<runtime_integral_data<T, U>>> integrals;

  // Attach cell kernels
  {
    std::span<const int> ids(ufcx_form.form_integral_ids
                                 + integral_offsets[cutcell],
                             num_integrals_type[cutcell]);
    auto itg = integrals.insert({cutfemx::fem::IntegralType::cutcell, {}});
    auto sd = subdomains.find(cutfemx::fem::IntegralType::cutcell);

    if (sd == subdomains.end()) {
    //integraltype not found
    //std::cout << "no such integral type" << std::endl;
    }
    else
    {
      std::vector<std::pair<std::int32_t, std::shared_ptr<cutfemx::quadrature::QuadratureRules<U>>>> domains = sd->second;
      for (int i = 0; i < num_integrals_type[cutcell]; ++i)
      {
        const int id = ids[i];
        //find if id exists
        for(auto domain: domains)
        {
          if(domain.first == id)
          {
            ufcx_integral* integral
                = ufcx_form.form_integrals[integral_offsets[cutcell] + i];
            assert(integral);

            // Build list of active coefficients
            std::vector<int> active_coeffs;
            for (int j = 0; j < ufcx_form.num_coefficients; ++j)
            {
              if (integral->enabled_coefficients[j])
                active_coeffs.push_back(j);
            }

            // Get element sources from Python-passed map
            std::vector<ElementSource> sources;
            auto key = std::make_pair(cutfemx::fem::IntegralType::cutcell, id);
            auto es_it = element_sources_map.find(key);
            if (es_it != element_sources_map.end())
            {
              sources = es_it->second;
            }
            else
            {
              // Fallback: create default sources for arguments
              for (std::size_t arg_idx = 0; arg_idx < spaces.size(); ++arg_idx)
              {
                sources.push_back({ElementSourceType::Argument, 
                                   static_cast<int>(arg_idx), -1, 1});
              }
            }

            kern_t k = nullptr;
            if constexpr (std::is_same_v<T, float>)
              k = integral->tabulate_tensor_runtime_float32;
            else if constexpr (std::is_same_v<T, double>)
              k = integral->tabulate_tensor_runtime_float64;

            itg.first->second.emplace_back(id, k, domain.second, sources, active_coeffs);
          }
        }
      }
    }

  }

  return CutForm<T, U>(form, integrals);
}


/// @brief Create a Form using a factory function that returns a pointer
/// to a `ufcx_form`.
template <dolfinx::scalar T, std::floating_point U = scalar_value_type_t<T>>
CutForm<T, U> create_form(
    ufcx_form* (*fptr)(),
    std::shared_ptr<const dolfinx::fem::Form<T,U>> form,
    const std::map<
        cutfemx::fem::IntegralType,
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
