// Copyright (c) 2022 Susanne Claus
// FEniCS Project
// SPDX-License-Identifier:    MIT

// Declares a simple C API wrapper around C++.

#ifndef CWRAPPER_H_
#define CWRAPPER_H_

#include <stdbool.h>

#ifdef __cplusplus
// extern C indicates to C++ compiler that the following code
// is meant to be called by a C compiler
// i.e. tells the compiler that C naming conventions etc should be used
extern "C" {
#endif

/// Basix element structure to map c basix element to C++ basix finite_element
typedef struct basix_element basix_element;

/// Create C-struct pointer of basix_element
/// The int values indicate the position of the values in the corresponding
/// enum classes in basix this avoids doubling all the enums in the C-Wrapper
basix_element* basix_element_create(const int basix_family, const int basix_cell_type, const int degree,
                                    const int lagrange_variant, const int dpc_variant, const bool discontinuous);

void basix_element_tabulate_shape(basix_element *element, const unsigned int num_points,
                                  const int nd, int* cshape);

void basix_element_tabulate(basix_element *element, const unsigned int gdim, const double* points,
                            const unsigned int num_points, const int nd,
                            double* table_data, long unsigned int table_data_size);

/// Destroy pointer to C-struct of basix_element
void basix_element_destroy(basix_element *element);
void basix_elements_destroy(basix_element** elements);

#ifdef __cplusplus
}
#endif


#endif /*CWRAPPER_H_*/
