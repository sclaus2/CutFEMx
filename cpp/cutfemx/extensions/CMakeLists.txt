# Extensions module for CutFEMx
# This module contains routines for extending solutions from interior nodes to exterior nodes

set(HEADERS_extensions
  ${CMAKE_CURRENT_SOURCE_DIR}/cell_root_mapping.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dof_root_mapping.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dof_basis_evaluation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix_operations.h
  PARENT_SCOPE)

# Source files will be added here as we create them
# target_sources(cutfemx PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/extension_routines.cpp
# )
