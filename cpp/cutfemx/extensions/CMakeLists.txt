# Extensions module for CutFEMx
# This module contains routines for extending solutions from interior nodes to exterior nodes

set(HEADERS_extensions
  ${CMAKE_CURRENT_SOURCE_DIR}/discrete_extension.h
  ${CMAKE_CURRENT_SOURCE_DIR}/discrete_extension_impl.h
  PARENT_SCOPE)

# Source files will be added here as we create them
# target_sources(cutfemx PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}/extension_routines.cpp
# )
