# Extensions module for CutFEMx
# This module contains routines for extending solutions from interior nodes to exterior nodes

set(HEADERS_extensions
  ${CMAKE_CURRENT_SOURCE_DIR}/cell_aggregation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/extension_penalty.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cell_aggregation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/extension_penalty.cpp
)
