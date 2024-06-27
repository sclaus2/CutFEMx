set(HEADERS_fem
  ${CMAKE_CURRENT_SOURCE_DIR}/entity_dofmap.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/entity_dofmap.cpp
)
