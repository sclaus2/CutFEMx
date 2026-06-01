set(HEADERS_fem
  ${CMAKE_CURRENT_SOURCE_DIR}/entity_dofmap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/deactivate.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/entity_dofmap.cpp
)
