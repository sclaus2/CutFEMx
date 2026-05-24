set(HEADERS_level_set
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_entities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/locate_entities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ghost_penalty_facets.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_normal.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_entities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/locate_entities.cpp
)
