set(HEADERS_level_set
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_entities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/locate_entities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ghost_penalty_facets.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_normal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/pt_tri_distance.h
  ${CMAKE_CURRENT_SOURCE_DIR}/band_selection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parallel_min_exchange.h
  ${CMAKE_CURRENT_SOURCE_DIR}/eikonal_update.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fmm.h
  ${CMAKE_CURRENT_SOURCE_DIR}/vertex_graph.h
  ${CMAKE_CURRENT_SOURCE_DIR}/sign.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_entities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/locate_entities.cpp
)
