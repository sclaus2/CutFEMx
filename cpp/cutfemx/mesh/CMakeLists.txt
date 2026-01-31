set(HEADERS_mesh
  ${CMAKE_CURRENT_SOURCE_DIR}/convert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl_surface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl_reader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/distribute_stl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/orient_surface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mp_predicates.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tri_intersection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cell_triangle_map.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.cpp
)
