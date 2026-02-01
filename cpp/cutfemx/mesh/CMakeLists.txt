set(HEADERS_mesh
  ${CMAKE_CURRENT_SOURCE_DIR}/convert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.h
  PARENT_SCOPE)

set(HEADERS_mesh_stl
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/stl_surface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/stl_reader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/distribute_stl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/orient_surface.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/mp_predicates.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/tri_intersection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/cell_triangle_map.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stl/mesh_adapt_stl.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.cpp
)
