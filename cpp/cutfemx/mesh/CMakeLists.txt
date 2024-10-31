set(HEADERS_mesh
  ${CMAKE_CURRENT_SOURCE_DIR}/convert.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_mesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/create_mesh.cpp
)
