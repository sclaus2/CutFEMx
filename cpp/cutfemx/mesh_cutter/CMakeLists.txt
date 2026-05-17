set(HEADERS_mesh_cutter
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshCutter.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshCutter.cpp
)
