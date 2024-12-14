set(HEADERS_fem
  ${CMAKE_CURRENT_SOURCE_DIR}/entity_dofmap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolate.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CutForm.h
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_tables.h
  ${CMAKE_CURRENT_SOURCE_DIR}/pack_coefficients.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assembler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_scalar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_vector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_matrix.h
  ${CMAKE_CURRENT_SOURCE_DIR}/typedefs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/deactivate.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/entity_dofmap.cpp
)
