set(HEADERS_cut
  ${CMAKE_CURRENT_SOURCE_DIR}/cut.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeQuadrature.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cut.cpp
)
