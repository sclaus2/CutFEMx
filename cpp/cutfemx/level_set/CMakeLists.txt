set(HEADERS_level_set
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_entities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/locate_entities.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cut_entities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/locate_entities.cpp
)
