set(HEADERS_basix_wrapper
  ${CMAKE_CURRENT_SOURCE_DIR}/cwrapper.h
  PARENT_SCOPE)

target_sources(cutfemx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cwrapper.cpp
)
