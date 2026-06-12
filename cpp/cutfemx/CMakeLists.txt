include(GNUInstallDirs)

add_library(cutfemx)

set(CUTFEMX_DIRS
  cut
  distance
  extensions
  fem
  geometry
  level_set
  mesh)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.h.in version.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

foreach(DIR ${CUTFEMX_DIRS})
  add_subdirectory(${DIR})
endforeach()

foreach(DIR ${CUTFEMX_DIRS})
  install(FILES ${HEADERS_${DIR}} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cutfemx/${DIR}
    COMPONENT Development)
endforeach()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/distance/stl
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cutfemx/distance
  COMPONENT Development
  FILES_MATCHING PATTERN "*.h")

target_include_directories(cutfemx PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..;${CMAKE_CURRENT_SOURCE_DIR}>")

target_link_libraries(cutfemx PUBLIC CUTCELLS::cutcells)
target_link_libraries(cutfemx PUBLIC dolfinx)
target_link_libraries(cutfemx PUBLIC spdlog::spdlog)

target_compile_features(cutfemx PUBLIC cxx_std_20)

if(DEFINED ENV{CONDA_PREFIX})
  target_include_directories(cutfemx BEFORE PUBLIC "$ENV{CONDA_PREFIX}/include")
endif()

if(CUTFEMX_USE_MULTIPRECISION_PSM)
  target_sources(cutfemx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/geogram.psm.MultiPrecision/MultiPrecision_psm.cpp)
  target_include_directories(cutfemx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/geogram.psm.MultiPrecision>)
  target_compile_definitions(cutfemx PUBLIC CUTFEMX_USE_MULTIPRECISION_PSM)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/geogram.psm.MultiPrecision/MultiPrecision_psm.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Development)
endif()
