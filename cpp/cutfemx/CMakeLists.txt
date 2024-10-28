# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Declare the library (target)
add_library(cutfemx)

# ------------------------------------------------------------------------------
# Add source files to the target
set(CUTFEMX_DIRS level_set fem mesh)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/version.h.in version.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add source to dolfinx target, and get sets of header files
foreach(DIR ${CUTFEMX_DIRS})
  add_subdirectory(${DIR})
endforeach()

foreach(DIR ${CUTFEMX_DIRS})
  set_target_properties(cutfemx PROPERTIES PRIVATE_HEADER "${HEADERS_${DIR}}")
  install(FILES ${HEADERS_${DIR}} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cutfemx/${DIR}
    COMPONENT Development)
endforeach()

target_include_directories(cutfemx PUBLIC
  $<INSTALL_INTERFACE:include>
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}>")

# dolfinx
target_link_libraries(cutfemx PUBLIC CUTCELLS::cutcells)
target_link_libraries(cutfemx PUBLIC dolfinx)
