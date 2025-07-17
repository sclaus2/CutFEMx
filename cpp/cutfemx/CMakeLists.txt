# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Declare the library (target)
add_library(cutfemx)

# ------------------------------------------------------------------------------
# Add source files to the target
set(CUTFEMX_DIRS level_set fem mesh quadrature extensions)

# Add source to cutfemx target, and get sets of header files
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

# Link with required libraries
target_link_libraries(cutfemx PUBLIC dolfinx)

# Required CUTCELLS library
if(TARGET CUTCELLS::cutcells)
    target_link_libraries(cutfemx PUBLIC CUTCELLS::cutcells)
    message(STATUS "Linking with CUTCELLS::cutcells")
elseif(TARGET cutcells)
    target_link_libraries(cutfemx PUBLIC cutcells)
    message(STATUS "Linking with cutcells")
else()
    message(FATAL_ERROR "CUTCELLS is required but not found. Please install CUTCELLS.")
endif()

# Required spdlog library
if(TARGET spdlog::spdlog)
    target_link_libraries(cutfemx PUBLIC spdlog::spdlog)
    message(STATUS "Linking with spdlog::spdlog")
else()
    message(FATAL_ERROR "spdlog is required but not found. Please install spdlog.")
endif()
