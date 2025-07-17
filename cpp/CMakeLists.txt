cmake_minimum_required(VERSION 3.21)

# Include version information
include(${CMAKE_CURRENT_SOURCE_DIR}/../VERSION.cmake)

# Define project with version
project(CUTFEMX 
    VERSION ${CUTFEMX_VERSION}
    DESCRIPTION "Cut Finite Element Methods for FEniCSx"
    LANGUAGES C CXX)

# Find required dependencies
find_package(PkgConfig REQUIRED)
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(DOLFINX REQUIRED)

# Find CUTCELLS (required dependency)
find_package(CUTCELLS REQUIRED)
if(CUTCELLS_FOUND)
    message(STATUS "CUTCELLS found at: ${CUTCELLS_DIR}")
    if(TARGET CUTCELLS::cutcells)
        get_target_property(CUTCELLS_LOCATION CUTCELLS::cutcells LOCATION)
        message(STATUS "CUTCELLS target location: ${CUTCELLS_LOCATION}")
    endif()
else()
    message(STATUS "CUTCELLS not found - will try to find as cutcells")
    find_package(cutcells REQUIRED)
    if(cutcells_FOUND)
        message(STATUS "cutcells found at: ${cutcells_DIR}")
        if(TARGET cutcells)
            get_target_property(CUTCELLS_LOCATION cutcells LOCATION)
            message(STATUS "cutcells target location: ${CUTCELLS_LOCATION}")
        endif()
    endif()
endif()

# Find spdlog (required dependency)
find_package(spdlog REQUIRED)
if(spdlog_FOUND)
    message(STATUS "spdlog found at: ${spdlog_DIR}")
    if(TARGET spdlog::spdlog)
        get_target_property(SPDLOG_LOCATION spdlog::spdlog LOCATION)
        message(STATUS "spdlog target location: ${SPDLOG_LOCATION}")
    endif()
endif()

# Add the cutfemx subdirectory which will create the library and add sources
add_subdirectory(cutfemx)

# Configure version header AFTER the library is created
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cutfemx/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cutfemx/version.h"
    @ONLY
)

# Add the configured version header to the target's include directories
target_include_directories(cutfemx PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Installation rules for the version header
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cutfemx/version.h"
    DESTINATION include/cutfemx
)




