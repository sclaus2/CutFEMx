cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------------------------
# Set project name and version number
project(CUTFEMX VERSION "0.1.0" LANGUAGES CXX C)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build cutfemx with shared libraries." ON)
option(CUTFEMX_USE_MULTIPRECISION_PSM "Use MultiPrecision_psm library for high precision arithmetic." ON)
option(CUTFEMX_USE_LOCAL_CUTCELLS "Build the sibling CutCells C++ source tree when available." ON)
option(CUTFEMX_ALLOW_SYSTEM_CUTCELLS "Allow CutFEMx to use a CutCells package from /usr or /usr/local." OFF)
#add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build cutfemx with shared libraries.")

include(FeatureSummary)

set(_cutfemx_conda_prefix "$ENV{CONDA_PREFIX}")

# Find DOLFINx config file
if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

function(_cutfemx_reject_system_cutcells)
  if(CUTFEMX_ALLOW_SYSTEM_CUTCELLS)
    return()
  endif()

  set(_cutfemx_cutcells_paths)
  if(DEFINED CutCells_DIR AND NOT TARGET cutcells)
    list(APPEND _cutfemx_cutcells_paths "${CutCells_DIR}")
  endif()
  if(TARGET CUTCELLS::cutcells)
    get_target_property(_cutfemx_cutcells_includes CUTCELLS::cutcells INTERFACE_INCLUDE_DIRECTORIES)
    if(_cutfemx_cutcells_includes)
      list(APPEND _cutfemx_cutcells_paths ${_cutfemx_cutcells_includes})
    endif()
    foreach(_cutfemx_prop
            IMPORTED_LOCATION
            IMPORTED_LOCATION_NOCONFIG
            IMPORTED_LOCATION_RELEASE
            IMPORTED_LOCATION_DEBUG
            IMPORTED_LOCATION_RELWITHDEBINFO
            IMPORTED_LOCATION_MINSIZEREL)
      get_target_property(_cutfemx_cutcells_location CUTCELLS::cutcells ${_cutfemx_prop})
      if(_cutfemx_cutcells_location)
        list(APPEND _cutfemx_cutcells_paths "${_cutfemx_cutcells_location}")
      endif()
    endforeach()
  endif()

  foreach(_cutfemx_cutcells_path IN LISTS _cutfemx_cutcells_paths)
    if(NOT _cutfemx_cutcells_path
       OR _cutfemx_cutcells_path MATCHES "^\\$<"
       OR _cutfemx_cutcells_path MATCHES "-NOTFOUND$")
      continue()
    endif()
    if(_cutfemx_cutcells_path MATCHES "^/usr/local(/|$)"
       OR _cutfemx_cutcells_path MATCHES "^/usr(/|$)")
      message(FATAL_ERROR
        "Refusing to use CutCells from '${_cutfemx_cutcells_path}'. "
        "Use the sibling CutCells source tree, set CutCells_DIR to the "
        "fenicsx-pr-clean environment, or configure with "
        "-DCUTFEMX_ALLOW_SYSTEM_CUTCELLS=ON if this system install is intentional.")
    endif()
  endforeach()
endfunction()

function(_cutfemx_find_packaged_cutcells)
  if(DEFINED CutCells_DIR AND NOT CUTFEMX_ALLOW_SYSTEM_CUTCELLS)
    if(CutCells_DIR MATCHES "^/usr/local(/|$)"
       OR CutCells_DIR MATCHES "^/usr(/|$)")
      message(FATAL_ERROR
        "Refusing to use CutCells_DIR='${CutCells_DIR}'. "
        "Use the sibling CutCells source tree, set CutCells_DIR to the "
        "fenicsx-pr-clean environment, or configure with "
        "-DCUTFEMX_ALLOW_SYSTEM_CUTCELLS=ON if this system install is intentional.")
    endif()
  endif()

  if(CUTFEMX_ALLOW_SYSTEM_CUTCELLS)
    find_package(CutCells REQUIRED CONFIG)
    return()
  endif()

  set(_cutfemx_cutcells_hints)
  if(DEFINED CutCells_DIR)
    list(APPEND _cutfemx_cutcells_hints "${CutCells_DIR}")
  endif()
  if(_cutfemx_conda_prefix)
    list(APPEND _cutfemx_cutcells_hints
      "${_cutfemx_conda_prefix}"
      "${_cutfemx_conda_prefix}/lib/cmake/cutcells")
  endif()

  find_package(CutCells REQUIRED CONFIG
    PATHS ${_cutfemx_cutcells_hints}
    NO_DEFAULT_PATH)
endfunction()

if(NOT TARGET CUTCELLS::cutcells)
  set(CUTFEMX_LOCAL_CUTCELLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CutCells/cpp")
  if(CUTFEMX_USE_LOCAL_CUTCELLS
     AND EXISTS "${CUTFEMX_LOCAL_CUTCELLS_DIR}/CMakeLists.txt")
    message(STATUS "Using sibling CutCells source: ${CUTFEMX_LOCAL_CUTCELLS_DIR}")
    add_subdirectory("${CUTFEMX_LOCAL_CUTCELLS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/cutcells" EXCLUDE_FROM_ALL)
    if(NOT TARGET CUTCELLS::cutcells)
      add_library(CUTCELLS::cutcells ALIAS cutcells)
    endif()
  else()
    _cutfemx_find_packaged_cutcells()
  endif()
endif()
_cutfemx_reject_system_cutcells()
if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()
if(NOT TARGET spdlog::spdlog)
  find_package(spdlog REQUIRED)
endif()

# Source files
add_subdirectory(cutfemx)

option(CUTFEMX_BUILD_DEMOS "Build demos" OFF)
option(BUILD_TESTING "Build tests" OFF)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CUTFEMX_BUILD_DEMOS)
  add_subdirectory(demo/fmm_distance)
endif()

# Install the cutfemx library
install(TARGETS cutfemx
  EXPORT CUTFEMXTargets
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cutfemx
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development)

# Configure CMake helpers
include(CMakePackageConfigHelpers)
write_basic_package_version_file(CUTFEMXConfigVersion.cmake VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(CUTFEMXConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/CUTFEMXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutfemx)

# Install CMake files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CUTFEMXConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/CUTFEMXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutfemx COMPONENT Development)
install(EXPORT CUTFEMXTargets FILE CUTFEMXTargets.cmake NAMESPACE CUTFEMX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutfemx)
