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
#add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build cutfemx with shared libraries.")

include(FeatureSummary)

# Find DOLFINx config file
if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()
if(NOT TARGET CUTCELLS::cutcells)
  set(CUTFEMX_LOCAL_CUTCELLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CutCells/cpp")
  if(EXISTS "${CUTFEMX_LOCAL_CUTCELLS_DIR}/CMakeLists.txt")
    add_subdirectory("${CUTFEMX_LOCAL_CUTCELLS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/cutcells" EXCLUDE_FROM_ALL)
    if(NOT TARGET CUTCELLS::cutcells)
      add_library(CUTCELLS::cutcells ALIAS cutcells)
    endif()
  else()
    find_package(CutCells REQUIRED)
  endif()
endif()
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
  add_subdirectory(demo/reinit)
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
write_basic_package_version_file(CUTFEMXConfigVersion.cmake VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)
configure_package_config_file(CUTFEMXConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/CUTFEMXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutfemx)

# Install CMake files
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CUTFEMXConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/CUTFEMXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutfemx COMPONENT Development)
install(EXPORT CUTFEMXTargets FILE CUTFEMXTargets.cmake NAMESPACE CUTFEMX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutfemx)

