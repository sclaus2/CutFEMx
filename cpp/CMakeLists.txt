cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------------------------
# Set project name and version number
project(CUTFEMX VERSION "0.2.0" LANGUAGES CXX C)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build cutfemx with shared libraries." ON)
option(CUTFEMX_USE_MULTIPRECISION_PSM "Use MultiPrecision_psm library for high precision arithmetic." ON)

# Find DOLFINx config file
if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(NOT TARGET CUTCELLS::cutcells)
  find_package(CutCells 0.4.0 REQUIRED CONFIG)
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
