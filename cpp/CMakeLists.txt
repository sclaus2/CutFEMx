cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------------------------
# Set project name and version number
project(CUTFEMX VERSION "0.1" LANGUAGES CXX C)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_SHARED_LIBS "Build cutfemx with shared libraries." ON)
#add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build cutfemx with shared libraries.")

include(FeatureSummary)

# Find DOLFINx config file
find_package(CutCells REQUIRED)
find_package(DOLFINX REQUIRED)

# Enable testing
#enable_testing()

# Source files
add_subdirectory(cutfemx)

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




