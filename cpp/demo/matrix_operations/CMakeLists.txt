cmake_minimum_required(VERSION 3.16)

project(matrix_operations_demo)

# Find required packages
find_package(DOLFINX REQUIRED)
find_package(spdlog REQUIRED)

# Check for PETSc
if(DOLFINX_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSC REQUIRED PETSc)
    add_compile_definitions(HAS_PETSC)
endif()

# Create the demo executable
add_executable(matrix_operations_demo matrix_operations_demo.cpp)

# Link libraries
target_link_libraries(matrix_operations_demo 
    dolfinx
    spdlog::spdlog
)

if(DOLFINX_ENABLE_PETSC)
    target_link_libraries(matrix_operations_demo ${PETSC_LIBRARIES})
    target_include_directories(matrix_operations_demo PRIVATE ${PETSC_INCLUDE_DIRS})
endif()

# Include directories
target_include_directories(matrix_operations_demo PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../  # For cutfemx headers
    ${DOLFINX_INCLUDE_DIRS}
)

# Set C++ standard
set_target_properties(matrix_operations_demo PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(matrix_operations_demo PRIVATE -Wall -Wextra)
endif()
