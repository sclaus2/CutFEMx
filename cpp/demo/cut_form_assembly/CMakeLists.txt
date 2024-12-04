cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME demo_cut_form_assembly)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Set standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-Wall -Wextra -pedantic)

find_package(CUTCELLS REQUIRED)
if(CUTCELLS_FOUND)
  message(STATUS "Found CutCells at ${CutCells_DIR}")
endif()

find_package(DOLFINX REQUIRED)
if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

find_package(CUTFEMX REQUIRED)
if(CUTFEMX_FOUND)
  message(STATUS "Found CutFEMx at ${CUTFEMX_DIR}")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME} main.cpp scalar.c)
target_link_libraries(${PROJECT_NAME} CUTCELLS::cutcells)
target_link_libraries(${PROJECT_NAME} dolfinx)
target_link_libraries(${PROJECT_NAME} cutfemx)
