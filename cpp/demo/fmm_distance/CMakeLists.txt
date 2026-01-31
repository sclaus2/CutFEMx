cmake_minimum_required(VERSION 3.19)
project(demo_fmm_distance LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED COMPONENTS CXX C)
find_package(Boost REQUIRED COMPONENTS timer)
find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NOT TARGET cutfemx)
  find_package(CutFEMx REQUIRED)
endif()

add_executable(demo_fmm_distance main.cpp)

target_link_libraries(demo_fmm_distance PRIVATE 
    cutfemx 
    dolfinx 
    Basix::basix 
    MPI::MPI_CXX)

# Automatically generate test data
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sphere.stl
  COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/generate_sphere.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_sphere.py
  COMMENT "Generating sphere.stl test data"
)

# Ensure data is generated before running demo
add_custom_target(generate_data ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/sphere.stl)
add_dependencies(demo_fmm_distance generate_data)
