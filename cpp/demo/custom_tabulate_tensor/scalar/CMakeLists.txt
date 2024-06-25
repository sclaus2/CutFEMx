cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME demo_custom_tabulate_tensor)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Set standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Debug)
add_compile_options(-Wall -Wextra -pedantic)

find_package(DOLFINX REQUIRED)
find_package(CUTFEMX REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME} main.cpp scalar.c)
target_link_libraries(${PROJECT_NAME} dolfinx)
target_link_libraries(${PROJECT_NAME} cutfemx)
