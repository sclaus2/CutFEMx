cmake_minimum_required(VERSION 3.5)

project(cutfemx-tests LANGUAGES C CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find DOLFINx config file
find_package(DOLFINX REQUIRED)
find_package(CUTFEMX REQUIRED)

add_custom_command(
  OUTPUT scalar.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/scalar.py
  VERBATIM
  DEPENDS scalar.py
  COMMENT "Compile scalar.py using FFCx"
)

find_package(Catch2 3)

if(NOT Catch2_FOUND)
  message(STATUS "Catch2 not found. Downloading.")
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
  )
  FetchContent_MakeAvailable(Catch2)
endif()

# These tests can use the Catch2-provided main
add_executable(tests
main.cpp
test.cpp
${CMAKE_CURRENT_BINARY_DIR}/scalar.c)

target_link_libraries(tests PRIVATE Catch2::Catch2 dolfinx cutfemx)

target_include_directories(tests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_compile_options(tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror -Wextra -pedantic>)

# Enable testing
enable_testing()

# Test target
add_test(tests tests)
