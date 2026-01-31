cmake_minimum_required(VERSION 3.16)

# Set project name and version number
project(cutfemx VERSION 0.1.0 LANGUAGES CXX C)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies
find_package(MPI REQUIRED)
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind 2.0.0 REQUIRED CONFIG)

find_package(CutCells REQUIRED)
find_package(DOLFINX REQUIRED)
find_package(spdlog REQUIRED)

# Add C++ library
add_subdirectory(cpp)

# Find DOLFINX Python Wrappers (required for array.h etc included by wrappers/fem.cpp)
# Find DOLFINX Python Wrappers (required for array.h etc included by wrappers/fem.cpp)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "
import sys
try:
    import dolfinx, os, mpi4py, petsc4py
    # DOLFINX
    inc = dolfinx.get_include()
    if os.path.exists(os.path.join(inc, 'array.h')):
        print(inc, end=';')
    elif os.path.exists(os.path.join(inc, 'dolfinx', 'wrappers', 'array.h')):
         print(os.path.join(inc, 'dolfinx', 'wrappers'), end=';')
    else:
        path = list(dolfinx.__path__)[0]
        if os.path.exists(os.path.join(path, 'wrappers', 'array.h')):
             print(os.path.join(path, 'wrappers'), end=';')
        else:
             print(';', end=';')
    
    # MPI4PY
    print(mpi4py.get_include(), end=';')

    # PETSC4PY
    print(petsc4py.get_include(), end='')
except ImportError as e:
    print(f'IMPORT_ERROR: {e}', end='')
except Exception as e:
    print(f'ERROR: {e}', end='')
"
  OUTPUT_VARIABLE PYTHON_INCLUDES
  ERROR_VARIABLE PYTHON_STDERR
)

if(PYTHON_INCLUDES MATCHES "IMPORT_ERROR" OR PYTHON_INCLUDES MATCHES "ERROR")
    message(WARNING "Python module detection failed: ${PYTHON_INCLUDES}")
    message(STATUS "Stderr: ${PYTHON_STDERR}")
else()
    list(LENGTH PYTHON_INCLUDES LEN)
    if(LEN EQUAL 3)
        list(GET PYTHON_INCLUDES 0 DOLFINX_WRAPPERS_DIR)
        list(GET PYTHON_INCLUDES 1 MPI4PY_INCLUDE_DIR)
        list(GET PYTHON_INCLUDES 2 PETSC4PY_INCLUDE_DIR)
    else()
        message(WARNING "Could not parse python includes. Output: ${PYTHON_INCLUDES}")
    endif()
endif()

# Fallback: Try to use environment variables or verify found paths
if(NOT DOLFINX_WRAPPERS_DIR)
  if(DEFINED ENV{DOLFINX_DIR})
     set(DOLFINX_WRAPPERS_DIR "$ENV{DOLFINX_DIR}/lib/python3.13/site-packages/dolfinx/wrappers") 
     # This is a guess, probably wrong for conda
  endif()
  message(WARNING "Could not automatically locate DOLFINX wrappers containing array.h. Build WILL FAIL if headers are not found.")
else()
  message(STATUS "Found DOLFINX wrappers: ${DOLFINX_WRAPPERS_DIR}")
endif()

message(STATUS "Found mpi4py include: ${MPI4PY_INCLUDE_DIR}")
message(STATUS "Found petsc4py include: ${PETSC4PY_INCLUDE_DIR}")

# Create Python External Module
file(GLOB_RECURSE WRAPPER_SOURCES "python/cutfemx/wrappers/*.cpp")

nanobind_add_module(cutfemx_cpp ${WRAPPER_SOURCES} NOMINSIZE)

target_include_directories(cutfemx_cpp PRIVATE 
    ${DOLFINX_WRAPPERS_DIR}
    ${MPI4PY_INCLUDE_DIR}
    ${PETSC4PY_INCLUDE_DIR}
)

target_link_libraries(cutfemx_cpp PRIVATE
    cutfemx
    MPI::MPI_CXX
    dolfinx
    CutCells
)

# Install the extension into the python package
install(TARGETS cutfemx_cpp DESTINATION cutfemx)
