cmake_minimum_required(VERSION 3.21)

project(cutfemx_nanobind)

if(WIN32)
  # Windows requires all symbols to be manually exported. This flag exports all
  # symbols automatically, as in Unix.
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif(WIN32)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(DOLFINX REQUIRED CONFIG)

if(DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

#Check for DOLFINx python wrappers
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import dolfinx; print(dolfinx.get_include())"
  OUTPUT_VARIABLE DOLFINX_PY_WRAPPERS_DIR
  RESULT_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_RESULT
  ERROR_VARIABLE DOLFINX_PY_WRAPPERS_COMMAND_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT DOLFINX_PY_WRAPPERS_COMMAND_RESULT)
  message(STATUS "Found DOLFINx python wrappers at ${DOLFINX_PY_WRAPPERS_DIR}")
else()
  message(FATAL_ERROR "DOLFINx python wrappers could not be found.")
endif()

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(CutCells REQUIRED CONFIG)
if(CutCells_FOUND)
  message(STATUS "Found CutCells at ${CutCells_DIR}")
endif()


find_package(CUTFEMX REQUIRED CONFIG)
if(CUTFEMX_FOUND)
  message(STATUS "Found CutFEMx at ${CUTFEMX_DIR}")
endif()


# Create the binding library nanobind handles its own calls to
# target_link_libraries
nanobind_add_module(
  cutfemx_cpp
  NOMINSIZE
  cutfemx/wrappers/cutfemx.cpp
  cutfemx/wrappers/level_set.cpp
  cutfemx/wrappers/quadrature.cpp
  cutfemx/wrappers/mesh.cpp
  cutfemx/wrappers/fem.cpp
  cutfemx/wrappers/petsc.cpp
)
target_compile_definitions(cutfemx_cpp PRIVATE cxx_std_20)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(cutfemx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
  target_compile_definitions(cutfemx_cpp PRIVATE HAS_PETSC4PY)
else()
  message(STATUS "petsc4py not found.")
endif()

# Check for mpi4py
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
  target_include_directories(cutfemx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

# Define HAS_PETSC4PY for compatibility with DOLFINx python wrappers
target_compile_definitions(cutfemx_cpp PRIVATE HAS_PETSC4PY)

# Add DOLFINx libraries
target_link_libraries(cutfemx_cpp PRIVATE dolfinx)
target_link_libraries(cutfemx_cpp PRIVATE CUTCELLS::cutcells)
target_link_libraries(cutfemx_cpp PRIVATE cutfemx)

# Add DOLFINx python, petsc4py and mpi4py include directories (with DOLFINx C++
# ones already being added by target_link_libraries)
target_include_directories(cutfemx_cpp PRIVATE ${DOLFINX_PY_WRAPPERS_DIR})
target_include_directories(cutfemx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
target_include_directories(cutfemx_cpp PRIVATE ${MPI4PY_INCLUDE_DIR})

# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(cutfemx_cpp PRIVATE bcrypt)
endif()


set_target_properties(cutfemx_cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cutfemx_cpp DESTINATION cutfemx)
