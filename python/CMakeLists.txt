cmake_minimum_required(VERSION 3.21)

# Include version information
include(${CMAKE_CURRENT_SOURCE_DIR}/../VERSION.cmake)

project(cutfemx_nanobind VERSION ${CUTFEMX_VERSION})

# Configure Python version file
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cutfemx/_version.py.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/cutfemx/_version.py"
    @ONLY
)

message(STATUS "Pythonpath $ENV{PYTHONPATH}")

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
# --- BEGIN: DOLFINx Python wrappers detection (robust & MPI-safe) ---
# Accept a user-provided path, or ENV var, or auto-detect via importlib (no import side effects).
# Also ensure we use the same Python interpreter as the build backend.
cmake_minimum_required(VERSION 3.21)  # you already require this or higher

# Find Python interpreter (scikit-build-core provides it, but be explicit)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# Cache var so -DDOLFIN_PY_WRAPPERS_DIR=... is recorded & visible in GUIs
set(DOLFINX_PY_WRAPPERS_DIR "${DOLFINX_PY_WRAPPERS_DIR}" CACHE PATH "Path to DOLFINx Python wrappers (dolfinx/wrappers)")

# If not provided, try environment
if(NOT DOLFINX_PY_WRAPPERS_DIR AND DEFINED ENV{DOLFINX_PY_WRAPPERS_DIR})
  set(DOLFINX_PY_WRAPPERS_DIR "$ENV{DOLFINX_PY_WRAPPERS_DIR}")
endif()

# If still not set, auto-detect WITHOUT importing dolfinx (so mpi4py won't call MPI_Init)
if(NOT DOLFINX_PY_WRAPPERS_DIR)
  # Use cmake -E env to set MPI4PY_RC_INITIALIZE=0, then Python -c to locate package
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env MPI4PY_RC_INITIALIZE=0
            "${Python_EXECUTABLE}" -c
            "import importlib.util, pathlib, sys; s=importlib.util.find_spec('dolfinx'); (print((pathlib.Path(s.origin).parent/'wrappers')) if s and s.origin else sys.exit(2))"
    OUTPUT_VARIABLE _WRAP_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _WRAP_ERR
    RESULT_VARIABLE _WRAP_RC
  )
  if(_WRAP_RC EQUAL 0 AND EXISTS "${_WRAP_DIR}")
    set(DOLFINX_PY_WRAPPERS_DIR "${_WRAP_DIR}")
  else()
    message(FATAL_ERROR
      "DOLFINx python wrappers could not be found.\n"
      "  Set -DDOLFINX_PY_WRAPPERS_DIR=/path/to/site-packages/dolfinx/wrappers\n"
      "  or export DOLFINX_PY_WRAPPERS_DIR in the environment.\n"
      "  Auto-detect error: ${_WRAP_ERR}")
  endif()
endif()

message(STATUS "Found DOLFINx python wrappers at ${DOLFINX_PY_WRAPPERS_DIR}")


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
  cutfemx/wrappers/extensions.cpp
  cutfemx/wrappers/petsc.cpp
)
target_compile_definitions(cutfemx_cpp PRIVATE cxx_std_20)

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_VARIABLE PETSC4PY_COMMAND_ERROR OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(cutfemx_cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
  target_compile_definitions(cutfemx_cpp PRIVATE HAS_PETSC4PY)
else()
  message(STATUS "petsc4py not found with error: ${PETSC4PY_COMMAND_ERROR}")
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
target_link_libraries(cutfemx_cpp PRIVATE CUTFEMX::cutfemx)

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
