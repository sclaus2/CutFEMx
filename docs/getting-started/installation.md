# Installation

This page describes a source-based installation of the current CutFEMx stack:

- FEniCSx 0.11 release packages: Basix, UFL, FFCx, and DOLFINx
- `runintgen` 0.1
- CutCells 0.4
- CutFEMx 0.2

The commands below assume a Unix-like shell and an activated conda environment.
They avoid user-specific paths; all installation prefixes are taken from the
active environment through `$CONDA_PREFIX`.

## Prerequisites

You need:

- Python 3.10 or newer
- a C++20 compiler
- CMake and `pkg-config`
- MPI and PETSc through the same environment used by DOLFINx
- `pip`, `scikit-build-core`, and `nanobind`

For a conda-forge environment, create and activate an environment first:

Choose any environment name and keep it active for all install steps:

```bash
conda create -n <env-name> -c conda-forge python=3.13
conda activate <env-name>
```

Install the common build and runtime dependencies:

```bash
conda install -c conda-forge \
  c-compiler cxx-compiler cmake ninja pkg-config pip \
  mpi mpich mpi4py petsc petsc4py \
  hdf5 libadios2 pugixml spdlog boost-cpp \
  parmetis libscotch ptscotch kahip \
  numpy scipy matplotlib pyvista pytest \
  scikit-build-core nanobind cffi
```

If you use OpenMPI instead of MPICH, install a consistent MPI/PETSc stack from
conda-forge and keep the same environment active for every build step.

## Real and complex scalar builds

The CutFEMx MatrixCSR/runtime assembly path is templated for all supported PDE
scalar types (`float32`, `float64`, `complex64`, `complex128`) on either
`float32` or `float64` mesh geometry. PETSc is different: a PETSc installation
is built for exactly one scalar mode and one real precision. PETSc-backed
assembly and solvers therefore require an environment whose `petsc`, `petsc4py`,
DOLFINx, and CutFEMx builds all use the PETSc scalar type you want to solve
with.

The commands above install the usual real-valued, double-precision PETSc stack.
For a `complex128` PETSc setup, create a separate environment and request the
complex PETSc and petsc4py variants before building DOLFINx:

```bash
conda create -n <complex-env-name> -c conda-forge python=3.13
conda activate <complex-env-name>

conda install -c conda-forge \
  c-compiler cxx-compiler cmake ninja pkg-config pip \
  mpi mpich mpi4py "petsc=*=complex*" "petsc4py=*=complex*" \
  hdf5 libadios2 pugixml spdlog boost-cpp \
  parmetis libscotch ptscotch kahip \
  numpy scipy matplotlib pyvista pytest \
  scikit-build-core nanobind cffi
```

If the solver cannot find a matching `petsc4py` build with the wildcard above,
query conda-forge for the available build strings and select the build that is
compiled against complex PETSc:

```bash
conda search -c conda-forge petsc petsc4py
```

Verify the scalar mode before building any FEniCSx component from source:

```bash
python - <<'PY'
import numpy as np
from petsc4py import PETSc

print("PETSc scalar:", PETSc.ScalarType)
assert np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating)
assert np.dtype(PETSc.RealType) == np.dtype(np.float64)
PY
```

For a `complex64` PETSc setup, PETSc must be configured for complex scalars and
single real precision, and DOLFINx/petsc4py must be built against that same
PETSc. If conda-forge does not provide a matching single-precision complex
variant for your platform, build PETSc, petsc4py, DOLFINx, and CutFEMx from
source in one environment. Verify it with:

```bash
python - <<'PY'
import numpy as np
from petsc4py import PETSc

print("PETSc scalar:", PETSc.ScalarType)
print("PETSc real:", PETSc.RealType)
assert np.dtype(PETSc.ScalarType) == np.dtype(np.complex64)
assert np.dtype(PETSc.RealType) == np.dtype(np.float32)
PY
```

Then follow the same source build steps below. Build DOLFINx, runintgen,
CutCells, and CutFEMx in this complex environment. Reusing DOLFINx or CutFEMx
artifacts built against a different PETSc scalar or precision environment is
not supported.

Before building packages from source, keep CMake and Python discovery tied to
the active environment:

```bash
export CONDA_PREFIX="${CONDA_PREFIX:?activate the conda environment first}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export CC="$CONDA_PREFIX/bin/clang"
export CXX="$CONDA_PREFIX/bin/clang++"
```

On macOS with the conda-forge C++ toolchain, DOLFINx 0.11 may also require:

```bash
export CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=13.4"
export CXXFLAGS="-D_LIBCPP_ENABLE_EXPERIMENTAL -D_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB -D_LIBCPP_HAS_NO_EXPERIMENTAL_SYNCSTREAM -D_LIBCPP_HAS_NO_INCOMPLETE_PSTL"
```

On Linux, or with a compiler toolchain that does not need these libc++
workarounds, omit the `CMAKE_ARGS` and `CXXFLAGS` lines. If `clang`/`clang++`
are not the compiler names in your environment, set `CC` and `CXX` to the
compiler wrappers installed in the active conda environment.

## Install FEniCSx

CutFEMx currently targets the FEniCSx 0.11 source release stack:

- Basix: `v0.11.0`
- UFL: `2026.1.0`
- FFCx: `v0.11.0`
- DOLFINx: `v0.11.0`

UFL uses calendar-style release tags; its latest tag is `2026.1.0` rather than
a `v0.11.0` tag. These versions match the dependency bounds declared by
DOLFINx and FFCx `v0.11.0`.

Verify the upstream tags if needed:

```bash
git ls-remote --exit-code --tags https://github.com/FEniCS/basix.git refs/tags/v0.11.0
git ls-remote --exit-code --tags https://github.com/FEniCS/ufl.git refs/tags/2026.1.0
git ls-remote --exit-code --tags https://github.com/FEniCS/ffcx.git refs/tags/v0.11.0
git ls-remote --exit-code --tags https://github.com/FEniCS/dolfinx.git refs/tags/v0.11.0
```

Fetch the tagged sources:

```bash
mkdir -p fenicsx-0.11
cd fenicsx-0.11

git clone --branch v0.11.0 --depth 1 https://github.com/FEniCS/basix.git
git clone --branch 2026.1.0 --depth 1 https://github.com/FEniCS/ufl.git
git clone --branch v0.11.0 --depth 1 https://github.com/FEniCS/ffcx.git
git clone --branch v0.11.0 --depth 1 https://github.com/FEniCS/dolfinx.git
```

Install Basix, UFL, and FFCx from those checkouts:

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade --force-reinstall --no-deps ./basix ./ufl ./ffcx
```

DOLFINx needs its C++ core and Python interface. Follow the DOLFINx source
installation workflow for your platform, using the same active environment as
the install prefix. In a source checkout this has the following shape:

```bash
cd dolfinx

cmake -G Ninja -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build cpp/build
cmake --install cpp/build

python -m pip install --check-build-dependencies --no-build-isolation --no-deps \
  ./python

cd ../..
```

Verify the FEniCSx installation:

```bash
python - <<'PY'
import basix
import dolfinx
import ffcx
import ufl

print("Basix:", basix.__version__)
print("DOLFINx:", dolfinx.__version__)
print("FFCx:", ffcx.__version__)
print("UFL:", ufl.__version__)
PY
```

## Install runintgen

Install `runintgen` after Basix, UFL, and FFCx so it builds against the same
headers and Python packages:

```bash
git clone https://github.com/sclaus2/runintgen.git
cd runintgen

python -m pip install --no-build-isolation --no-deps --force-reinstall .

cd ..
```

Verify:

```bash
python - <<'PY'
import runintgen

print("runintgen:", getattr(runintgen, "__version__", "installed"))
PY
```

## Install CutCells

CutCells provides the geometric cutting backend used by CutFEMx. The Python
build can build the bundled C++ core, but building and installing the C++ core
explicitly is useful when developing or when CMake needs to discover the
CutCells package.

```bash
git clone https://github.com/sclaus2/cutcells.git CutCells
cd CutCells

cmake -G Ninja -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCUTCELLS_WITH_ALGOIM=ON
cmake --build cpp/build
cmake --install cpp/build

CMAKE_ARGS="${CMAKE_ARGS:-} -DCUTCELLS_WITH_ALGOIM=ON" \
  python -m pip install --no-build-isolation --no-deps --force-reinstall \
  ./python

cd ..
```

Verify:

```bash
python - <<'PY'
import cutcells

print("CutCells:", getattr(cutcells, "__version__", "installed"))
PY
```

## Install CutFEMx

Install CutFEMx last, after FEniCSx, runintgen, and CutCells are available in
the same environment:

```bash
git clone https://github.com/sclaus2/CutFEMx.git
cd CutFEMx

env CONDA_PREFIX="$CONDA_PREFIX" \
  CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  PYTHONNOUSERSITE=1 \
  CC="$CONDA_PREFIX/bin/clang" \
  CXX="$CONDA_PREFIX/bin/clang++" \
  CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=13.4 -DCUTCELLS_WITH_ALGOIM=ON" \
  CXXFLAGS="-D_LIBCPP_ENABLE_EXPERIMENTAL -D_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB -D_LIBCPP_HAS_NO_EXPERIMENTAL_SYNCSTREAM -D_LIBCPP_HAS_NO_INCOMPLETE_PSTL" \
  python -m pip install --no-build-isolation --no-deps --force-reinstall .
```

Use the same active conda environment and `pip` flags for `runintgen`,
`CutCells/python`, and CutFEMx. If the build variables above are already
exported, the shorter `python -m pip install ...` form is equivalent. Omit
`-DCUTCELLS_WITH_ALGOIM=ON` only if you intentionally want to build without the
optional Algoim quadrature backend.

`--no-build-isolation` is required because CutFEMx must use the active
FEniCSx/MPI/PETSc/runintgen stack and DOLFINx wrapper headers. `--no-deps`
prevents `pip` from replacing the pinned FEniCSx packages with incompatible
release packages.

Verify:

```bash
python - <<'PY'
import cutcells
import cutfemx
import dolfinx
import runintgen

print("DOLFINx:", dolfinx.__version__)
print("runintgen:", getattr(runintgen, "__version__", "installed"))
print("CutCells:", getattr(cutcells, "__version__", "installed"))
print("CutFEMx:", getattr(cutfemx, "__version__", "installed"))
PY
```

## Build the documentation

Install the documentation dependencies from the CutFEMx repository root:

```bash
python -m pip install -r docs/requirements.txt
```

Build the HTML documentation:

```bash
cd docs
env LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 MPLCONFIGDIR=.matplotlib-cache \
  python -m sphinx -b html . _build/html
```

If `sphinx-build` is already on your `PATH`, the shorter command is enough:

```bash
make html
```

The generated site is written to `docs/_build/html/index.html`.

## Common issues

### DOLFINx or Basix cannot be found by CMake

Make sure the same conda environment is active for all steps and pass the active
prefix to CMake:

```bash
cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" ...
```

### `pip` tries to replace FEniCSx packages

Use `--no-deps` when installing runintgen, CutCells, and CutFEMx from source.
The FEniCSx packages must stay on the same tagged stack.

### Build isolation cannot find MPI or DOLFINx headers

Use `--no-build-isolation`. The builds need the compiler, MPI, PETSc, DOLFINx,
and Basix from the active environment.

## Next steps

- [Quickstart tutorial](quickstart.md)
- [User guide](../user-guide/index.md)
