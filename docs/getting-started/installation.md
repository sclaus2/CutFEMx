# Installation

This page describes a source-based installation of the current CutFEMx
development stack:

- FEniCSx development packages: Basix, UFL, FFCx, and DOLFINx
- `runintgen`
- CutCells
- CutFEMx

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

```bash
conda create -n cutfemx-dev -c conda-forge python=3.13
conda activate cutfemx-dev
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

## Install FEniCSx

CutFEMx currently targets the FEniCSx development stack:

- `fenics-basix >= 0.11.0.dev0`
- `fenics-ufl >= 2025.3.0.dev0`
- `fenics-ffcx >= 0.11.0.dev0`
- `fenics-dolfinx >= 0.11.0.dev0`

Install the Python packages from the FEniCS `main` branches:

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade --force-reinstall --no-deps \
  "fenics-basix @ git+https://github.com/FEniCS/basix.git@main" \
  "fenics-ufl @ git+https://github.com/FEniCS/ufl.git@main" \
  "fenics-ffcx @ git+https://github.com/FEniCS/ffcx.git@main"
```

DOLFINx needs its C++ core and Python interface. Follow the DOLFINx source
installation workflow for your platform, using the same active environment as
the install prefix. In a source checkout this has the following shape:

```bash
git clone https://github.com/FEniCS/dolfinx.git
cd dolfinx

cmake -G Ninja -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build cpp/build
cmake --install cpp/build

python -m pip install --check-build-dependencies --no-build-isolation --no-deps \
  ./python

cd ..
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
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build cpp/build
cmake --install cpp/build

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

python -m pip install --no-build-isolation --no-deps --force-reinstall .
```

`--no-build-isolation` is required because CutFEMx must use the active
FEniCSx/MPI/PETSc/runintgen stack and DOLFINx wrapper headers. `--no-deps`
prevents `pip` from replacing development FEniCSx packages with incompatible
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
env LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 MPLCONFIGDIR=/tmp/cutfemx-mplconfig \
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
The FEniCSx packages must stay on the same development stack.

### Build isolation cannot find MPI or DOLFINx headers

Use `--no-build-isolation`. The builds need the compiler, MPI, PETSc, DOLFINx,
and Basix from the active environment.

## Next steps

- [Quickstart tutorial](quickstart.md)
- [User guide](../user-guide/index.md)
