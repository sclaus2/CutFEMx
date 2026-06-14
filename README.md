<p align="center">
<img src="img/cutfemx_logo.png" alt="CutFEMx Logo" style="width:50%;" />
</p>

CutFEMx is a cut finite element library for FEniCSx. It supports cut finite
element methods as described in

```bibtex
@article{CutFEM2015,
  title={CutFEM: discretizing geometry and partial differential equations},
  author={Burman, Erik and Claus, Susanne and Hansbo, Peter and Larson, Mats G and Massing, Andr{\'e}},
  journal={International Journal for Numerical Methods in Engineering},
  volume={104},
  number={7},
  pages={472--501},
  year={2015},
  publisher={Wiley Online Library}
}
```

The current version targets the FEniCSx 0.11 release stack
(Basix/UFL/FFCx/DOLFINx), with runtime quadrature provided by
[runintgen](https://github.com/sclaus2/runintgen) and cut geometry provided by
[CutCells](https://github.com/sclaus2/cutcells).
The CutFEMx 0.2 release line is intended to be used with runintgen 0.1 and
CutCells 0.4.

## License

CutFEMx is licensed under the MIT License except for files that carry different
SPDX identifiers. DOLFINx-derived form and assembler sources under
`cpp/dolfinx_custom_data` are licensed under LGPL-3.0-or-later, and the bundled
geogram.psm.MultiPrecision source is BSD-3-Clause. See
`THIRD_PARTY_NOTICES.md` for provenance and details.

![SVG Image](img/quadrature.svg)

![image info](img/cut_poisson_2.png)
Poisson problem in a circular domain described by a level set function.

## Features

- **Unfitted discretization**: solve PDEs on geometries defined by level-set
  functions without remeshing.
- **Runtime quadrature**: generate quadrature rules on cut entities with
  CutCells and assemble them through runintgen.
- **Stabilization**: ghost-penalty stabilization for robustness with small cut
  cells.
- **Parallel support**: compatible with MPI-distributed FEniCSx workflows.

The documentation is available at
[https://sclaus2.github.io/CutFEMx/](https://sclaus2.github.io/CutFEMx/).

## Installation

These instructions install the FEniCSx 0.11 stack from conda-forge, then build
runintgen, CutCells, and CutFEMx from source into the same environment.

### 1. Create a FEniCSx 0.11 environment

```bash
conda create -n cutfemx -c conda-forge \
  python=3.12 \
  fenics-dolfinx=0.11.0 fenics-basix=0.11.0 \
  fenics-ffcx=0.11.0 fenics-ufl=2026.1.0 \
  mpi4py petsc4py \
  cmake ninja pkg-config compilers \
  scikit-build-core nanobind cffi \
  numpy scipy matplotlib pytest
conda activate cutfemx
```

Tie CMake, Python, and the compiler to the active environment before building
any of the source packages:

```bash
export CONDA_PREFIX="${CONDA_PREFIX:?activate the conda environment first}"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
export PYTHONNOUSERSITE=1
export CC="$CONDA_PREFIX/bin/clang"
export CXX="$CONDA_PREFIX/bin/clang++"
```

On macOS with the conda-forge C++ toolchain and DOLFINx 0.11 headers, also use:

```bash
export CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=13.4"
export CXXFLAGS="-D_LIBCPP_ENABLE_EXPERIMENTAL -D_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB -D_LIBCPP_HAS_NO_EXPERIMENTAL_SYNCSTREAM -D_LIBCPP_HAS_NO_INCOMPLETE_PSTL"
```

On Linux, or with a compiler toolchain that does not need these libc++
workarounds, omit the `CMAKE_ARGS` and `CXXFLAGS` lines. If `clang` and
`clang++` are not the compiler names in your environment, set `CC` and `CXX` to
the compiler wrappers installed by conda-forge.

Verify that the conda packages are the expected FEniCSx versions:

```bash
python - <<'PY'
import importlib.metadata as m

for name in ["fenics-basix", "fenics-dolfinx", "fenics-ffcx", "fenics-ufl"]:
    print(name, m.version(name))
PY
```

### 2. Install runintgen

runintgen is independent of CutCells and can be installed first:

```bash
git clone https://github.com/sclaus2/runintgen.git
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  ./runintgen
```

Check the install:

```bash
python - <<'PY'
import importlib.metadata as m
import runintgen

print("runintgen", m.version("runintgen"))
PY
```

### 3. Install CutCells

CutCells is independent of runintgen. Install the C++ library into the active
conda prefix, then build the Python wrapper against that installed library:

```bash
git clone https://github.com/sclaus2/cutcells.git CutCells
cd CutCells

cmake -G Ninja -S cpp -B cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.4 \
  -DCUTCELLS_WITH_ALGOIM=ON
cmake --build cpp/build
cmake --install cpp/build

python -m pip install --no-build-isolation --no-deps --force-reinstall \
  ./python

cd ..
```

On Linux, omit the `-DCMAKE_OSX_DEPLOYMENT_TARGET=13.4` line from the direct
CutCells CMake configure command.

When Algoim support is enabled, the CutCells CMake configuration looks for
BLAS/LAPACK in `CMAKE_PREFIX_PATH` before falling back to system locations. If
you need to override that choice, set `CUTCELLS_ALGOIM_LAPACK_LIBRARIES`.

Check the install:

```bash
python - <<'PY'
import importlib.metadata as m
import cutcells

print("cutcells", m.version("cutcells"))
PY
```

### 4. Install CutFEMx

CutFEMx depends on both runintgen and the installed CutCells C++/Python
packages:

```bash
git clone https://github.com/sclaus2/CutFEMx.git
python -m pip install --no-build-isolation --no-deps --force-reinstall \
  ./CutFEMx
```

`--no-build-isolation` is required because the build must use the active conda
environment's FEniCSx, MPI/PETSc, runintgen, and CutCells installations.
`--no-deps` prevents `pip` from replacing that tested stack.

Verify the complete stack:

```bash
python - <<'PY'
import importlib.metadata as m
import cutcells
import cutfemx
import dolfinx
import runintgen

for name in ["runintgen", "cutcells", "cutfemx", "fenics-dolfinx"]:
    print(name, m.version(name))
PY
```

## Building The Documentation

The documentation is a Sphinx/MyST project in `docs/`. Use the same conda
environment that contains CutFEMx and the FEniCSx stack, especially when
regenerating tutorial figures.

Install the documentation dependencies once:

```bash
cd CutFEMx/docs
python -m pip install -r requirements.txt
```

Build the HTML documentation:

```bash
bash build_docs.sh clean
bash build_docs.sh build
```

The generated site is written to `docs/_build/html/index.html`. To serve it
locally:

```bash
bash build_docs.sh serve --port 8000
```

Tutorial screenshots and interactive HTML views are generated from the actual
demo-style CutFEMx solves in `docs/tutorials/make_pyvista_scenes.py`. Regenerate
them before building when the tutorials or demo visualizations change:

```bash
bash build_docs.sh visuals
bash build_docs.sh build
```

Equivalently, generate visuals as part of the build:

```bash
CUTFEMX_DOCS_GENERATE_VISUALS=1 bash build_docs.sh build
```

You can also select a conda environment without activating it first:

```bash
CUTFEMX_DOCS_CONDA_ENV=<env-name> bash build_docs.sh build
```

For a direct Sphinx command, run this from `docs/`:

```bash
make html
```

## Available Demos

The `python/demo` directory contains examples illustrating the usage of CutFEMx:

- **Poisson equation** (`demo_poisson.py`): solve a cut Poisson problem on a
  flower-shaped domain with Nitsche terms and ghost-penalty stabilization.
- **Cut DG Poisson** (`demo_dg_poisson.py`): solve a DG Poisson problem with
  runtime quadrature on cells and the active interior skeleton.
- **Interface Poisson** (`demo_interface_poisson.py`): solve an interface
  problem across level-set-defined subdomains.
- **Moving domain** (`demo_moving_poisson.py`): update quadrature rules and
  integration domains as the interface moves.
- **Exterior-facet runtime quadrature** (`demo_boundary_sphere_perimeter.py`):
  integrate on cut exterior facets.
- **Distance from STL** (`demo_stl_distance.py`): compute a signed distance
  field from an STL surface.
- **Level-set reinitialization** (`demo_reinit.py`): convert a distorted level
  set into a signed distance field.

## Geometry and Level Sets

CutFEMx provides tools for handling complex geometries through level-set
functions. The library can generate distance fields from standard STL files and
adaptively refine a background mesh around an STL surface.

<p align="center">
  <i>Example: signed distance field computed from a generated demo STL surface on an adaptively refined mesh using `demo_stl_distance.py`.</i>
</p>
