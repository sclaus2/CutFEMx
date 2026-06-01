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

The current development version targets the FEniCSx main-branch stack
(Basix/UFL/FFCx/DOLFINx), with runtime quadrature provided by
[runintgen](https://github.com/sclaus2/runintgen) and cut geometry provided by
[CutCells](https://github.com/sclaus2/cutcells).

## License

CutFEMx is licensed under the MIT License except for DOLFINx-derived form and
assembler sources under `cpp/dolfinx_custom_data`, which are licensed under
LGPL-3.0-or-later. See `THIRD_PARTY_NOTICES.md` for provenance and details.

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

## Installation

These instructions assume a Unix-like shell and an activated conda-forge
environment. They intentionally avoid user-specific paths; all CMake installs
use the active environment prefix through `$CONDA_PREFIX`.

### 1. Create an environment

```bash
conda create -n cutfemx-dev -c conda-forge python=3.13
conda activate cutfemx-dev

conda install -c conda-forge \
  c-compiler cxx-compiler cmake ninja pkg-config pip \
  mpi mpich mpi4py petsc petsc4py \
  hdf5 libadios2 pugixml spdlog boost-cpp \
  parmetis libscotch ptscotch kahip \
  numpy scipy matplotlib pyvista pytest \
  scikit-build-core nanobind cffi
```

Use one consistent MPI/PETSc stack for every build step. If you use OpenMPI
instead of MPICH, install matching MPI, PETSc, `mpi4py`, and `petsc4py`
packages from conda-forge.

### 2. Install FEniCSx development packages

CutFEMx currently targets:

- `fenics-basix >= 0.11.0.dev0`
- `fenics-ufl >= 2025.3.0.dev0`
- `fenics-ffcx >= 0.11.0.dev0`
- `fenics-dolfinx >= 0.11.0.dev0`

Install Basix, UFL, and FFCx from the FEniCS `main` branches:

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade --force-reinstall --no-deps \
  "fenics-basix @ git+https://github.com/FEniCS/basix.git@main" \
  "fenics-ufl @ git+https://github.com/FEniCS/ufl.git@main" \
  "fenics-ffcx @ git+https://github.com/FEniCS/ffcx.git@main"
```

Build and install DOLFINx from source into the same environment:

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

### 3. Install runintgen

```bash
git clone https://github.com/sclaus2/runintgen.git
cd runintgen

python -m pip install --no-build-isolation --no-deps --force-reinstall .

cd ..
```

### 4. Install CutCells

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

### 5. Install CutFEMx

```bash
git clone https://github.com/sclaus2/CutFEMx.git
cd CutFEMx

python -m pip install --no-build-isolation --no-deps --force-reinstall .
```

`--no-build-isolation` is required because CutFEMx must use the active
FEniCSx/MPI/PETSc/runintgen stack and DOLFINx wrapper headers. `--no-deps`
prevents `pip` from replacing development FEniCSx packages with incompatible
release packages.

Verify the installation:

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

## Available Demos

The `python/demo` directory contains examples illustrating the usage of CutFEMx:

- **Poisson equation** (`demo_poisson.py`): solve a cut Poisson problem on a
  circular domain with Nitsche terms and ghost-penalty stabilization.
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
  <img src="img/dino.png" alt="Dino STL distance field" width="45%" />
  <img src="img/dino2.png" alt="Dino STL distance field (sliced)" width="45%" />
</p>
<p align="center">
  <i>Example: signed distance field computed from a dino STL surface on an adaptively refined mesh using `demo_stl_distance.py`.</i>
</p>
