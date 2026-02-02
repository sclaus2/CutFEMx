<p align="center">
<img src="img/cutfemx_logo.png" alt="CutFEMx Logo" style="width:50%;" />
</p>

CutFEMx is a cut finite element library for FEniCSx. 
It is developed to support cut finite element methods as described in 

```bash
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

The current version supports FEniCSx 0.9 with a customised ffcx version for runtime quadrature at [ffcx-runtime](https://github.com/sclaus2/ffcx-runtime-0.9.0).

![SVG Image](img/quadrature.svg)

![image info](img/cut_poisson_2.png)
Poisson problem in a circular domain described by a level set function.

## Features
- **Unfitted Discretization**: Solve PDEs on geometries defined by level set functions without remeshing.
- **Runtime Quadrature**: Automatically generates quadrature rules on cut elements using the [CutCells](https://github.com/sclaus2/cutcells) library and a custom FFCX backend.
- **Stabilization**: Implements **Ghost Penalty** stabilization to ensure condition number robustness independent of cut geometry..
- **Parallel Support**: Fully compatible with MPI for large-scale distributed simulations.

## Installation Instructions

The CutFEMx library requires a FEniCSx installation version 0.9.0 with an extended version of ffcx from here `git clone git@github.com:sclaus2/ffcx-runtime-0.9.0.git` . CutFEMx also requires CutCells. The installation instructions using conda to manage the dependencies are detailed below. Make sure to use python 3.12. 

1. Create and activate a new conda environment:
    ```bash
    conda create -n cutfemx python=3.12
    conda activate cutfemx
    ```

2. Install general packages required for the build:
    ```bash
    conda install -c conda-forge cxx-compiler cmake python pkg-config pip nanobind
    ```

3. Install dependencies:
    ```bash
    conda install -c conda-forge numpy scipy sympy numba pyvista pytest
    conda install -c conda-forge blas blas-devel lapack libblas libcblas liblapack liblapacke libtmglib
    conda install -c conda-forge mpi mpich kahip libboost-devel parmetis libscotch libptscotch pugixml spdlog ccache
    conda install -c conda-forge mpi4py petsc4py slepc4py scikit-build-core 
    conda install -c conda-forge 'hdf5=*=mpi*' 'petsc=*=*real*' 'slepc=*=*real*' 'libadios2=*=mpi*'
    ```

4. Install Basix:
    ```bash
    git clone git@github.com:FEniCS/basix.git
    cd basix
    git checkout v0.9.0

    cd cpp
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -B build-dir -S .
    cmake --build build-dir
    cmake --install build-dir
    
    cd ../python
    pip install .
    cd ../..
    ```

5. Install UFL:
    ```bash
    git clone https://github.com/FEniCS/ufl.git
    cd ufl
    git checkout 2024.2.0
    pip install .
    cd ..
    ```

6. Install runtime integral extended FFCX:
    ```bash
    git clone git@github.com:sclaus2/ffcx-runtime-0.9.0.git
    cd ffcx-runtime-0.9.0
    pip install .
    cd ..
    ```

7. Install DOLFINx:
    ```bash
    git clone git@github.com:FEniCS/dolfinx.git
    cd dolfinx
    git checkout v0.9.0

    cd cpp
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -B build-dir -S .
    cmake --build build-dir
    cmake --install build-dir

    cd ../python
    pip install -r build-requirements.txt
    pip install --check-build-dependencies --no-build-isolation .
    ```

8. Install CutCells:
    ```bash
    git clone git@github.com:sclaus2/cutcells.git

    cd cutcells/cpp
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -B build-dir -S .
    cmake --build build-dir
    cmake --install build-dir

    cd ../python
    pip install .
    ```

9. Install CutFEMx:
    ```bash
    git clone https://github.com/sclaus2/CutFEMx
    cd CutFEMx

    # 1. Install C++ library and headers (Required for C++ demos/tests)
    cd cpp
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -B build-dir -S .
    cmake --build build-dir
    cmake --install build-dir
    cd ..

    # 2. Install Python interface
    python -m pip install --check-build-dependencies --no-build-isolation -v .
    ```

## Available Demos

The `python/demo` directory contains several examples illustrating the usage of CutFEMx:

- **Poisson Equation** (`demo_cut_poisson.py`): Solving the Poisson equation on a circular domain using Nitsche's method and ghost penalty stabilization.
- **Stokes Flow** (`demo_stokes.py`): Simulating Stokes flow around a cylinder using P1-P1 elements. Features include:
  - Global Continuous Interior Penalty (CIP) stabilization for pressure.
  - Ghost penalty stabilization for velocity.
  - VTX output for parallel visualization.
- **Moving Domain** (`demo_moving_poisson.py`): solving a time-dependent problem with a moving interface, demonstrating how to update quadrature rules and integration domains dynamically at each time step.
- **Distance from STL** (`demo_stl_distance.py`): Computing a signed distance field from an arbitrary STL surface using a parallel Fast Marching Method. Automatically refining a background mesh around an STL surface to improve geometric resolution.
- **Level Set Reinitialization** (`demo_reinit.py`): Converting a distorted level set function (e.g., parabolic) into a clean signed distance field using parallel FMM.

## Geometry and Level Sets

CutFEMx provides robust tools for handling complex geometries via level set functions. The library can generate distance fields from standard STL files and adaptively refine the mesh around them.

<p align="center">
  <img src="img/dino.png" alt="Dino STL distance field" width="45%" />
  <img src="img/dino2.png" alt="Dino STL distance field (sliced)" width="45%" />
</p>
<p align="center">
  <i>Example: Signed distance field computed from a dino STL surface on an adaptively refined mesh using `demo_stl_distance.py`.</i>
</p>
