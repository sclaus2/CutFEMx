# Installation Guide

This guide will walk you through installing CutFEMx and its dependencies step by step.

## Prerequisites

CutFEMx requires:
- **CutCells** 
- **FEniCSx 0.9.0** with custom ffcx runtime
- **Python 3.8+**
- **C++20 compatible compiler**
- **MPI** for parallel computation
- **PETSc** for linear algebra (optional but recommended) 

and other FEniCSx related dependencies. 

## Quick Installation (Conda - Recommended)

The easiest way to install CutFEMx is using conda to manage dependencies:

### 1. Create Environment

```bash
conda create -n cutfemx
conda activate cutfemx
```

### 2. Install Build Tools

```bash
conda install -c conda-forge cxx-compiler cmake python pkg-config pip nanobind scikit-build-core
```

### 3. Install Dependencies

```bash
# Scientific computing
conda install -c conda-forge numpy scipy sympy numba pyvista pytest spdlog

# Linear algebra
conda install -c conda-forge blas blas-devel lapack libblas libcblas liblapack liblapacke libtmglib

# Parallel computing and PETSc/SLEPc
conda install -c conda-forge mpi mpich kahip libboost-devel parmetis libscotch libptscotch pugixml
conda install -c conda-forge mpi4py petsc4py slepc4py 
conda install -c conda-forge 'hdf5=*=mpi*' 'petsc=*=*real*' 'slepc=*=*real*' 'libadios2=*=mpi*'
```

### 4. Install FEniCSx with Custom FFCx

1. Install Basix:
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
    ```

2. Install UFL:
    ```bash
    git clone https://github.com/FEniCS/ufl.git
    cd ufl
    git checkout 2024.2.0
    pip install .
    ```

3. Install runtime integral extended FFCX:
    ```bash
    git clone git@github.com:sclaus2/ffcx-runtime-0.9.0.git
    cd ffcx
    pip install .
    ```

4. Install DOLFINx:
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

5. Install CutCells:
    ```bash
    git clone git@github.com:sclaus2/cutcells.git

    cd cutcells/cpp
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -B build-dir -S .
    cmake --build build-dir
    cmake --install build-dir

    cd ../python
    pip install .
    ```

6. Install CutFEMx:
    ```bash
    git clone https://github.com/sclaus2/CutFEMx

    cd CutFEMx/cpp
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -B build-dir -S .
    cmake --build build-dir
    cmake --install build-dir

    cd ../python
    pip install -r build-requirements.txt
    pip install --no-build-isolation .
    ```

## Verification

Test your installation:

```python
import cutfemx
import dolfinx
import numpy as np

# Create a simple test
print(f"CutFEMx version: {cutfemx.__version__}")
print(f"DOLFINx version: {dolfinx.__version__}")

```

## Common Issues

### Issue: FFCx Runtime Not Found
```bash
# Solution: Ensure the custom FFCx is installed first
pip uninstall ffcx
pip install git+https://github.com/sclaus2/ffcx-runtime-0.9.0.git
```

### Issue: PETSc Version Conflicts
```bash
# Solution: Use conda to manage PETSc versions
conda install -c conda-forge 'petsc=*=*real*' --force-reinstall
```

### Issue: MPI Compilation Errors
```bash
# Solution: Ensure MPI compilers are available
export CC=mpicc
export CXX=mpicxx
```

## Next Steps

- **[Quick Start Tutorial](quickstart.md)** - Create your first CutFEMx program
<!-- - **[Basic Concepts](concepts.md)** - Understand cut finite element methods
- **[Examples Gallery](../examples/)** - Explore working examples -->

## Getting Help

- **GitHub Issues**: [Report bugs or ask questions](https://github.com/sclaus2/CutFEMx/issues)
<!-- - **Documentation**: [User Guide](../user-guide/) for detailed usage -->
