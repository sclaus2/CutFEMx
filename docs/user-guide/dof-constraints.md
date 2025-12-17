# DOF Constraints

Degree of freedom (DOF) constraints in CutFEMx are handled through the extensions module, which provides tools for managing cut cells and applying constraints to maintain well-posed problems.

## Overview

CutFEMx provides constraint functionality through:

1. **Extension Operations**: Extending functions from interior cells to cut cells
2. **Cut Cell Identification**: Finding badly cut cells that need special treatment
3. **Matrix Operations**: Tools for applying constraints to linear systems
4. **Basis Function Evaluation**: Evaluating basis functions at DOF coordinates

The primary tools are in the `cutfemx.extensions` module.

## Extension Operations

### Function Extension
CutFEMx provides extension operators to handle cut cells by extending values from interior cells:

```python
import cutfemx.extensions as ext

# Extend a function from interior to cut cells
extended_function = ext.extend_function(
    original_function,
    root_mapping,
    source_dofs,
    target_dofs
)
```

### Root Mapping
The root mapping connects cut cells to interior cells for extension operations:

```python
# Build mapping from cut cells to interior cells
root_mapping = ext.build_root_mapping(
    badly_cut_parents,
    interior_cells, 
    mesh,
    cut_cells
)

# Create DOF-level mappings
dof_root_mapping = ext.build_dof_to_root_mapping(
    root_mapping,
    function_space
)
```

## Cut Cell Management

### Identifying Badly Cut Cells
CutFEMx can identify cells that are poorly cut and need special treatment:

```python
# Find cells with small cut fractions
badly_cut_indices = ext.get_badly_cut_parent_indices(
    cut_cells,
    mesh_size_h=0.1,      # Characteristic mesh size
    threshold_factor=0.1   # Volume threshold (10% of h)
)
```

### DOF-Cell Mappings
Utility functions for managing DOF relationships:

```python
# Build mapping from DOFs to cells
dof_to_cells = ext.build_dof_to_cells_mapping(function_space)

# Create root mapping for cell indices
cell_root_map = ext.create_root_to_cell_map(
    root_mapping,
    function_space
)
```

## Basis Function Evaluation

### Evaluating at DOF Coordinates
A key feature for constraint application:

```python
# Evaluate basis functions at DOF coordinates
basis_values = ext.evaluate_basis_at_dof_coordinates(
    function_space,
    cut_cells,
    cut_dofs
)

# This returns coefficients for expressing cut DOFs 
# in terms of interior DOFs
```

## Constraint Application

### Matrix Constraint Application
Using the constraint information from basis evaluation:

```python
# Apply constraints to a matrix system
constraints = {
    cut_dof_id: [(interior_dof_1, coeff_1), (interior_dof_2, coeff_2)],
    # ... more constraints
}

# Apply constraints to matrix and vector
ext.apply_constraints(A, b, constraints)
```

### Example: Complete Constraint Workflow
```python
import numpy as np
import dolfinx
import cutfemx.extensions as ext
from cutfemx.mesh import create_cut_mesh
from cutfemx.level_set import locate_entities

# 1. Create mesh and identify cut cells
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Level set function (e.g., circle)
def level_set(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.5

# Find cut entities
cut_entities = locate_entities(mesh, level_set)

# 2. Identify badly cut cells
badly_cut = ext.get_badly_cut_parent_indices(
    cut_cells, 
    mesh_size_h=mesh.hmax(),
    threshold_factor=0.1
)

# 3. Build root mapping for extension
interior_cells = [i for i in range(mesh.topology.index_map(2).size_local) 
                  if i not in cut_entities]
                  
root_mapping = ext.build_root_mapping(
    badly_cut,
    interior_cells,
    mesh,
    cut_cells
)

# 4. Get constraint coefficients
constraints = ext.evaluate_basis_at_dof_coordinates(V, cut_cells, cut_dofs)

# 5. Apply to linear system
ext.apply_constraints(A, b, constraints)
```

## Integration with Cut Mesh

### Working with Cut Meshes
The constraint functionality works with CutFEMx's cut mesh capabilities:

```python
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature

# Create cut mesh from cut cells
cut_mesh = create_cut_mesh(MPI.COMM_WORLD, cut_cells, background_mesh, part="negative")

# Generate runtime quadrature
quad_rules = runtime_quadrature(level_set_function, "negative", order=2)

# Apply constraints in the context of cut finite element assembly
# (constraints help ensure stability and proper boundary conditions)
```

## Practical Considerations

### Parameter Selection
- **Threshold factor**: Typically 0.1 (10% of mesh size) for identifying badly cut cells
- **Extension strategy**: Root mapping connects cut cells to nearby interior cells
- **Constraint tolerance**: Depends on problem conditioning and desired accuracy

### Workflow Integration
1. **Mesh setup**: Create background mesh and define level set
2. **Cut identification**: Find cut cells and identify problematic ones
3. **Mapping creation**: Build root mappings for extension operations
4. **Constraint generation**: Use basis evaluation to get constraint coefficients
5. **System modification**: Apply constraints to maintain well-posedness

## Examples

See the demonstration scripts in the `python/demo/` directory:

- `demo_constraint_application.py` - Basic constraint application workflow
- `demo_dof_basis_evaluation.py` - Basis function evaluation at DOF coordinates  
- `demo_extension_operators.py` - Function extension operations
- `demo_badly_cut_identification.py` - Identifying and handling badly cut cells

## Next Steps

- **[Quadrature Rules](quadrature.md)** - Learn about integration on cut elements
- **[Cut Meshes](cut-meshes.md)** - Understanding cut mesh generation  
- **[Examples Gallery](../examples/)** - See complete constraint examples

---

*CutFEMx provides the essential tools for handling constraints in cut finite element methods, focusing on extension operators and proper treatment of cut cells.*
