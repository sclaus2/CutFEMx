# Element Classification

Element classification is a crucial step in cut finite element methods, determining how each background mesh element interacts with the implicit geometry defined by level set functions.

## Overview

CutFEMx classifies background mesh elements based on their relationship to the level set function $\phi(\mathbf{x})$:

<div class="workflow-box">
<h4>Classification Process</h4>

1. **Evaluate** level set function in each cell dof
2. **Analyze** sign patterns to determine element type
3. **Generate** lists of elements for each category
</div>

## Element Types

### Inside Elements
Elements completely within the computational domain:

<div class="success-box">
<h4>Inside Elements</h4>

- **Condition**: $\phi(\mathbf{x}_i) < 0$ in all element dofs
- **Treatment**: Standard finite element assembly
- **Quadrature**: Standard Gaussian quadrature
- **Stability**: No special treatment needed
</div>

### Outside Elements
Elements completely outside the computational domain:

<div class="note-box">
<h4>Outside Elements</h4>

- **Condition**: $\phi(\mathbf{x}_i) > 0$ in all element dofs
- **Treatment**: excluded from computation for fictitious domain problems
- **DOFs**: deactivated for ficitious domain problems
</div>

### Cut Elements
Elements intersected by the domain boundary:

<div class="warning-box">
<h4>Cut Elements</h4>

- **Condition**: $\phi(\mathbf{x}_i)$ changes sign, i.e. has both positive and negative values in element dofs
- **Treatment**: Requires specialized handling
- **Quadrature**: Runtime-generated quadrature rules
- **Stability**: May need ghost penalty stabilization
</div>

## CutFEMx Classification Functions

### Element lists `locate_entities`

```python
from cutfemx.level_set import locate_entities
from dolfinx import fem

# Create level set function
def circle(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.5

# Define as DOLFINx function
phi = fem.Function(V)
phi.interpolate(circle)

# Locate different element types
tdim = mesh.topology.dim
inside_cells = locate_entities(phi, tdim, "phi<0")
outside_cells = locate_entities(phi, tdim, "phi>0")
intersected_cells = locate_entities(phi, tdim, "phi=0")
```

### Cutting intersected cells `cut_entities`

```python
from cutfemx.level_set import cut_entities

# Get DOF coordinates for cut element analysis
dof_coordinates = V.tabulate_dof_coordinates()

# Process cut elements to determine interior portions
cut_cells = cut_entities(phi, dof_coordinates, intersected_cells, tdim, "phi<0")
```

## Constraint Specifications

CutFEMx uses string expressions to specify different parts of the domain:

<div class="algorithm-box">
<h4>String Syntax</h4>

- **`"phi<0"`**: Elements inside domain
- **`"phi>0"`**: Elements outside domain
- **`"phi=0"`**: Elements cut by boundary
</div>

## 🔧 Practical Implementation

### Complete Classification Workflow

```python
def complete_element_classification(mesh, phi, V):
    """
    Complete element classification workflow
    """
    tdim = mesh.topology.dim
    
    # Step 1: Basic classification
    inside_cells = locate_entities(phi, tdim, "phi<0")
    outside_cells = locate_entities(phi, tdim, "phi>0")
    intersected_cells = locate_entities(phi, tdim, "phi=0")
    
    # Step 2: Create element mappings
    element_types = {}
    for cell in inside_cells:
        element_types[cell] = 'inside'
    for cell in outside_cells:
        element_types[cell] = 'outside'
    for cell in intersected_cells:
        element_types[cell] = 'cut'
    
    return {
        'inside': inside_cells,
        'outside': outside_cells,
        'cut': intersected_cells,
        'element_types': element_types
    }
```

### Visualization of Classification

```python
def visualize_element_classification(mesh, classification, filename="classification.pvd"):
    """
    Visualize element classification results
    """
    try:
        import pyvista as pv
        from dolfinx import plot
        
        # Create mesh for visualization
        cells, types, x = plot.vtk_mesh(mesh, mesh.topology.dim)
        grid = pv.UnstructuredGrid(cells, types, x)
        
        # Create classification array
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        class_array = np.zeros(num_cells)
        
        # Assign classification values
        class_array[classification['inside']] = 1  # Inside = 1
        class_array[classification['cut']] = 2    # Cut = 2
        class_array[classification['outside']] = 3  # Outside = 3
        
        # Add to grid
        grid.cell_data["classification"] = class_array
        grid.set_active_scalars("classification")
        
        # Create plot
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, cmap="viridis")
        plotter.add_scalar_bar(title="Element Type")
        plotter.show()
        
    except ImportError:
        print("PyVista not available for visualization")
```

## 📊 Statistics and Quality Metrics

### Classification Statistics

```python
def classification_statistics(classification):
    """
    Compute statistics about element classification
    """
    n_inside = len(classification['inside'])
    n_outside = len(classification['outside'])
    n_cut = len(classification['cut'])
    n_total = n_inside + n_outside + n_cut
    
    stats = {
        'total_elements': n_total,
        'inside_elements': n_inside,
        'outside_elements': n_outside,
        'cut_elements': n_cut,
        'inside_fraction': n_inside / n_total,
        'outside_fraction': n_outside / n_total,
        'cut_fraction': n_cut / n_total
    }
    
    return stats

def print_classification_report(stats):
    """
    Print classification report
    """
    print(f"Element Classification Report:")
    print(f"  Total elements: {stats['total_elements']}")
    print(f"  Inside elements: {stats['inside_elements']} ({stats['inside_fraction']:.1%})")
    print(f"  Outside elements: {stats['outside_elements']} ({stats['outside_fraction']:.1%})")
    print(f"  Cut elements: {stats['cut_elements']} ({stats['cut_fraction']:.1%})")
```

## 🔍 Common Issues and Solutions

### Numerical Precision

```python
# Problem: Level set values very close to zero
def handle_precision_issues(phi_values, tol=1e-12):
    """
    Handle numerical precision issues in classification
    """
    # Snap values close to zero
    phi_clean = np.where(np.abs(phi_values) < tol, 0.0, phi_values)
    
    # Ensure consistent classification
    return phi_clean
```

## 🔗 Next Steps

- **[Numerical Integration](integration.md)** - Understand quadrature on cut elements

---

*Element classification is the foundation for all subsequent cut finite element operations. Understanding this process is crucial for successful CutFEMx applications.*
