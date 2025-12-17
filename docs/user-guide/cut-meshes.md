# Cut Mesh Generation

Cut meshes are the foundation of CutFEMx's embedded boundary approach. Instead of generating conforming meshes around complex geometries, CutFEMx uses a simple background mesh that is "cut" by level set functions representing the domain boundaries.

## Overview

Cut finite element methods work by:

1. **Background Mesh**: Start with a simple structured or unstructured mesh that covers the entire computational domain
2. **Level Set Function**: Define geometry implicitly using a level set function φ(x) where:
   - φ(x) < 0: inside the domain
   - φ(x) = 0: on the boundary
   - φ(x) > 0: outside the domain
3. **Cut Elements**: Identify elements that intersect the boundary (where φ changes sign)
4. **Integration**: Use specialized quadrature rules for cut elements

## Basic Cut Mesh Example

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh, create_cut_cells_mesh

# Create background mesh
N = 32
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle
)

# Create function space for level set
V = fem.functionspace(msh, ("Lagrange", 1))

# Define a circle level set function
def circle(x):
    radius = 0.5
    return np.sqrt(x[0]**2 + x[1]**2) - radius

# Interpolate level set onto function space
level_set = fem.Function(V)
level_set.interpolate(circle)

# Get mesh dimensions
tdim = msh.topology.dim  # Topological dimension
gdim = msh.geometry.dim  # Geometric dimension

# Locate entities that intersect the boundary
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

# Get DOF coordinates for cutting algorithm
dof_coordinates = V.tabulate_dof_coordinates()

# Cut the intersected entities to create sub-elements
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")

# Create cut mesh for the inside domain
cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)

# Create interface mesh (elements on the boundary)
interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")
interface_mesh = create_cut_cells_mesh(msh.comm, interface_cells)

print(f"Found {len(intersected_entities)} intersected elements")
print(f"Generated {len(cut_cells.cut_cells)} cut cells")
```

## Level Set Functions

### Implicit Geometry Definition

CutFEMx supports various ways to define level set functions as Python functions:

#### Analytical Functions
```python
# Sphere in 3D
def sphere(x):
    radius = 1.0
    return np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) - radius

# Box/Rectangle using signed distance
def box(x):
    half_width, half_height = 0.5, 0.3
    return np.maximum(np.abs(x[0]) - half_width, np.abs(x[1]) - half_height)

# Ellipse
def ellipse(x):
    a, b = 0.8, 0.4  # Semi-axes
    return (x[0]/a)**2 + (x[1]/b)**2 - 1.0

# Union of shapes (minimum for union)
def union_shapes(x):
    circle = np.sqrt(x[0]**2 + x[1]**2) - 0.3
    box = np.maximum(np.abs(x[0] - 0.5) - 0.2, np.abs(x[1]) - 0.1)
    return np.minimum(circle, box)

# Intersection (maximum for intersection)
def intersection_shapes(x):
    circle = np.sqrt(x[0]**2 + x[1]**2) - 0.5
    box = np.maximum(np.abs(x[0]) - 0.3, np.abs(x[1]) - 0.3)
    return np.maximum(circle, box)
```

#### Using DOLFINx Functions
```python
from dolfinx import fem

# Create function space and level set function
V = fem.functionspace(msh, ("Lagrange", 1))
level_set = fem.Function(V)

# Interpolate analytical function
level_set.interpolate(circle)

# Or set values directly for more complex cases
level_set.x.array[:] = computed_values
```

## Entity Location and Classification

CutFEMx provides functions to locate different types of entities:

```python
from cutfemx.level_set import locate_entities

# Locate different entity types
intersected_cells = locate_entities(level_set, tdim, "phi=0")  # Cut elements
inside_cells = locate_entities(level_set, tdim, "phi<0")       # Inside domain
outside_cells = locate_entities(level_set, tdim, "phi>0")     # Outside domain

# For strict inequalities
inside_strict = locate_entities(level_set, tdim, "phi<=0")    # Inside + boundary
outside_strict = locate_entities(level_set, tdim, "phi>=0")   # Outside + boundary

print(f"Intersected cells: {len(intersected_cells)}")
print(f"Inside cells: {len(inside_cells)}")
print(f"Outside cells: {len(outside_cells)}")
```

## Working with Cut Cells

The `cut_entities` function returns a `CutCells` object containing the geometric information:

```python
# Cut the intersected entities
cut_cells = cut_entities(level_set, dof_coordinates, intersected_cells, tdim, "phi<0")

# Access cut cell information
print(f"Number of cut cells: {len(cut_cells.cut_cells)}")
print(f"Parent cell mapping: {cut_cells.parent_map}")

# Each cut cell contains:
# - Vertex coordinates
# - Connectivity information
# - Cell types
for i, cell in enumerate(cut_cells.cut_cells):
    parent_id = cut_cells.parent_map[i]
    print(f"Cut cell {i} comes from parent cell {parent_id}")
```

## Advanced Features

### Time-Dependent Geometries

For moving boundaries, update the level set and regenerate cut meshes:

```python
class MovingCircle:
    def __init__(self, velocity=0.1):
        self.velocity = velocity
        self.time = 0.0
    
    def update_time(self, t):
        self.time = t
    
    def __call__(self, x):
        center_x = self.velocity * self.time
        return np.sqrt((x[0] - center_x)**2 + x[1]**2) - 0.2

# Create moving geometry
moving_circle = MovingCircle(velocity=0.1)

# Time stepping loop
dt = 0.1
for t in np.arange(0, 1.0, dt):
    moving_circle.update_time(t)
    
    # Update level set
    level_set.interpolate(moving_circle)
    
    # Regenerate cut mesh
    intersected_entities = locate_entities(level_set, tdim, "phi=0")
    if len(intersected_entities) > 0:
        cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
        cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)
```

### Ghost Penalty and Interface Handling

For stabilization and interface handling:

```python
from cutfemx.level_set import ghost_penalty_facets, facet_topology, compute_normal

# Get facets for ghost penalty stabilization
gp_facets = ghost_penalty_facets(level_set, "phi<0")
gp_topology = facet_topology(msh, gp_facets)

# Compute interface normals
V_vec = fem.functionspace(msh, ("DG", 0, (gdim,)))
interface_normal = fem.Function(V_vec)
compute_normal(interface_normal, level_set, intersected_entities)

print(f"Ghost penalty facets: {len(gp_facets)}")
```

## Integration with Quadrature

Cut meshes work seamlessly with CutFEMx quadrature rules:

```python
from cutfemx.quadrature import runtime_quadrature
import ufl

# Create quadrature rules for cut elements
order = 2
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)

# Define measures for integration
quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]
dx_cut = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

# Standard measures for background mesh
dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities), (1, intersected_cells)], domain=msh)
```

## Best Practices

### Level Set Function Design
1. **Smooth Functions**: Use smooth level sets to avoid integration issues
2. **Proper Scaling**: Ensure level set magnitude is reasonable relative to mesh size
3. **Avoid Zero Gradients**: Maintain non-zero gradients for robust cutting

### Mesh Considerations
1. **Resolution**: Ensure background mesh adequately resolves boundary features
2. **Element Quality**: Monitor cut element quality for very small volume fractions
3. **Adaptive Strategies**: Consider mesh refinement near boundaries

### Performance Tips
1. **Reuse Computations**: Cache DOF coordinates and level set evaluations
2. **Update Only When Needed**: Only recompute cut mesh when geometry changes
3. **Vectorization**: Use NumPy operations in level set functions

## Example: Complete Workflow

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, io

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh

# Create mesh and function space
msh = mesh.create_rectangle(MPI.COMM_WORLD, [(-1, -1), (1, 1)], [32, 32])
V = fem.functionspace(msh, ("Lagrange", 1))

# Define level set
def flower(x):
    r = np.sqrt(x[0]**2 + x[1]**2)
    theta = np.arctan2(x[1], x[0])
    return r - (0.5 + 0.2 * np.cos(5 * theta))

# Create level set function
level_set = fem.Function(V)
level_set.interpolate(flower)

# Generate cut mesh
tdim = msh.topology.dim
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

dof_coordinates = V.tabulate_dof_coordinates()
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)

# Export for visualization
with io.XDMFFile(msh.comm, "cut_mesh.xdmf", "w") as file:
    file.write_mesh(cut_mesh._mesh)
    file.write_function(level_set)

print("Cut mesh generation complete!")
```

## Next Steps

- **[DOF Constraints](dof-constraints.md)** - Learn how to apply constraints on cut boundaries
- **[Quadrature Rules](quadrature.md)** - Understand integration on cut elements
- **[FEM Assembly](../api/fem.md)** - Assemble forms on cut meshes

---

*Cut mesh generation is the foundation of cut finite element methods. The implicit geometry representation through level sets provides flexibility while maintaining computational efficiency.*

