# Level Set Functions

Level set functions provide an elegant way to represent complex geometries implicitly without mesh generation. Under a level set function, we understand a scalar valued function with the following properties. 

<div class="math-box">
<h4>Level Set Function </h4>

$$
\begin{cases}
\phi(\mathbf{x}) < 0 & \Rightarrow \mathbf{x} \text{ is inside the domain } \Omega \\
\phi(\mathbf{x}) = 0 & \Rightarrow \mathbf{x} \text{ is on the boundary } \partial\Omega \\
\phi(\mathbf{x}) > 0 & \Rightarrow \mathbf{x} \text{ is outside the domain}
\end{cases}
$$
</div>

This definitiom allows us to represent any closed domain $\Omega$ as the negative level set of a function $\phi$.
Geometrical properties of the domain boundary $\partial \Omega$ can be determined from the level set function. 

<div class="math-box">
<h4>Normal and curvature </h4>

The normal on $\partial \Omega$ pointing outward from $\Omega$ is given by 

$$
\mathbf{n}_{\phi} = \frac{\nabla \phi}{|\nabla \phi|}
$$

and the mean curvature of the boundary is given by 

$$
\kappa = \nabla \cdot \mathbf{n}_{\phi}
$$
</div>


### Signed Distance Functions

In order to have well defined normals and curvature along $\partial \Omega$ a level set function needs to be sufficiently smooth around its zero level set. In practice, a signed distance property near the boundary is desirable, i.e. $|\nabla\phi| \approx 1$. This garantues that normals can be computed accurately (division by $|\nabla \phi|$) and that the level set profile is steep enough to make a computational reconstruction of the zero level set contour line accurate as well.

Hence, the ideal level set function is a signed distance function:

$$\phi(\mathbf{x}) = \pm d(\mathbf{x}, \partial\Omega)$$

where $d(\mathbf{x}, \partial\Omega)$ is the distance from point $\mathbf{x}$ to the boundary $\partial\Omega$.

**Properties of signed distance functions:**
- $|\nabla\phi| = 1$ everywhere
- Provides stable numerical behavior

## Basic Level Set Examples

### Simple Shapes

```python
import numpy as np

# Circle/Sphere
def circle(x, center=(0.0, 0.0), radius=0.5):
    """Level set for a circle/sphere"""
    return np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2) - radius

# Rectangle/Box using max norm
def rectangle(x, center=(0.0, 0.0), width=1.0, height=0.8):
    """Level set for a rectangle using max norm"""
    return np.maximum(
        np.abs(x[0] - center[0]) - width/2, 
        np.abs(x[1] - center[1]) - height/2
    )

# Ellipse
def ellipse(x, center=(0.0, 0.0), a=0.5, b=0.3):
    """Level set for an ellipse with semi-axes a and b"""
    return ((x[0] - center[0])/a)**2 + ((x[1] - center[1])/b)**2 - 1.0

# Plane/Half-space
def plane(x, normal=(1.0, 0.0), point=(0.0, 0.0)):
    """Level set for a plane defined by normal and point"""
    return normal[0] * (x[0] - point[0]) + normal[1] * (x[1] - point[1])
```

### Complex Shapes

```python
# Star shape using polar coordinates
def star(x, n_points=5, outer_radius=0.5, inner_radius=0.2):
    """Level set for a star shape with n_points"""
    # Convert to polar coordinates
    r = np.sqrt(x[0]**2 + x[1]**2)
    theta = np.arctan2(x[1], x[0])
    
    # Star radius as function of angle
    angle_per_point = 2 * np.pi / n_points
    local_angle = np.mod(theta, angle_per_point)
    # Distance from center of star point
    t = np.abs(local_angle - angle_per_point/2) / (angle_per_point/2)
    star_radius = inner_radius + (outer_radius - inner_radius) * (1 - t)
    
    return r - star_radius

# Rounded rectangle
def rounded_rectangle(x, width=1.0, height=0.6, radius=0.1):
    """Level set for a rounded rectangle"""
    # Distance to rectangle edges
    dx = np.maximum(np.abs(x[0]) - width/2 + radius, 0.0)
    dy = np.maximum(np.abs(x[1]) - height/2 + radius, 0.0)
    
    # Distance to rounded corners
    corner_dist = np.sqrt(dx**2 + dy**2)
    
    # Combine with interior distance
    interior_dist = np.maximum(
        np.abs(x[0]) - width/2 + radius,
        np.abs(x[1]) - height/2 + radius
    )
    
    return np.where(dx > 0, corner_dist - radius, interior_dist)
```

## Boolean Operations

Level set functions naturally support boolean operations to create complex geometries:

### Union (OR operation)
```python
def union_shapes(x):
    """Level set for union of multiple shapes"""
    circle1 = np.sqrt((x[0] + 0.3)**2 + x[1]**2) - 0.2
    circle2 = np.sqrt((x[0] - 0.3)**2 + x[1]**2) - 0.2
    return np.minimum(circle1, circle2)  # Union = minimum
```

### Intersection (AND operation)
```python
def intersection(x):
    """Level set for intersection of circle and rectangle"""
    circle_phi = np.sqrt(x[0]**2 + x[1]**2) - 0.5
    box_phi = np.maximum(np.abs(x[0]) - 0.4, np.abs(x[1]) - 0.4)
    return np.maximum(circle_phi, box_phi)  # Intersection = maximum
```

### Difference (NOT operation)
```python
def difference(x):
    """Level set for circle with rectangular hole"""
    circle_phi = np.sqrt(x[0]**2 + x[1]**2) - 0.5
    hole_phi = np.maximum(np.abs(x[0]) - 0.2, np.abs(x[1]) - 0.2)
    return np.maximum(circle_phi, -hole_phi)  # Difference = max(A, -B)
```

<div class="note-box">
<h4>Boolean Operations Summary</h4>

- **Union**: $\phi_{A \cup B} = \min(\phi_A, \phi_B)$
- **Intersection**: $\phi_{A \cap B} = \max(\phi_A, \phi_B)$
- **Difference**: $\phi_{A \setminus B} = \max(\phi_A, -\phi_B)$
- **Complement**: $\phi_{\bar{A}} = -\phi_A$
</div>

## Advanced Level Set Construction

### Smooth Blending
For smoother transitions between shapes:

```python
def smooth_union(phi1, phi2, k=0.1):
    """Smooth union using polynomial blending"""
    h = np.maximum(k - np.abs(phi1 - phi2), 0.0) / k
    return np.minimum(phi1, phi2) - h * h * h * k * (1.0/6.0)

def smooth_intersection(phi1, phi2, k=0.1):
    """Smooth intersection using polynomial blending"""
    h = np.maximum(k - np.abs(phi1 - phi2), 0.0) / k
    return np.maximum(phi1, phi2) + h * h * h * k * (1.0/6.0)
```

<!-- ### Parametric Level Sets
For geometries defined by parameters:

```python
class ParametricLevelSet:
    """Base class for parametric level set functions"""
    
    def __init__(self, **params):
        self.params = params
    
    def __call__(self, x):
        return self.evaluate(x, **self.params)
    
    def evaluate(self, x, **params):
        raise NotImplementedError
    
    def update_params(self, **new_params):
        self.params.update(new_params)

class MovingCircle(ParametricLevelSet):
    """Circle that can move and change radius"""
    
    def evaluate(self, x, center=(0.0, 0.0), radius=0.5):
        return np.sqrt((x[0] - center[0])**2 + (x[1] - center[1])**2) - radius

# Usage
moving_circle = MovingCircle(center=(0.0, 0.0), radius=0.3)
phi_t0 = moving_circle(x)  # At initial time

# Update for new time step
moving_circle.update_params(center=(0.1, 0.0), radius=0.35)
phi_t1 = moving_circle(x)  # At new time
``` -->

## CutFEMx Integration

### Creating DOLFINx Functions
```python
from dolfinx import fem

# Create function space
V = fem.functionspace(mesh, ("Lagrange", 1))

# Create level set function
phi = fem.Function(V)
phi.interpolate(circle)  # Use any level set function

# For time-dependent problems
phi_old = fem.Function(V)
phi_old.x.array[:] = phi.x.array[:]  # Copy current state
```

### Visualization and Debugging
```python
def visualize_level_set(phi, mesh, filename="levelset.pvd"):
    """Visualize level set function"""
    try:
        import pyvista as pv
        from dolfinx import plot
        
        # Create VTK data
        cells, types, x = plot.vtk_mesh(mesh, mesh.topology.dim)
        grid = pv.UnstructuredGrid(cells, types, x)
        
        # Add level set values
        grid.point_data["phi"] = phi.x.array
        grid.set_active_scalars("phi")
        
        # Create isosurface at phi=0
        contour = grid.contour(isosurfaces=[0.0])
        
        # Plot
        plotter = pv.Plotter()
        plotter.add_mesh(grid, opacity=0.3, show_edges=True)
        plotter.add_mesh(contour, color="red", line_width=5)
        plotter.show()
        
    except ImportError:
        print("PyVista not available for visualization")
```

## 🔗 Next Steps

- **[Element Classification](element-classification.md)** - Learn how CutFEMx uses level sets to classify elements
- **[Quick Start Tutorial](quickstart.md)** - Apply level set functions in a complete example
