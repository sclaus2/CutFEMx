# Visualization Guide

Visualization is essential for understanding and validating cut finite element solutions. CutFEMx provides comprehensive visualization capabilities through PyVista integration, enabling interactive 3D plotting of cut meshes, solution fields, quadrature points, and analysis results.

This guide demonstrates the visualization techniques used throughout CutFEMx, from basic mesh plotting to advanced multi-view comparisons and specialized visualizations for cut element analysis.

## Prerequisites

<div class="note-box">
<h4>Required Dependencies</h4>

All CutFEMx visualization examples require PyVista for 3D plotting:

```bash
pip install pyvista
```

Optional dependencies for enhanced functionality:
- **matplotlib**: Color analysis and additional plotting options
- **vtk**: Advanced VTK operations (usually installed with PyVista)
</div>

## Core Visualization Workflow

### Basic Setup Pattern

All CutFEMx visualizations follow a consistent pattern:

```python
import numpy as np
from mpi4py import MPI

# CutFEMx imports
from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points

# DOLFINx imports  
from dolfinx import fem, mesh, plot

# Visualization
try:
    import pyvista
except ModuleNotFoundError:
    print("PyVista is required for visualization")
    exit(0)

# Create plotter
plotter = pyvista.Plotter()  # Single plot
# OR
plotter = pyvista.Plotter(shape=(1, 3))  # Multi-subplot layout
```

### VTK Mesh Conversion

CutFEMx uses DOLFINx's VTK mesh conversion for PyVista compatibility:

```python
# Convert different mesh components to VTK format
cells, types, x = plot.vtk_mesh(V)  # Function space mesh
grid = pyvista.UnstructuredGrid(cells, types, x)

# For cut mesh visualization
ccells, ctypes, cx = plot.vtk_mesh(cut_mesh._mesh)
cgrid = pyvista.UnstructuredGrid(ccells, ctypes, cx)

# For facet visualization (ghost penalty, boundaries)
facets, ftypes, fx = plot.vtk_mesh(msh, tdim - 1, facet_ids)
fgrid = pyvista.UnstructuredGrid(facets, ftypes, fx)
```

## Essential Visualization Types

### 1. Cut Mesh Visualization

The foundation of CutFEMx visualization is displaying the cut mesh geometry:

<div class="code-box">
<h4>Basic Cut Mesh Display</h4>

```python
# Generate cut mesh
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

dof_coordinates = V.tabulate_dof_coordinates()
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)

# Visualize cut mesh
ccells, ctypes, cx = plot.vtk_mesh(cut_mesh._mesh)
cgrid = pyvista.UnstructuredGrid(ccells, ctypes, cx)

plotter = pyvista.Plotter()
plotter.add_mesh(cgrid, show_edges=True, color="lightblue", opacity=0.8)
plotter.add_title("Cut Mesh Geometry")
plotter.view_xy()
plotter.show()
```
</div>

<div class="math-box">
<h4>Cut Mesh Generation Process</h4>

The cut mesh represents the intersection of the computational domain with the background mesh:

$$\Omega_h^{\text{cut}} = \bigcup_{K \in \mathcal{T}_h} K \cap \{\mathbf{x} : \phi(\mathbf{x}) < 0\}$$

where:
- $\mathcal{T}_h$: Background triangulation
- $\phi(\mathbf{x})$: Level set function
- $K \cap \{\mathbf{x} : \phi(\mathbf{x}) < 0\}$: Cut portion of element $K$
</div>

### 2. Solution Field Visualization

Displaying scalar and vector solutions on cut meshes:

```python
# Attach solution data to mesh
grid.point_data["u"] = u_sol
grid.set_active_scalars("u")

# Basic solution plot
plotter.add_mesh(grid, scalars="u", show_edges=True, show_scalar_bar=True)

# Warped solution (3D effect)
warped = grid.warp_by_scalar(scale_factor=10.0)
plotter.add_mesh(warped, scalars="u", show_edges=True, show_scalar_bar=True)
```

### 3. Quadrature Point Visualization

Runtime quadrature points are crucial for understanding integration accuracy:

<div class="algorithm-box">
<h4>Quadrature Point Display</h4>

```python
# Generate runtime quadrature
order = 2
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)

# Get physical coordinates
points_inside = physical_points(inside_quadrature, msh)
points_interface = physical_points(interface_quadrature, msh)

# Visualize quadrature points
plotter.add_points(
    points_inside, 
    render_points_as_spheres=True, 
    color="black", 
    point_size=8,
    label="Volume Quadrature"
)

plotter.add_points(
    points_interface, 
    render_points_as_spheres=True, 
    color="red", 
    point_size=10,
    label="Interface Quadrature"
)
```
</div>

## Multi-View Visualization Layouts

### Three-Panel Standard Layout

The most common CutFEMx visualization pattern uses three synchronized views:

<div class="example-box">
<h4>Standard Three-Panel Layout</h4>

```python
plotter = pyvista.Plotter(shape=(1, 3))

# Panel 1: Cut mesh with quadrature points
plotter.subplot(0, 0)
plotter.add_mesh(cgrid, show_edges=True, color="lightgray")
plotter.add_points(points_phys, render_points_as_spheres=True, color="black", point_size=8)
plotter.add_title("Cut Mesh with Quadrature Points")
plotter.view_xy()

# Panel 2: Mesh overlay with ghost penalty facets
plotter.subplot(0, 1)
plotter.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.7, label="Background")
plotter.add_mesh(cgrid, show_edges=True, color="blue", opacity=0.9, label="Cut Mesh")
plotter.add_mesh(fgrid, show_edges=True, line_width=5, color="red", label="Ghost Penalty")
plotter.add_title("Ghost Penalty Facets and Cut Mesh")
plotter.view_xy()

# Panel 3: Solution visualization
plotter.subplot(0, 2)
warped = grid.warp_by_scalar(scale_factor=10.0)
plotter.add_mesh(warped, scalars="u", show_edges=True, show_scalar_bar=True)
plotter.add_title("Solution (Warped)")
plotter.camera_position = "iso"

plotter.show()
```
</div>

### Adaptive Panel Layouts

For complex analyses, create custom layouts based on the number of components to display:

```python
# 2x2 grid for four-way comparison
plotter = pyvista.Plotter(shape=(2, 2))

# Dynamic subplot selection
n_views = 4
for i in range(n_views):
    row, col = divmod(i, 2)
    plotter.subplot(row, col)
    # Add specific visualization for each panel
```

## Advanced Visualization Techniques

### 1. Level Set Contour Visualization

Display the implicit geometry definition:

```python
# Add level set contour to show geometry boundary
grid.point_data["level_set"] = level_set.x.array
contour_mesh = grid.contour([0], scalars="level_set")
plotter.add_mesh(contour_mesh, color="black", line_width=3, label="Geometry Boundary")
```

<div class="note-box">
<h4>Contour Interpretation</h4>

Level set contours reveal the implicit geometry:
- **Zero contour ($\phi = 0$)**: Exact boundary of computational domain
- **Negative contours ($\phi < 0$)**: Interior iso-surfaces
- **Positive contours ($\phi > 0$)**: Exterior iso-surfaces
</div>

### 2. DOF and Root Cell Mapping

For extension operator visualization:

<div class="code-box">
<h4>DOF-to-Root Mapping Visualization</h4>

```python
# Build root mapping (from extension operators demo)
from cutfemx.extensions import build_root_mapping, build_dof_to_root_mapping

# Generate color mapping for root cells
unique_root_cells = list(set(dof_to_root_mapping.values()))
root_cell_to_color = {cell: i + 1 for i, cell in enumerate(unique_root_cells)}

# Color cells by root assignment
root_cell_colors = np.zeros(num_cells)
for cell_id in unique_root_cells:
    if cell_id < num_cells:
        root_cell_colors[cell_id] = root_cell_to_color[cell_id]

grid.cell_data["Root_Colors"] = root_cell_colors

# Color DOFs by their root cell assignment
dof_colors = np.array([root_cell_to_color[dof_to_root_mapping[dof]] 
                       for dof in range(len(dof_coordinates))])

# Create DOF point cloud
dof_point_cloud = pyvista.PolyData(dof_coordinates)
dof_point_cloud.point_data["Root_Cell_Color"] = dof_colors

# Visualization with consistent coloring
COLOR_PALETTE = "hsv"  # Excellent for many distinct colors
plotter.add_mesh(grid, scalars="Root_Colors", cmap=COLOR_PALETTE, show_edges=True)
plotter.add_mesh(
    dof_point_cloud, 
    scalars="Root_Cell_Color", 
    cmap=COLOR_PALETTE,
    point_size=20,
    render_points_as_spheres=True
)
```
</div>

### 3. Ghost Penalty Facet Analysis

Visualize stabilization structure:

```python
# Get ghost penalty facets
from cutfemx.level_set import ghost_penalty_facets, facet_topology

gp_ids = ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh, gp_ids)

# Create facet visualization
facets, ftypes, fx = plot.vtk_mesh(msh, tdim - 1, gp_ids)
fgrid = pyvista.UnstructuredGrid(facets, ftypes, fx)

# Highlight different facet types
plotter.add_mesh(fgrid, show_edges=True, line_width=5, color="red", 
                 label="Ghost Penalty Facets")
```

<div class="warning-box">
<h4>Ghost Penalty Visualization Considerations</h4>

When visualizing ghost penalty facets:
- **Line width**: Use thick lines (5-8 pixels) for visibility
- **Color contrast**: Red on gray background provides good visibility  
- **Opacity**: Keep background mesh semi-transparent (0.7-0.8)
- **Viewing angle**: Use `view_xy()` for 2D problems to avoid perspective distortion
</div>

## Colormap Selection and Analysis

### Optimal Colormaps for Many Colors

For visualizations requiring many distinct colors (DOF mapping, cell classification):

<div class="algorithm-box">
<h4>Colormap Performance Analysis</h4>

```python
def analyze_colormap_distinctness(colormap_name, num_colors):
    """Analyze how many distinct colors a colormap provides."""
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap(colormap_name)
    colors_sampled = []
    
    for i in range(1, num_colors + 1):
        normalized_value = i / num_colors
        color = cmap(normalized_value)
        colors_sampled.append(color)
    
    # Check for distinctness with tolerance
    unique_colors = []
    tolerance = 0.1
    
    for color in colors_sampled:
        is_unique = True
        for existing_color in unique_colors:
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(color[:3], existing_color[:3])))
            if distance < tolerance:
                is_unique = False
                break
        if is_unique:
            unique_colors.append(color)
    
    return len(unique_colors), num_colors

# Recommended colormaps for 35+ distinct colors:
EXCELLENT_COLORMAPS = [
    "hsv",          # Hue-Saturation-Value cycle (35/35 distinct)
    "jet",          # Classic jet colormap (34/35 distinct)  
    "gist_ncar",    # NCAR graphics (33/35 distinct)
]

GOOD_COLORMAPS = [
    "nipy_spectral",  # Spectral colors (26/35 distinct)
    "tab20",          # Qualitative colors (20 distinct)
]
```
</div>

### Colormap Usage Guidelines

<div class="success-box">
<h4>Best Practices for Color Selection</h4>

**For Many Distinct Elements (35+):**
- **Primary choice**: `"hsv"` - perfect coverage
- **Alternative**: `"jet"` or `"gist_ncar"`

**For Scalar Fields:**
- **Diverging data**: `"RdBu"`, `"coolwarm"`
- **Sequential data**: `"viridis"`, `"plasma"`
- **Positive data**: `"YlOrRd"`, `"Blues"`

**For Qualitative Data:**
- **Few categories**: `"Set1"`, `"Dark2"`
- **Many categories**: `"tab20"`, `"tab20b"`
</div>

## Interactive Features and Camera Control

### Camera Positioning

```python
# Standard viewing angles
plotter.view_xy()        # Top-down 2D view
plotter.view_isometric() # 3D isometric view
plotter.camera_position = "iso"  # Alternative isometric

# Custom camera positioning
plotter.camera.parallel_projection = True  # Remove perspective distortion
plotter.camera.focal_point = (0.0, 0.0, 0.0)  # Center of interest
plotter.camera.position = (0.0, 0.0, 2.0)     # Camera location
```

### Interactive Widgets and Controls

```python
# Add coordinate axes for orientation
plotter.add_axes()

# Add mesh legends
plotter.add_legend()

# Save visualization output
plotter.save_graphic("output.svg")  # Vector format
plotter.screenshot("output.png")    # Raster format
```

## Specialized Visualization Examples

### Convergence Study Visualization

For analysis workflows involving multiple mesh refinements:

<div class="example-box">
<h4>Convergence Analysis Display</h4>

```python
def visualize_convergence_study(refinement_levels, solutions, errors):
    """Visualize multiple refinement levels and convergence."""
    
    n_levels = len(refinement_levels)
    plotter = pyvista.Plotter(shape=(2, n_levels))
    
    for i, (level, sol, err) in enumerate(zip(refinement_levels, solutions, errors)):
        # Top row: Solution at each refinement level
        plotter.subplot(0, i)
        # Add solution visualization for level i
        plotter.add_title(f"Level {level}: h = {1.0/(2**level):.3f}")
        
        # Bottom row: Error distribution
        plotter.subplot(1, i)  
        # Add error visualization for level i
        plotter.add_title(f"Error: {err:.2e}")
    
    plotter.show()
```
</div>

### Time-Dependent Visualization

For moving boundary problems:

```python
def animate_moving_boundary(time_steps, level_sets, solutions):
    """Create animation of moving boundary problem."""
    
    plotter = pyvista.Plotter()
    
    # Initialize with first time step
    current_mesh = create_visualization_mesh(level_sets[0], solutions[0])
    mesh_actor = plotter.add_mesh(current_mesh, scalars="u")
    
    def update_frame(frame_number):
        # Update mesh and solution for current time step
        new_mesh = create_visualization_mesh(level_sets[frame_number], solutions[frame_number])
        mesh_actor.mapper.SetInputData(new_mesh)
        plotter.add_title(f"Time: {time_steps[frame_number]:.3f}")
    
    # Setup animation
    plotter.open_movie("animation.mp4")
    for i in range(len(time_steps)):
        update_frame(i)
        plotter.write_frame()
    plotter.close()
```

## Debugging and Diagnostic Visualizations

### Quadrature Integration Verification

<div class="warning-box">
<h4>Quadrature Diagnostics</h4>

```python
def diagnose_quadrature_coverage(level_set, quadrature_rules):
    """Verify quadrature points properly cover cut elements."""
    
    plotter = pyvista.Plotter(shape=(1, 2))
    
    # Left: Cut mesh with quadrature points
    plotter.subplot(0, 0)
    plotter.add_mesh(cgrid, show_edges=True, color="lightgray", opacity=0.8)
    
    # Different colors for different quadrature types
    inside_points = physical_points(quadrature_rules["inside"], msh)
    interface_points = physical_points(quadrature_rules["interface"], msh)
    
    plotter.add_points(inside_points, color="blue", point_size=8, label="Volume")
    plotter.add_points(interface_points, color="red", point_size=10, label="Interface")
    plotter.add_title("Quadrature Point Coverage")
    
    # Right: Integration domain verification
    plotter.subplot(0, 1)
    # Add level set contours to verify domain boundaries
    grid.point_data["phi"] = level_set.x.array
    contours = grid.contour([-0.1, 0.0, 0.1], scalars="phi")
    plotter.add_mesh(contours, line_width=2)
    plotter.add_title("Level Set Contours")
    
    plotter.show()
```
</div>

### Element Classification Visualization

```python
def visualize_element_classification(level_set, msh):
    """Display different element types in cut mesh."""
    
    # Classify elements
    intersected = locate_entities(level_set, tdim, "phi=0")
    inside = locate_entities(level_set, tdim, "phi<0") 
    outside = locate_entities(level_set, tdim, "phi>0")
    
    # Create classification array
    classification = np.zeros(msh.topology.index_map(tdim).size_local)
    classification[inside] = 1      # Inside elements
    classification[intersected] = 2 # Cut elements  
    classification[outside] = 3     # Outside elements
    
    # Visualize classification
    cells, types, x = plot.vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.cell_data["Classification"] = classification
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, scalars="Classification", show_edges=True, 
                     cmap="Set1", show_scalar_bar=True)
    plotter.add_title("Element Classification")
    plotter.show()
```

## Performance and Memory Considerations

### Large Mesh Visualization

<div class="note-box">
<h4>Memory Management for Large Visualizations</h4>

```python
def visualize_large_mesh(mesh, solution, max_cells=50000):
    """Handle visualization of large meshes efficiently."""
    
    if mesh.num_cells > max_cells:
        # Subsample for visualization
        stride = mesh.num_cells // max_cells
        cell_indices = np.arange(0, mesh.num_cells, stride)
        
        # Create subsampled mesh
        submesh = mesh.copy()
        submesh = submesh.extract_cells(cell_indices)
        
        print(f"Visualization subsampled: {mesh.num_cells} → {submesh.num_cells} cells")
        
        # Use subsampled mesh for visualization
        return visualize_mesh(submesh, solution[::stride])
    else:
        return visualize_mesh(mesh, solution)
```
</div>

### Batch Visualization Processing

For automated analysis workflows:

```python
def batch_visualize_results(results_directory, output_format="png"):
    """Process multiple results files automatically."""
    
    import os
    import glob
    
    result_files = glob.glob(os.path.join(results_directory, "*.xdmf"))
    
    for i, result_file in enumerate(result_files):
        # Load solution from file
        solution = load_solution(result_file)
        
        # Create visualization
        plotter = pyvista.Plotter(off_screen=True)  # No GUI for batch processing
        # Add visualization elements...
        
        # Save output
        output_name = f"result_{i:03d}.{output_format}"
        plotter.screenshot(output_name)
        plotter.close()
        
        print(f"Processed: {result_file} → {output_name}")
```

## Summary and Best Practices

<div class="success-box">
<h4>Visualization Best Practices</h4>

**Setup and Dependencies:**
✅ Always check for PyVista availability with try/except blocks  
✅ Use consistent import patterns across all visualization scripts  
✅ Choose appropriate colormap for your data type and number of distinct values  

**Layout and Design:**
✅ Use multi-panel layouts for comprehensive analysis  
✅ Include informative titles and legends  
✅ Maintain consistent camera angles across related plots  
✅ Add coordinate axes for spatial orientation  

**Performance:**
✅ Subsample large meshes for interactive visualization  
✅ Use off-screen rendering for batch processing  
✅ Save visualizations in appropriate formats (SVG for papers, PNG for reports)  

**Debugging:**
✅ Visualize quadrature points to verify integration domains  
✅ Display level set contours to confirm geometry definition  
✅ Use element classification plots to understand mesh structure  
</div>

<div class="algorithm-box">
<h4>Complete Visualization Workflow Template</h4>

```python
def complete_cutfemx_visualization(mesh, level_set, solution, output_dir="./plots"):
    """Comprehensive visualization template for CutFEMx results."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate cut mesh and quadrature
    intersected = locate_entities(level_set, tdim, "phi=0")
    inside = locate_entities(level_set, tdim, "phi<0")
    
    dof_coordinates = V.tabulate_dof_coordinates()
    cut_cells = cut_entities(level_set, dof_coordinates, intersected, tdim, "phi<0")
    cut_mesh = create_cut_mesh(mesh.comm, cut_cells, mesh, inside)
    
    inside_quad = runtime_quadrature(level_set, "phi<0", order=2)
    interface_quad = runtime_quadrature(level_set, "phi=0", order=2)
    
    # 2. Create three-panel standard visualization
    plotter = pyvista.Plotter(shape=(1, 3))
    
    # Panel 1: Cut mesh with quadrature
    plotter.subplot(0, 0)
    add_cut_mesh_with_quadrature(plotter, cut_mesh, inside_quad, mesh)
    
    # Panel 2: Ghost penalty and mesh overlay  
    plotter.subplot(0, 1)
    add_mesh_overlay_with_stabilization(plotter, mesh, cut_mesh, level_set)
    
    # Panel 3: Solution visualization
    plotter.subplot(0, 2)
    add_solution_visualization(plotter, mesh, solution)
    
    # 3. Save and display
    plotter.save_graphic(os.path.join(output_dir, "complete_analysis.svg"))
    plotter.screenshot(os.path.join(output_dir, "complete_analysis.png"))
    plotter.show()
    
    return plotter
```
</div>

This comprehensive visualization guide provides the foundation for creating publication-quality plots and interactive analysis tools with CutFEMx. The techniques demonstrated here support both research workflows and practical engineering applications, enabling clear communication of cut finite element results.
