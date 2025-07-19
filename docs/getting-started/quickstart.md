# Quickstart Tutorial

This tutorial demonstrates the basic workflow of CutFEMx by solving a cut Poisson problem with Nitsche boundary conditions on a circular domain. We'll walk through the complete `demo_cut_poisson.py` example step by step.

### Mathematical Foundation

The complete weak formulation implemented in this tutorial:

<div class="math-box">
<h4>CutFEM Weak Formulation</h4>

Find $u \in V_h^{\text{cut}}$ such that:

$$a_h(u,v) + S_h(u,v) = L_h(v) \quad \forall v \in V_h^{\text{cut}}$$

with 

$$a(u,v) = \int_\Omega \nabla u \cdot \nabla v \, dx - \int_{\partial\Omega} (\nabla u \cdot \mathbf{n}_K) v \, ds - \int_{\partial\Omega} (\nabla v \cdot \mathbf{n}_K) u \, ds + \frac{\gamma}{h} \int_{\partial\Omega} uv \, ds$$

and

$$L(v) = \int_\Omega fv \, dx - \int_{\partial\Omega} (\nabla v \cdot \mathbf{n}_K) g \, ds + \frac{\gamma}{h} \int_{\partial\Omega} gv \, ds$$

where the boundary terms ensure that the Dirichlet condition $u = g$ is enforced weakly. $S_h(u,v)$ denotes the stabilization term

$$S_h(u,v)=\gamma_g \sum_{F \in \mathcal{F}_{gp}} h_F \int_F [\![ \nabla u ]\!]_{\mathbf{n}_F} \cdot [\![ \nabla v ]\!]_{\mathbf{n}_F} \, dS$$


This formulation combines:
- **Volume integral**: finite element discretization on domain $\Omega$
- **Nitsche terms**: Boundary condition enforcement on $\partial \Omega$
- **Ghost penalty**: Stabilization for cut elements
</div>

## Problem Description

We solve the Poisson equation with homogeneous Dirichlet boundary conditions on a circular domain (radius 0.5) embedded in a rectangular background mesh. The problem demonstrates CutFEMx's core capabilities:

- **Implicit geometry definition** using level set functions
- **Automatic mesh cutting** for intersected elements  
- **Runtime quadrature rule generation** for accurate integration
- **Nitsche method** for weak boundary condition enforcement
- **Ghost penalty stabilization** for numerical stability

## Complete Example

Here's the complete demo with detailed explanations:

```python
import numpy as np
from mpi4py import MPI

# CutFEMx imports for level set operations
from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.level_set import compute_normal
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import assemble_vector, cut_form, assemble_matrix, deactivate, cut_function

# DOLFINx imports
from dolfinx import fem, mesh, plot, default_real_type
from dolfinx.fem import form
import ufl
from ufl import dS, dx, grad, inner, dc, FacetNormal, CellDiameter, dot, avg, jump

# Step 1: Create background mesh
N = 21
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle,
)
gdim = 2
tdim = msh.topology.dim

# Define function space
V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Problem parameters
f = fem.Constant(msh, default_real_type(1.0))  # source term
g = fem.Constant(msh, default_real_type(0.0))  # boundary value
gamma = 100     # Nitsche penalty parameter
gamma_g = 1e-1  # Ghost penalty parameter

n = FacetNormal(msh)
h = CellDiameter(msh)

# Step 2: Define level set function (circle with radius 0.5)
def circle(x):
    return np.sqrt((x[0])**2 + (x[1])**2) - 0.5

level_set = fem.Function(V)
level_set.interpolate(circle)

# Step 3: Locate cells relative to the level set
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

# Step 4: Compute interface normal vectors
V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K, level_set, intersected_entities)

# Step 5: Generate runtime quadrature rules
order = 2
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)

# Step 6: Combine quadrature domains
quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]

# Step 7: Get ghost penalty facets and their connectivities
gp_ids = ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh, gp_ids)

# Step 8: Define integration measures
dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)  # Volume integration
dsq = dx_rt(1)          # Interface integration

# Step 9: Define variational forms
# Bilinear form with Nitsche method
a = inner(grad(u), grad(v)) * dxq

# Nitsche terms for boundary conditions
a += -dot(grad(u), n_K) * v * dsq       # consistency term
a += -dot(grad(v), n_K) * u * dsq       # symmetry term  
a += gamma * 1.0 / h * u * v * dsq      # penalty term

# Ghost penalty for stability
a += avg(gamma_g) * avg(h) * dot(jump(grad(u), n), jump(grad(v), n)) * dS(0)

# Linear form
L = f * v * dxq
L += -dot(grad(v), n_K) * g * dsq       # consistency term
L += gamma * 1.0 / h * v * g * dsq      # penalty term

# Step 10: Create cut forms and assemble
a_cut = cut_form(a)
L_cut = cut_form(L)

b = assemble_vector(L_cut)
A = assemble_matrix(a_cut)

# Step 11: Deactivate DOFs outside the domain
deactivate(A, "phi>0", level_set, [V])

# Step 12: Solve linear system
A_py = A.to_scipy()
from scipy.sparse.linalg import spsolve
u_sol = spsolve(A_py, b.array)

# Step 13: Prepare data for visualization of cut mesh and quadrature points 
dof_coordinates = V.tabulate_dof_coordinates()
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)

points_phys = physical_points(inside_quadrature, msh)

try:
    import pyvista
    plotter = pyvista.Plotter(shape=(1, 3))

    # Prepare mesh data for visualization
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    
    facets, ftypes, fx = plot.vtk_mesh(msh, tdim - 1, gp_ids)
    fgrid = pyvista.UnstructuredGrid(facets, ftypes, fx)

    ccells, ctypes, cx = plot.vtk_mesh(cut_mesh._mesh)
    cgrid = pyvista.UnstructuredGrid(ccells, ctypes, cx)

    grid.point_data["u"] = u_sol
    grid.set_active_scalars("u")

    # Create visualizations
    plotter.subplot(0, 0)
    plotter.add_mesh(cgrid, show_edges=True, color="lightgray")
    plotter.add_points(points_phys, render_points_as_spheres=True, color="black", point_size=8)
    plotter.add_title("Cut Mesh with Quadrature Points")

    plotter.subplot(0, 1)
    plotter.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.7)
    plotter.add_mesh(cgrid, show_edges=True, color="blue", opacity=0.9)
    plotter.add_mesh(fgrid, show_edges=True, line_width=5, color="red")
    plotter.add_title("Ghost Penalty Facets and Cut Mesh")

    plotter.subplot(0, 2)
    warped = grid.warp_by_scalar(scale_factor=10.0)
    plotter.add_mesh(warped, scalars="u", show_edges=True, show_scalar_bar=True)
    plotter.add_title("Full Mesh Solution (Warped)")

    plotter.show()
except ModuleNotFoundError:
    print("PyVista not available for visualization")
    print(f"Solution computed with {len(u_sol)} DOFs")
    print(f"Solution range: [{np.min(u_sol):.3f}, {np.max(u_sol):.3f}]")
```

## Step-by-Step Explanation

### 1. Background Mesh and Function Space

```python
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 2))
```

We start with a regular triangular mesh over a rectangular domain that fully contains our geometry. This mesh serves as the background computational mesh. The actual problem domain will be a circle implicitly defined by a level set function.

### 2. Level Set Function and Geometry Definition

```python
def circle(x):
    return np.sqrt((x[0])**2 + (x[1])**2) - 0.5

level_set = fem.Function(V)
level_set.interpolate(circle)
```

The level set function $\phi(\mathbf{x})$ defines the geometry implicitly:
- **$\phi < 0$**: inside the circular domain (radius < 0.5)
- **$\phi = 0$**: on the circle boundary (radius = 0.5)  
- **$\phi > 0$**: outside the domain (radius > 0.5)

For our circular domain, the level set function is:
$\phi(\mathbf{x}) = \sqrt{x_0^2 + x_1^2} - 0.5$

This approach allows complex geometries to be defined without generating conforming meshes.

### 3. Cell Classification and Mesh Cutting

```python
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")
```

The `locate_entities` function returns the indices of entities that are 
- **Inside entities**: completely within the domain
- **Intersected entities**: crossed by the level set boundary

### 4. Interface Normal Computation

```python
V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
n_K = fem.Function(V_DG)
compute_normal(n_K, level_set, intersected_entities)
```

CutFEMx has different functions to obtain geometrical information from the level set function. The `compute_normal` function computes the normal $\mathbf{n}_K = \frac{\nabla \phi}{|\nabla \phi| + \epsilon}$, $\epsilon= 1e-12$ on all `intersected entities`. These normals are needed for implementing Nitsche's method to enforce Dirichlet boundary conditions on the circle boundary.

### 5. Runtime Quadrature Generation

```python
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)
```

CutFEMx generates specialized quadrature rules at runtime:
- **Inside quadrature**: for integration over cut volume elements inside the domain
- **Interface quadrature**: for integration over cut boundary surfaces

These rules ensure accurate numerical integration on the complex cut geometry.

### 6. Quadrature Domain Organization

```python
quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]
```

The quadrature domains are organized as a list of tuples that map subdomain IDs to their corresponding quadrature rules:
- **ID 0**: Inside quadrature for volume integrals over the cut domain
- **ID 1**: Interface quadrature for boundary integrals over the level set surface

This organization allows to properly route integration over different regions to the appropriate quadrature rules. They are used in the UFL formulation as described in Step 8.

### 7. Ghost Penalty Stabilization

```python
gp_ids = ghost_penalty_facets(level_set, "phi<0")
gp_topo = facet_topology(msh, gp_ids)
```
Ghost penalty terms stabilize the cut finite element method by penalizing jumps in gradients across faces between cut elements and between cut elements and uncut elements in the interior. This prevents spurious oscillations and ensures stability.

<div class="math-box">
<h4>Ghost Penalty Formulation</h4>

The ghost penalty term for the Poisson problem is:

$$\gamma_g \sum_{F \in \mathcal{F}_{gp}} h_F \int_F [\![ \nabla u ]\!]_{\mathbf{n}_F} \cdot [\![ \nabla v ]\!]_{\mathbf{n}_F} \, dS$$

where $\gamma_g$ is the ghost penalty parameter (to be chosen $<<1$), $\mathcal{F}_{gp}$ are the ghost penalty faces, $h_F$ is the face diameter, and $[\![ \cdot ]\!]_{\mathbf{n}_F}$ denotes the normal jump across the face.
</div>

### 8. Integration Measures Definition

```python
dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topo)], domain=msh)
dx_rt = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

dxq = dx_rt(0) + dx(0)  # Volume integration
dsq = dx_rt(1)          # Interface integration
```

CutFEMx uses specialized integration measures that combine standard UFL measures with runtime quadrature:
- **`dx`**: Standard volume integration over inside entities
- **`dS`**: Integration over ghost penalty facets  
- **`dx_rt`**: Runtime quadrature measure for cut elements
- **`dxq`**: Combined volume integration (standard + runtime)
- **`dsq`**: Interface integration using runtime quadrature

<div class="note-box">
<h4>Integration Measure Types</h4>

- **`"dx"`**: Volume integration over cells
- **`"dS"`**: Integration over interior facets
- **`"dC"`**: Custom CutFEMx runtime quadrature measure
- **Subdomain data**: Maps subdomain IDs to entity lists or quadrature rules
</div>

### 9. Nitsche Method Formulation

```python
# Bilinear form with Nitsche method
a = inner(grad(u), grad(v)) * dxq
a += -dot(grad(u), n_K) * v * dsq       # consistency term
a += -dot(grad(v), n_K) * u * dsq       # symmetry term  
a += gamma * 1.0 / h * u * v * dsq      # penalty term

# Ghost penalty for stability
a += avg(gamma_g) * avg(h) * dot(jump(grad(u), n), jump(grad(v), n)) * dS(0)

# Linear form
L = f * v * dxq
L += -dot(grad(v), n_K) * g * dsq       # consistency term
L += gamma * 1.0 / h * v * g * dsq      # penalty term
```

The Nitsche method enforces Dirichlet boundary conditions weakly through three terms:
- **Consistency**: $-\int_{\partial\Omega} (\nabla u \cdot \mathbf{n}_K) v \, ds$ ensures the PDE is satisfied in a weak sense
- **Symmetry**: $-\int_{\partial\Omega} (\nabla v \cdot \mathbf{n}_K) u \, ds$ maintains the symmetry of the bilinear form
- **Penalty**: $\frac{\gamma}{h} \int_{\partial\Omega} u v \, ds$ enforces the boundary condition $u = g$ with parameter $\gamma$ and ensures coercitivity of the bilinear form. 

The complete bilinear form becomes:

$$a(u,v) = \int_\Omega \nabla u \cdot \nabla v \, dx - \int_{\partial\Omega} (\nabla u \cdot \mathbf{n}_K) v \, ds - \int_{\partial\Omega} (\nabla v \cdot \mathbf{n}_K) u \, ds + \frac{\gamma}{h} \int_{\partial\Omega} uv \, ds$$

The corresponding right-hand side becomes:

$$L(v) = \int_\Omega fv \, dx - \int_{\partial\Omega} (\nabla v \cdot \mathbf{n}_K) g \, ds + \frac{\gamma}{h} \int_{\partial\Omega} gv \, ds$$

where the boundary terms ensure that the Dirichlet condition $u = g$ is enforced weakly.

<div class="warning-box">
<h4>Parameter Selection</h4>

The Nitsche penalty parameter $\gamma$ should typically be chosen between 10-100. Too small values may lead to unstable solutions (loss of coercivity), while too large values can affect accuracy and conditioning.
</div>

### 10. Cut Form Creation and Assembly

```python
a_cut = cut_form(a)
L_cut = cut_form(L)

b = assemble_vector(L_cut)
A = assemble_matrix(a_cut)
```

The `cut_form` function is a crucial CutFEMx component that transforms standard UFL forms into specialized cut finite element forms. This function:

- **Processes runtime quadrature**: Integrates the custom quadrature rules generated for cut elements into the form
- **Handles cut geometry**: Ensures integration is performed only over the actual computational domain  

The specialized `assemble_matrix` and `assemble_vector` functions from CutFEMx handle the complex assembly process on cut elements, including:
- Integration over cut element portions using runtime quadrature
- Proper handling of interface terms with computed normals

<div class="note-box">
<h4>CutFEMx Assembly Process</h4>

Unlike standard DOLFINx assembly, CutFEMx assembly must handle: 
- **Runtime quadrature rules**: Custom quadrature points and weights for each cut element
- **Interface integration**: Boundary integrals over implicitly defined surfaces (manifolds)
</div>

### 11. DOF Deactivation and Solving

```python
deactivate(A, "phi>0", level_set, [V])
```

CutFEMx deactivates degrees of freedom outside the computational domain by setting all degrees of freedom outside of the domain to zero. 

<div class="warning-box">
<h4>Why DOF Deactivation is Critical</h4>

Without deactivation, DOFs in the exterior region cause
- **Zero rows** in the system matrix
- **Singular system**: Leading to `NaN` values in the solution vector
</div>

### 12. Linear System Solution

```python
A_py = A.to_scipy()
from scipy.sparse.linalg import spsolve
u_sol = spsolve(A_py, b.array)
```

After assembly and DOF deactivation, we solve the linear system $Au = b$:

- **Matrix conversion**: `A.to_scipy()` converts the dolfinx matrix to SciPy sparse format
- **Direct solver**: `spsolve()` uses a direct sparse solver (typically SuperLU)
- **Solution vector**: `u_sol` contains the coefficients for the finite element solution

<div class="note-box">
<h4>Solver Characteristics for CutFEM</h4>

Cut finite element systems have unique properties:
- **Condition number**: Can be deteriorated by small cut elements and therefore need
- **Stabilitzation**: Ghost penalty improves conditioning
- **Sparsity pattern**: Modified by cut cell integration, embedded interface and ghost penalty terms

</div>

<div class="algorithm-box">
<h4>Solution Process Breakdown</h4>

1. **Matrix conversion**: dolfinx → SciPy sparse (CSR format)
2. **Factorization**: Direct LU decomposition for small-medium problems

For larger problems, consider:
- **Iterative methods**: CG for symmetric positive definite systems or GMRES for non-symmetric systems
- **Preconditioners**: Algebraic multigrid (AMG) or incomplete LU
- **Parallel solvers**: PETSc KSP with MPI for distributed memory
</div>

### 13. Visualization and Post-Processing

```python
# Prepare data for visualization of cut mesh and quadrature points 
dof_coordinates = V.tabulate_dof_coordinates()
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")
cut_mesh = create_cut_mesh(msh.comm, cut_cells, msh, inside_entities)

points_phys = physical_points(inside_quadrature, msh)
```

The visualization setup involves several CutFEMx components:

- **Cut mesh creation**: `create_cut_mesh()` generates a mesh containing `inside_entities` and the cut elements
- **Quadrature visualization**: `physical_points()` returns physical coordinates of quadrature points
- **DOF coordinates**: Used for cutting of cells by level set function 

<div class="note-box">
<h4>Cut Mesh Construction Details</h4>

The cut mesh creation process:
1. **Entity extraction**: `cut_entities()` computes the intersection between the level set and mesh entitities and returns local cut meshes for each entitiy
2. **Mesh generation**: `create_cut_mesh()` creates a new mesh from selected entities and the local cut meshes
</div>

<div class="math-box">
<h4>Quadrature Point Mapping</h4>

Physical quadrature points are computed via:

$\mathbf{x}_q = \sum_{i=1}^{n} N_i(\boldsymbol{\xi}_q) \mathbf{x}_i$

where:
- $\mathbf{x}_q$: Physical coordinates of quadrature point
- $N_i(\boldsymbol{\xi}_q)$: Lagrange shape function evaluated at reference point $\boldsymbol{\xi}_q$
- $\mathbf{x}_i$: Physical coordinates of mesh vertices
- $n$: Number of vertices per element
</div>

<div class="example-box">
<h4>PyVista Visualization Pipeline</h4>

```python
try:
    import pyvista
    plotter = pyvista.Plotter(shape=(1, 3))

    # Prepare mesh data for visualization
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    
    facets, ftypes, fx = plot.vtk_mesh(msh, tdim - 1, gp_ids)
    fgrid = pyvista.UnstructuredGrid(facets, ftypes, fx)

    ccells, ctypes, cx = plot.vtk_mesh(cut_mesh._mesh)
    cgrid = pyvista.UnstructuredGrid(ccells, ctypes, cx)

    grid.point_data["u"] = u_sol
    grid.set_active_scalars("u")

    # Create visualizations
    plotter.subplot(0, 0)
    plotter.add_mesh(cgrid, show_edges=True, color="lightgray")
    plotter.add_points(points_phys, render_points_as_spheres=True, color="black", point_size=8)
    plotter.add_title("Cut Mesh with Quadrature Points")

    plotter.subplot(0, 1)
    plotter.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.7)
    plotter.add_mesh(cgrid, show_edges=True, color="blue", opacity=0.9)
    plotter.add_mesh(fgrid, show_edges=True, line_width=5, color="red")
    plotter.add_title("Ghost Penalty Facets and Cut Mesh")

    plotter.subplot(0, 2)
    warped = grid.warp_by_scalar(scale_factor=10.0)
    plotter.add_mesh(warped, scalars="u", show_edges=True, show_scalar_bar=True)
    plotter.add_title("Full Mesh Solution (Warped)")

    plotter.show()
except ModuleNotFoundError:
    print("PyVista not available for visualization")
    print(f"Solution computed with {len(u_sol)} DOFs")
    print(f"Solution range: [{np.min(u_sol):.3f}, {np.max(u_sol):.3f}]")
```
</div>

The visualization creates three informative views:

1. **Cut Mesh with Quadrature Points**: Shows the actual computational domain with runtime integration points
2. **Ghost Penalty Facets**: Visualizes stabilization faces (in red) overlaid mesh and cut mesh 
3. **Solution Visualization**: Displays the computed solution with 3D warping 

<div class="algorithm-box">
<h4>PyVista Visualization Pipeline</h4>

The visualization process involves several steps:

1. **VTK mesh conversion**: `plot.vtk_mesh()` converts DOLFINx mesh to VTK format
2. **Grid creation**: `pyvista.UnstructuredGrid()` creates visualization-ready grids
3. **Data attachment**: Solution values are attached as point data to the grid
4. **Multi-plot layout**: Three subplots show different aspects of the solution

**Key visualization components:**
- **Cut mesh grid**: Shows only the physical computational domain
- **Ghost penalty facets**: Highlights stabilization faces
- **Solution warping**: 3D deformation proportional to solution magnitude
</div>

<div class="example-box">
<h4>Visualization Output Interpretation</h4>

**Subplot 1 - Cut Mesh with Quadrature Points:**
- Gray mesh: Cut elements forming the computational domain
- Black spheres: Integration points where the weak form is evaluated in the cut cells
- **What to check**: Quadrature points should be inside the cut cell region

**Subplot 2 - Ghost Penalty Facets and Cut Mesh:**
- Blue mesh: Cut elements (computational domain)
- Red lines: Faces where ghost penalty stabilization is applied
- **What to check**: Ghost penalty faces should connect neighboring cut elements and cut elements to elements fully inside the domain.

**Subplot 3 - Solution Visualization:**
- Color mapping: Solution values mapped to colors via scalar bar
- Warping: 3D displacement proportional to solution magnitude
- **What to check**: Solution should be smooth (especially in the intersected elements) and satisfy boundary conditions
</div>

<div class="success-box">
<h4>Visualization Benefits</h4>

- **Verification**: Ensures the cut mesh represents the intended geometry
- **Solution validation**: Visual inspection can detect numerical issues
- **Communication**: Clear plots help explain cut finite element concepts
- **Debugging**: Visualization helps identify problems in mesh generation or assembly
</div>


## Summary and Next Steps

This tutorial demonstrated the complete workflow for solving a Poisson equation on an implicitly defined domain using cut finite elements.

<div class="success-box">
<h4>Key steps</h4>

✅ **Implicit domain definition**: Used a level set function to define a circular domain  
✅ **Cut finite element setup**: Created function spaces and quadrature rules for cut elements  
✅ **Stabilization**: Applied ghost penalty stabilization for numerical stability  
✅ **Nitsche's method**: Enforced boundary conditions without mesh conformity  
✅ **Linear system solution**: Solved the discrete system with proper DOF deactivation  
✅ **Visualization**: Created informative plots showing the cut mesh and solution  
</div>

### Key CutFEMx Concepts Covered

1. **Level Set Functions**: Implicit geometry representation using signed distance functions
2. **Cut Element Integration**: Custom quadrature rules for intersected elements  
3. **Ghost Penalty Stabilization**: Ensuring stability across cut element boundaries
4. **Nitsche's Method**: Weak enforcement of boundary conditions on implicit boundaries
5. **DOF Deactivation**: Removing exterior degrees of freedom for well-posed systems
   

### Next Steps

<div class="example-box">
<h4>Explore More Examples</h4>

1. **Different geometries**: Try ellipses, rectangles, or more complex level set functions
2. **3D problems**: Extend to three-dimensional domains using spheres or other 3D shapes
3. **Elasticity**: Solve vector-valued problems with CutFEMx
</div>

### Troubleshooting Common Issues

<div class="warning-box">
<h4>Common Problems and Solutions</h4>

**Problem**: Singular matrix or `NaN` values in solution
- **Solution**: Check DOF deactivation is applied correctly
- **Check**: Verify the level set function defines the domain properly

**Problem**: Poor convergence or oscillatory solutions  
- **Solution**: Increase ghost penalty parameter `gamma_gp`
- **Check**: Ensure sufficient quadrature points for cut elements

**Problem**: Slow assembly or memory issues
- **Solution**: Use iterative solvers instead of direct methods
- **Check**: Monitor memory usage during assembly phase

**Problem**: Visualization artifacts or strange mesh appearance
- **Solution**: Verify cut mesh construction and entity selection
- **Check**: Ensure level set function is smooth and well-defined
</div>

### Further Reading

<!-- - **CutFEMx Documentation**: Explore the full API reference and advanced examples -->
- **CutFEM Theory**: Burman, E., et al. "CutFEM: Discretizing geometry and partial differential equations" (2015)
- **Nitsche's Method**: Nitsche, J. "Über ein Variationsprinzip zur Lösung von Dirichlet-Problemen" (1971)
- **Ghost Penalty**: Burman, E. "Ghost penalty" (2010)

