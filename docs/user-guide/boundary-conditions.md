# Boundary Conditions on Cut Boundaries

CutFEMx provides methods for applying boundary conditions on implicitly defined boundaries using Nitsche's method and specialized integration measures for cut elements.

## Nitsche's Method for Dirichlet Conditions

Nitsche's method weakly enforces Dirichlet boundary conditions on cut boundaries without requiring boundary-fitted meshes:

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl

from cutfemx.level_set import locate_entities, cut_entities, compute_normal
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature

# Create background mesh and function space
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle
)
V = fem.functionspace(msh, ("Lagrange", 1))

# Define level set for circular boundary
def circle_level_set(x):
    return np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2) - 0.3

# Create level set function
level_set = fem.Function(V)
level_set.interpolate(circle_level_set)

# Locate cut entities
tdim = msh.topology.dim
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

# Compute interface normal
V_vec = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))
interface_normal = fem.Function(V_vec)
compute_normal(interface_normal, level_set, intersected_entities)

# Define Dirichlet boundary value
u_D = fem.Function(V)
u_D.interpolate(lambda x: x[0] + x[1])  # Prescribed value

# Create quadrature rules for interface integration
order = 2
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)
quad_domains = [(0, interface_quadrature)]

# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define measures
dx = ufl.Measure("dx", subdomain_data=[(0, inside_entities)], domain=msh)
ds_interface = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

# Nitsche parameters
beta = 10.0  # Penalty parameter
h = ufl.CellDiameter(msh)

# Nitsche formulation for Dirichlet BC
# Standard interior term
a_interior = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx(0)

# Interface terms for Nitsche's method
# Note: For cut boundaries, interface normal should be used appropriately
# This is a simplified formulation - see demo files for complete implementation
a_nitsche = (
    - ufl.avg(ufl.dot(ufl.grad(u), interface_normal)) * ufl.jump(v) * ds_interface(0)
    - ufl.avg(ufl.dot(ufl.grad(v), interface_normal)) * ufl.jump(u) * ds_interface(0)
    + beta / ufl.avg(h) * ufl.jump(u) * ufl.jump(v) * ds_interface(0)
)

# Right-hand side
f = fem.Constant(msh, 1.0)
L_interior = f * v * dx(0)
L_nitsche = (
    - ufl.avg(ufl.dot(ufl.grad(v), interface_normal)) * u_D * ds_interface(0)
    + beta / ufl.avg(h) * u_D * ufl.jump(v) * ds_interface(0)
)

# Complete variational form
a = a_interior + a_nitsche
L = L_interior + L_nitsche
```

## Interface Integration

For proper integration on cut boundaries, use the interface quadrature:

```python
from cutfemx.fem import cut_form

# Create interface cells for boundary integration
dof_coordinates = V.tabulate_dof_coordinates()
interface_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi=0")

# Create cut form with interface quadrature
interface_form = cut_form(ufl.inner(u, v) * ds_interface)
interface_form.update_runtime_domains({"dC": quad_domains})
```

## Working with Boundaries in Practice

A complete example from the CutFEMx demos:

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, la
import ufl

from cutfemx.level_set import locate_entities, cut_entities, ghost_penalty_facets, facet_topology
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form

# Create mesh
N = 32
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle
)

V = fem.functionspace(msh, ("Lagrange", 1))
tdim = msh.topology.dim

# Level set function
def circle(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.5

level_set = fem.Function(V)
level_set.interpolate(circle)

# Locate entities
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

# Create quadrature rules
order = 2
inside_quadrature = runtime_quadrature(level_set, "phi<0", order)
interface_quadrature = runtime_quadrature(level_set, "phi=0", order)

# Define measures
quad_domains = [(0, inside_quadrature), (1, interface_quadrature)]
dx_cut = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)

# Ghost penalty for stabilization
gp_facets = ghost_penalty_facets(level_set, "phi<0")
gp_topology = facet_topology(msh, gp_facets)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topology)], domain=msh)

# Define problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(msh, 1.0)

# Standard form with cut cell integration
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_cut(0)
L = f * v * dx_cut(0)

# Ghost penalty stabilization
gamma_gp = 0.1
h = ufl.CellDiameter(msh)
a_gp = gamma_gp * ufl.avg(h) * ufl.dot(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS(0)

# Complete bilinear form
a_total = a + a_gp

# Create cut forms
cut_a = cut_form(a_total)
cut_L = cut_form(L)

# Update with runtime quadrature
cut_a.update_runtime_domains({"dC": quad_domains})
cut_L.update_runtime_domains({"dC": quad_domains})
```

## Boundary Value Problems

For problems with prescribed boundary values, use the extension operators:

```python
from cutfemx.extensions import get_badly_cut_parent_indices, build_root_mapping

# Identify badly cut cells that need stabilization
h_estimate = 2.0 / N
badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold_factor=0.1)

# Build root mapping for extension
if len(badly_cut_parents) > 0:
    root_mapping = build_root_mapping(badly_cut_parents, inside_entities, msh, cut_cells)
    # Apply constraints using the extension operators
```

## Flux Boundary Conditions

For Neumann-type conditions, integrate the prescribed flux on the interface:

```python
# Prescribed flux
def flux_function(x):
    return np.sin(np.pi * x[0])

g_N = fem.Function(V)
g_N.interpolate(flux_function)

# Neumann term (natural boundary condition)
L_neumann = g_N * v * dx_cut(1)  # Interface quadrature domain

# Add to right-hand side
L_total = L + L_neumann
```

## Best Practices

### Penalty Parameter Selection
1. **Start with moderate values**: β ∈ [1, 100]
2. **Scale with element size**: β ~ h^(-1) for stability
3. **Problem-dependent tuning**: May need adjustment based on problem physics

### Stabilization
1. **Ghost penalty**: Use for cut cell stabilization
2. **Interface stabilization**: Add penalty terms on cut boundaries
3. **Extension operators**: Use for badly cut cells

### Integration Accuracy
1. **Sufficient quadrature order**: Use order ≥ 2p for degree p elements
2. **Interface resolution**: Ensure accurate boundary representation
3. **Monitor conditioning**: Check system conditioning with penalty terms

## Example: Complete Boundary Value Problem

```python
# Complete example: Poisson with Dirichlet BC on cut boundary
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, la, io
import ufl

from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form, assemble_matrix, assemble_vector

# Setup mesh and function space
msh = mesh.create_rectangle(MPI.COMM_WORLD, [(-1, -1), (1, 1)], [32, 32])
V = fem.functionspace(msh, ("Lagrange", 1))

# Level set (circle)
level_set = fem.Function(V)
level_set.interpolate(lambda x: np.sqrt(x[0]**2 + x[1]**2) - 0.5)

# Locate entities and create quadrature
tdim = msh.topology.dim
intersected_entities = locate_entities(level_set, tdim, "phi=0")
inside_entities = locate_entities(level_set, tdim, "phi<0")

inside_quadrature = runtime_quadrature(level_set, "phi<0", 2)
quad_domains = [(0, inside_quadrature)]

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(msh, 1.0)

dx_cut = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_cut(0)
L = f * v * dx_cut(0)

# Create and assemble cut forms
cut_a = cut_form(a)
cut_L = cut_form(L)
cut_a.update_runtime_domains({"dC": quad_domains})
cut_L.update_runtime_domains({"dC": quad_domains})

A = assemble_matrix(cut_a)
b = assemble_vector(cut_L)

# Solve system
solver = la.LinearSolver(A, la.SolverType.default)
uh = fem.Function(V)
solver.solve(uh.x, b)

# Export solution
with io.XDMFFile(msh.comm, "solution.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)

print("Boundary value problem solved successfully!")
```

## Next Steps

- **[Stabilization Techniques](stabilization.md)** - Learn about stabilization for cut elements
- **[DOF Constraints](dof-constraints.md)** - Understand constraint application
- **[Advanced Examples](../examples/)** - See complete boundary condition implementations

---

*Boundary conditions on cut boundaries require careful treatment of interface terms and stabilization. The Nitsche method provides a robust framework for weakly enforcing boundary conditions.*

