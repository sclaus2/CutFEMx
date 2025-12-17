# Stabilization Techniques

CutFEMx provides stabilization methods to ensure well-posed problems when dealing with cut finite element methods. These techniques address challenges like badly cut cells and interface conditions.

## Ghost Penalty Method

The ghost penalty method adds stability by penalizing jumps across element faces that are connected to cut cells:

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl

from cutfemx.level_set import locate_entities, ghost_penalty_facets, facet_topology
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form

# Create mesh and function space
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle
)
V = fem.functionspace(msh, ("Lagrange", 1))

# Define level set function
def circle_level_set(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.5

level_set = fem.Function(V)
level_set.interpolate(circle_level_set)

# Locate entities
tdim = msh.topology.dim
inside_entities = locate_entities(level_set, tdim, "phi<0")

# Get ghost penalty facets
gp_facets = ghost_penalty_facets(level_set, "phi<0")
gp_topology = facet_topology(msh, gp_facets)

print(f"Found {len(gp_facets)} ghost penalty facets")

# Create ghost penalty stabilization
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
gamma_gp = 0.1  # Ghost penalty parameter
h = ufl.CellDiameter(msh)

# Ghost penalty term (stabilizes across interior facets)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topology)], domain=msh)
ghost_penalty_term = gamma_gp * ufl.avg(h) * ufl.dot(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS(0)

print("Ghost penalty stabilization created")
```

## Extension Operator Stabilization

For badly cut cells, use extension operators to provide stability:

```python
from cutfemx.level_set import cut_entities
from cutfemx.extensions import get_badly_cut_parent_indices, build_root_mapping

# Get DOF coordinates and cut cells
dof_coordinates = V.tabulate_dof_coordinates()
intersected_entities = locate_entities(level_set, tdim, "phi=0")
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, tdim, "phi<0")

# Identify badly cut cells
h_estimate = 2.0 / 32  # Mesh size estimate
threshold_factor = 0.1
badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold_factor)

print(f"Found {len(badly_cut_parents)} badly cut cells")

# Build root mapping for stabilization
if len(badly_cut_parents) > 0:
    root_mapping = build_root_mapping(badly_cut_parents, inside_entities, msh, cut_cells)
    print("Root mapping built for extension operators")
```

## Complete Stabilized Problem

Here's a complete example of a stabilized Poisson problem:

```python
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, la, io
import ufl

from cutfemx.level_set import locate_entities, ghost_penalty_facets, facet_topology
from cutfemx.quadrature import runtime_quadrature
from cutfemx.fem import cut_form, assemble_matrix, assemble_vector

# Setup mesh and function space
N = 32
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(N, N),
    cell_type=mesh.CellType.triangle
)
V = fem.functionspace(msh, ("Lagrange", 1))

# Level set function (circle)
def circle(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.5

level_set = fem.Function(V)
level_set.interpolate(circle)

# Locate entities
tdim = msh.topology.dim
inside_entities = locate_entities(level_set, tdim, "phi<0")

# Create quadrature for inside domain
inside_quadrature = runtime_quadrature(level_set, "phi<0", 2)
quad_domains = [(0, inside_quadrature)]

# Ghost penalty setup
gp_facets = ghost_penalty_facets(level_set, "phi<0")
gp_topology = facet_topology(msh, gp_facets)

# Define variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(msh, 1.0)

# Measures
dx_cut = ufl.Measure("dC", subdomain_data=quad_domains, domain=msh)
dS = ufl.Measure("dS", subdomain_data=[(0, gp_topology)], domain=msh)

# Standard bilinear form
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * dx_cut(0)
L = f * v * dx_cut(0)

# Ghost penalty stabilization
gamma_gp = 0.01  # Stabilization parameter
h = ufl.CellDiameter(msh)
a_gp = gamma_gp * ufl.avg(h) * ufl.dot(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS(0)

# Complete stabilized form
a_stabilized = a + a_gp

# Create cut forms and assemble
cut_a = cut_form(a_stabilized)
cut_L = cut_form(L)

# Update with runtime quadrature
cut_a.update_runtime_domains({"dC": quad_domains})
cut_L.update_runtime_domains({"dC": quad_domains})

# Assemble system
A = assemble_matrix(cut_a)
b = assemble_vector(cut_L)

# Solve
solver = la.LinearSolver(A, la.SolverType.default)
uh = fem.Function(V)
solver.solve(uh.x, b)

print("Stabilized problem solved successfully!")

# Export solution
with io.XDMFFile(msh.comm, "stabilized_solution.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
```

## Nitsche Method for Boundary Conditions

Nitsche's method provides stability for boundary condition enforcement:

```python
# Nitsche stabilization parameters
beta_nitsche = 10.0  # Penalty parameter
p = 1  # Polynomial degree

# Scaling with element size for stability
h_scaling = ufl.CellDiameter(msh)
penalty_term = beta_nitsche * p**2 / h_scaling

# Example Nitsche BC implementation (conceptual)
# In practice, this requires careful interface integration
# See boundary-conditions.md for complete implementation
```

## Parameter Selection Guidelines

### Ghost Penalty Parameters

```python
def compute_ghost_penalty_parameter(mesh_size, polynomial_degree=1):
    """
    Compute ghost penalty parameter.
    
    Parameters:
    - mesh_size: Characteristic mesh size
    - polynomial_degree: Finite element polynomial degree
    
    Returns:
    - gamma: Ghost penalty parameter
    """
    # Rule of thumb: γ ~ h^(2p-1) for degree p elements
    gamma = 0.01 * mesh_size**(2*polynomial_degree - 1)
    return max(gamma, 1e-6)  # Minimum bound

# Usage
h = 2.0 / N
gamma_gp = compute_ghost_penalty_parameter(h, polynomial_degree=1)
print(f"Recommended ghost penalty parameter: {gamma_gp}")
```

### Extension Operator Thresholds

```python
def compute_threshold_factor(problem_type="poisson"):
    """
    Compute threshold for badly cut cell identification.
    
    Parameters:
    - problem_type: Type of PDE problem
    
    Returns:
    - threshold: Volume fraction threshold
    """
    thresholds = {
        "poisson": 0.1,
        "advection": 0.05,  # More restrictive for hyperbolic problems
        "stokes": 0.15,     # Less restrictive for Stokes
        "elasticity": 0.1
    }
    return thresholds.get(problem_type, 0.1)

# Usage
threshold = compute_threshold_factor("poisson")
badly_cut_parents = get_badly_cut_parent_indices(cut_cells, h_estimate, threshold)
```

## Advanced Stabilization Patterns

### Problem-Specific Stabilization

```python
def create_stabilized_bilinear_form(problem_type, u, v, msh, level_set, gamma_gp=0.01):
    """
    Create stabilized bilinear form based on problem type.
    """
    # Get ghost penalty terms
    gp_facets = ghost_penalty_facets(level_set, "phi<0")
    gp_topology = facet_topology(msh, gp_facets)
    dS = ufl.Measure("dS", subdomain_data=[(0, gp_topology)], domain=msh)
    
    h = ufl.CellDiameter(msh)
    
    if problem_type == "poisson":
        # Standard ghost penalty for second-order problems
        return gamma_gp * ufl.avg(h) * ufl.dot(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS(0)
    
    elif problem_type == "advection":
        # Additional streamline stabilization might be needed
        # This would require velocity field information
        return gamma_gp * ufl.avg(h) * ufl.jump(u) * ufl.jump(v) * dS(0)
    
    else:
        # Default to gradient jump penalty
        return gamma_gp * ufl.avg(h) * ufl.dot(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS(0)

# Usage
stabilization_term = create_stabilized_bilinear_form("poisson", u, v, msh, level_set)
```

## Monitoring Stabilization Effectiveness

### Conditioning Assessment

```python
def assess_stabilization_effectiveness(A_original, A_stabilized):
    """
    Compare condition numbers to assess stabilization impact.
    """
    # Note: This is conceptual - actual condition number computation 
    # for large matrices requires specialized techniques
    
    print("Stabilization assessment:")
    print(f"Matrix size: {A_stabilized.size}")
    print(f"Non-zeros increased by stabilization")
    
    # In practice, you might monitor:
    # - Solver convergence rates
    # - Residual norms
    # - Energy norms of the solution
    
    return True

# Usage after assembly
# assess_stabilization_effectiveness(A_original, A_stabilized)
```

### Convergence Monitoring

```python
def monitor_convergence_with_stabilization(mesh_sizes, gamma_values):
    """
    Study convergence behavior with different stabilization parameters.
    """
    results = []
    
    for h in mesh_sizes:
        for gamma in gamma_values:
            # Solve problem with current h and gamma
            # error = solve_and_compute_error(h, gamma)
            # results.append((h, gamma, error))
            pass
    
    # Analyze results to find optimal parameters
    return results

# Example usage
# mesh_sizes = [1/16, 1/32, 1/64, 1/128]
# gamma_values = [0.001, 0.01, 0.1, 1.0]
# results = monitor_convergence_with_stabilization(mesh_sizes, gamma_values)
```

## Best Practices

### When to Use Stabilization

1. **Always use ghost penalty** for cut cell problems
2. **Use extension operators** when >10% of cut cells are badly cut
3. **Monitor solution quality** by checking residuals and energy norms
4. **Start with conservative parameters** and reduce if needed

### Parameter Tuning Strategy

```python
def tune_stabilization_parameters(initial_gamma=0.01):
    """
    Systematic approach to parameter tuning.
    """
    # 1. Start with recommended values
    gamma = initial_gamma
    
    # 2. Solve and check convergence
    # converged = solve_with_gamma(gamma)
    
    # 3. If not converged, increase gamma
    # while not converged and gamma < 1.0:
    #     gamma *= 2
    #     converged = solve_with_gamma(gamma)
    
    # 4. If converged too quickly, try smaller gamma
    # if converged and gamma > 1e-4:
    #     gamma /= 2
    #     test_convergence = solve_with_gamma(gamma)
    
    return gamma

# Usage
# optimal_gamma = tune_stabilization_parameters()
```

### Implementation Checklist

- [ ] Identify cut cells and badly cut cells
- [ ] Compute ghost penalty facets
- [ ] Set appropriate stabilization parameters
- [ ] Add ghost penalty terms to bilinear form
- [ ] Consider extension operators for badly cut cells
- [ ] Monitor solver convergence
- [ ] Validate solution quality

## Troubleshooting

### Common Issues

1. **Poor convergence**: Increase ghost penalty parameter
2. **Over-stabilization**: Reduce penalty parameters
3. **Solver issues**: Check matrix conditioning
4. **Unphysical oscillations**: Add more stabilization

### Diagnostic Tools

```python
def diagnose_stabilization_issues(uh, level_set, cut_cells):
    """
    Diagnostic function for stabilization problems.
    """
    # Check for oscillations near the interface
    # Monitor residual distribution
    # Assess cut cell quality
    
    print("Stabilization diagnostics:")
    print(f"Solution range: [{uh.x.array.min():.3e}, {uh.x.array.max():.3e}]")
    print(f"Cut cells: {len(cut_cells.cut_cells)}")
    
    # Additional diagnostics would go here
    return True

# Usage after solving
# diagnose_stabilization_issues(uh, level_set, cut_cells)
```

## Next Steps

- **[Boundary Conditions](boundary-conditions.md)** - Nitsche method implementation
- **[DOF Constraints](dof-constraints.md)** - Extension operator techniques  
- **[Cut Mesh Generation](cut-meshes.md)** - Understanding cut element quality

---

*Stabilization is crucial for robust cut finite element implementations. The ghost penalty method and extension operators provide effective stabilization strategies for most applications.*
