# Quadrature Rules

Accurate numerical integration is critical in cut finite element methods. CutFEMx provides runtime quadrature generation for integration on cut elements and boundaries where standard Gaussian quadrature is not suitable.

## Overview

CutFEMx provides quadrature capabilities through the `cutfemx.quadrature` module:

1. **Runtime Quadrature**: Generate quadrature rules for level set-defined regions
2. **Physical Points**: Convert quadrature points to physical coordinates
3. **Integration on Cut Domains**: Proper integration for negative, positive, and interface regions

## Basic Quadrature Usage

```python
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
import dolfinx
import numpy as np

# Create mesh and level set function
mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, ((0.0, 0.0), (1.0, 1.0)), (20, 20))
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Define level set (circle)
def circle(x):
    return np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2) - 0.2

level_set = dolfinx.fem.Function(V)
level_set.interpolate(circle)

# Generate runtime quadrature for the negative region
order = 2
quad_rules = runtime_quadrature(level_set, "phi<0", order)

# Get physical coordinates of quadrature points
phys_points = physical_points(quad_rules, mesh)
```

## Runtime Quadrature Generation

### `runtime_quadrature(level_set, ls_part, order)`
Generate quadrature rules for level set-defined regions.

**Parameters:**
- `level_set`: DOLFINx Function representing the level set
- `ls_part`: String specifying the region:
  - `"phi<0"` - Negative region (inside)
  - `"phi>0"` - Positive region (outside)
  - `"phi=0"` - Interface/boundary
- `order`: Quadrature order for polynomial exactness

**Returns:** QuadratureRules object containing points and weights

### `physical_points(runtime_rules, mesh)`
Convert reference quadrature points to physical coordinates.

**Parameters:**
- `runtime_rules`: QuadratureRules object from `runtime_quadrature`
- `mesh`: DOLFINx mesh object

**Returns:** NumPy array of physical coordinates

## Complete Example: Cut Domain Integration

```python
import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.fem
import ufl
from cutfemx.level_set import locate_entities, cut_entities
from cutfemx.mesh import create_cut_mesh
from cutfemx.quadrature import runtime_quadrature, physical_points
from cutfemx.fem import cut_form, assemble_scalar

# Create mesh and function space
mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, 
    ((0.0, 0.0), (1.0, 1.0)), 
    (15, 15),
    dolfinx.mesh.CellType.triangle
)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Define level set function
def circle(x):
    return np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2) - 0.2

level_set = dolfinx.fem.Function(V)
level_set.interpolate(circle)

# Find cut entities
dim = mesh.topology.dim
intersected_entities = locate_entities(level_set, dim, "phi=0")
inside_entities = locate_entities(level_set, dim, "phi<0")

# Get DOF coordinates for cut cell creation
dof_coordinates = V.tabulate_dof_coordinates()

# Create cut cells
cut_cells = cut_entities(level_set, dof_coordinates, intersected_entities, dim, "phi<0")

# Create cut mesh
cut_mesh = create_cut_mesh(mesh.comm, cut_cells, mesh, inside_entities)

# Generate runtime quadrature
order = 2
quad_rules = runtime_quadrature(level_set, "phi<0", order)

# Get physical points for visualization
phys_points = physical_points(quad_rules, mesh)

print(f"Generated {len(phys_points)} quadrature points")
print(f"Physical points shape: {phys_points.shape}")
```

## Integration with Cut Finite Element Assembly

The runtime quadrature integrates with CutFEMx's assembly system:

```python
# Define variational form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dolfinx.fem.Constant(mesh, 1.0)

# Bilinear and linear forms
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Create cut forms using the cut mesh
form_a = cut_form(a, cut_mesh)
form_L = cut_form(L, cut_mesh)

# Assembly uses the runtime quadrature automatically
A = cutfemx.fem.assemble_matrix(form_a)
b = cutfemx.fem.assemble_vector(form_L)
```

## Level Set Regions

### Available Region Types

1. **Negative Region (`"phi<0"`)**: Interior of the level set
2. **Positive Region (`"phi>0"`)**: Exterior of the level set  
3. **Interface (`"phi=0"`)**: Boundary/interface

### Usage Examples

```python
# Quadrature for different regions
interior_quad = runtime_quadrature(level_set, "phi<0", order=2)
exterior_quad = runtime_quadrature(level_set, "phi>0", order=2)
boundary_quad = runtime_quadrature(level_set, "phi=0", order=2)

# Get physical points for each region
interior_points = physical_points(interior_quad, mesh)
exterior_points = physical_points(exterior_quad, mesh)
boundary_points = physical_points(boundary_quad, mesh)
```

## Practical Considerations

### Quadrature Order Selection
- **Order 1-2**: Sufficient for linear problems
- **Order 3-4**: Recommended for higher-order elements
- **Higher orders**: For problems requiring high accuracy

### Performance Tips
- Generate quadrature rules once and reuse when possible
- For time-dependent problems, regenerate only when geometry changes
- Consider caching quadrature rules for repeated computations

### Integration with Level Set Functions
The quadrature system works seamlessly with DOLFINx Functions:

```python
# Any DOLFINx Function can serve as a level set
level_set = dolfinx.fem.Function(V)
level_set.interpolate(your_geometry_function)

# Quadrature automatically adapts to the geometry
quad_rules = runtime_quadrature(level_set, "phi<0", order)
```

## Examples

See the demonstration scripts in the `python/demo/` directory:

- `demo_runtime_quadrature.py` - Basic runtime quadrature generation
- `demo_cut_poisson.py` - Integration with cut finite element assembly

## Next Steps

- **[Cut Meshes](cut-meshes.md)** - Understanding cut mesh generation
- **[DOF Constraints](dof-constraints.md)** - Constraint handling for cut elements
- **[Examples Gallery](../examples/)** - Complete quadrature examples

---

*Accurate quadrature is essential for reliable cut finite element computations. CutFEMx provides the runtime tools needed for proper integration on complex cut geometries.*
    cut_mesh,
    cutfemx::DecompositionType::Triangulation  // or Tetrahedral for 3D
);

// Set integration accuracy
sub_quad.set_polynomial_degree(4);  // Exact for polynomials up to degree 4
sub_quad.set_subdivision_depth(3);  // Maximum recursive subdivisions

// Generate quadrature for specific element
auto rule = sub_quad.generate_rule(element_index);

// For visualization and debugging
auto sub_elements = sub_quad.get_subelements(element_index);
```

### Triangulation-based Decomposition
```cpp
// 2D cut elements decomposed into triangles
cutfemx::TriangulationQuadrature tri_quad(cut_mesh);

// Configure triangulation algorithm
tri_quad.set_algorithm(cutfemx::TriangulationAlgorithm::Delaunay);
tri_quad.set_vertex_tolerance(1e-12);  // Geometric tolerance
tri_quad.enable_quality_improvement(true);

// Generate high-quality triangulation
for (auto element : cut_mesh.cut_elements()) {
    auto triangles = tri_quad.triangulate_element(element);
    
    // Each triangle gets standard Gaussian quadrature
    for (const auto& triangle : triangles) {
        auto gauss_rule = cutfemx::GaussianQuadrature::triangle(polynomial_degree);
        // Transform to physical triangle and add to overall rule
    }
}
```

## Moment Fitting Quadrature

For cases where sub-element decomposition is too expensive:

```cpp
#include <cutfemx/quadrature/moment_fitting.h>

cutfemx::MomentFittingQuadrature moment_quad(
    cut_mesh,
    polynomial_degree,
    max_quadrature_points  // Balance accuracy vs. efficiency
);

// Configure moment fitting
moment_quad.set_optimization_tolerance(1e-10);
moment_quad.set_regularization_parameter(1e-8);  // For ill-conditioned systems

// Generate quadrature rule
auto rule = moment_quad.generate_rule(element_index);

// Check integration accuracy
double accuracy = moment_quad.check_accuracy(rule, polynomial_degree);
if (accuracy < desired_tolerance) {
    // Increase number of points or use different strategy
}
```

### Moment Fitting Theory
Moment fitting solves the linear system:
```
∑ᵢ wᵢ ∫ pⱼ(xᵢ) dΩ = ∫ pⱼ(x) dΩ
```
where pⱼ are polynomial basis functions up to the desired degree.

```cpp
// Custom moment fitting with specific polynomial basis
std::vector<std::function<double(const std::array<double, 2>&)>> basis = {
    [](const auto& x) { return 1.0; },                    // Constant
    [](const auto& x) { return x[0]; },                   // Linear x
    [](const auto& x) { return x[1]; },                   // Linear y
    [](const auto& x) { return x[0] * x[0]; },            // Quadratic x²
    [](const auto& x) { return x[0] * x[1]; },            // Bilinear xy
    [](const auto& x) { return x[1] * x[1]; }             // Quadratic y²
};

cutfemx::CustomMomentFitting custom_quad(cut_mesh, basis);
auto rule = custom_quad.generate_rule(element_index);
```

## Adaptive Quadrature

For integrands with sharp gradients or discontinuities:

```cpp
#include <cutfemx/quadrature/adaptive_quadrature.h>

// Define integrand function
auto integrand = [&](const std::array<double, 2>& x) -> double {
    // Example: function with sharp gradient near boundary
    double dist = level_set_function(x);
    return std::exp(-100.0 * dist * dist);  // Sharp Gaussian near boundary
};

cutfemx::AdaptiveQuadrature adaptive_quad(cut_mesh);
adaptive_quad.set_error_tolerance(1e-8);
adaptive_quad.set_max_subdivisions(10);

// Integrate with automatic refinement
double integral = adaptive_quad.integrate(element_index, integrand);
```

### Adaptive Strategies
```cpp
// Error-based refinement
adaptive_quad.set_refinement_strategy(
    cutfemx::RefinementStrategy::ErrorBased,
    error_threshold
);

// Gradient-based refinement
adaptive_quad.set_refinement_strategy(
    cutfemx::RefinementStrategy::GradientBased,
    gradient_threshold
);

// Level set distance-based refinement
adaptive_quad.set_refinement_strategy(
    cutfemx::RefinementStrategy::DistanceBased,
    distance_threshold
);
```

## Boundary Integration

Special quadrature rules for integrating along cut boundaries:

```cpp
#include <cutfemx/quadrature/boundary_quadrature.h>

// Generate quadrature for boundary integrals
cutfemx::BoundaryQuadrature boundary_quad(cut_mesh, level_set_function);

// Configure boundary extraction
boundary_quad.set_boundary_tolerance(1e-10);
boundary_quad.set_curve_approximation_order(3);  // Piecewise cubic approximation

// Get boundary quadrature for element
auto boundary_rule = boundary_quad.generate_boundary_rule(element_index, polynomial_degree);

// Integrate boundary term (e.g., for Nitsche method)
double boundary_integral = 0.0;
for (std::size_t i = 0; i < boundary_rule.points().size(); ++i) {
    const auto& point = boundary_rule.points()[i];
    const auto& normal = boundary_rule.normals()[i];  // Outward normal
    double weight = boundary_rule.weights()[i];
    
    double boundary_term = compute_boundary_term(point, normal);
    boundary_integral += weight * boundary_term;
}
```

### Boundary Curve Parameterization
```cpp
// Extract boundary curves as parametric representations
auto boundary_curves = boundary_quad.extract_boundary_curves(element_index);

for (const auto& curve : boundary_curves) {
    // Parameterize curve as γ(t) for t ∈ [0, 1]
    auto parametrization = curve.parametrization();
    auto tangent = curve.tangent();
    auto normal = curve.normal();
    
    // Integrate along curve using parameter t
    double curve_integral = 0.0;
    auto gauss_1d = cutfemx::GaussianQuadrature::line(polynomial_degree);
    
    for (const auto& [t, w] : gauss_1d) {
        auto point = parametrization(t);
        auto tang = tangent(t);
        double jacobian = tang.norm();  // |dγ/dt|
        
        curve_integral += w * jacobian * boundary_function(point);
    }
}
```

## High-Order Quadrature

For high-order finite element methods:

```cpp
// High-order accurate quadrature
cutfemx::HighOrderQuadrature ho_quad(cut_mesh);

// Configure for high polynomial degrees
ho_quad.set_polynomial_degree(8);  // Exact integration up to degree 8
ho_quad.set_geometric_approximation_order(6);  // High-order boundary approximation

// Use specialized high-order algorithms
ho_quad.set_algorithm(cutfemx::QuadratureAlgorithm::HighOrderMomentFitting);

// Generate rule with optimal point distribution
auto ho_rule = ho_quad.generate_rule(element_index);

// Verify integration accuracy for high-order polynomials
bool accurate = ho_quad.verify_polynomial_exactness(ho_rule, 8);
```

## Performance Optimization

### Quadrature Caching
```cpp
// Cache quadrature rules for repeated use
cutfemx::QuadratureCache cache;

// Enable caching for cut elements
cache.enable_cut_element_caching(true);
cache.set_cache_policy(cutfemx::CachePolicy::LeastRecentlyUsed);

// Generate and cache rules
for (auto element : cut_mesh.cut_elements()) {
    auto rule = cache.get_or_create_rule(element, polynomial_degree);
    // Use rule for integration
}

// Cache statistics
std::cout << "Cache hit rate: " << cache.hit_rate() << std::endl;
std::cout << "Memory usage: " << cache.memory_usage() << " MB" << std::endl;
```

### Parallel Quadrature Generation
```cpp
// Generate quadrature rules in parallel
cutfemx::ParallelQuadratureFactory parallel_factory(cut_mesh, MPI_COMM_WORLD);

// Distribute elements across processes
parallel_factory.distribute_elements(cut_mesh.cut_elements());

// Generate all rules in parallel
auto all_rules = parallel_factory.generate_all_rules(polynomial_degree);

// Synchronize results across processes
parallel_factory.synchronize_rules();
```

## Quality Assessment

### Integration Error Analysis
```cpp
#include <cutfemx/quadrature/quality_assessment.h>

cutfemx::QuadratureQuality quality_checker(cut_mesh);

// Test integration of known functions
std::vector<std::function<double(const std::array<double, 2>&)>> test_functions = {
    [](const auto& x) { return 1.0; },                           // Constant
    [](const auto& x) { return x[0] + x[1]; },                   // Linear
    [](const auto& x) { return x[0] * x[0] + x[1] * x[1]; },     // Quadratic
    [](const auto& x) { return std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]); }  // Oscillatory
};

for (auto element : cut_mesh.cut_elements()) {
    auto rule = quadrature_factory.create_rule(element, polynomial_degree);
    
    for (const auto& test_func : test_functions) {
        double numerical_integral = quality_checker.integrate(rule, test_func);
        double analytical_integral = quality_checker.analytical_integral(element, test_func);
        double error = std::abs(numerical_integral - analytical_integral);
        
        std::cout << "Element " << element << ", Error: " << error << std::endl;
    }
}
```

### Condition Number Analysis
```cpp
// For moment fitting quadrature, check condition number
auto moment_matrix = moment_quad.get_moment_matrix(element_index);
double condition_number = compute_condition_number(moment_matrix);

if (condition_number > 1e12) {
    std::cout << "Warning: Ill-conditioned moment fitting system" << std::endl;
    // Consider using regularization or different strategy
}
```

## Integration with FEniCSx

### Custom Integration Measures
```cpp
#include <dolfinx/fem/Form.h>
#include <cutfemx/integration/cut_measures.h>

// Create custom measure for cut domain integration
auto cut_dx = cutfemx::create_cut_measure(cut_mesh, polynomial_degree);

// Use in UFL forms (pseudo-code)
// a = inner(grad(u), grad(v)) * cut_dx
// L = f * v * cut_dx

// Apply to FEniCSx forms
dolfinx::fem::Form<double> bilinear_form = /* ... */;
cutfemx::apply_cut_integration(bilinear_form, cut_dx);
```

### Assembly Integration
```cpp
// Custom assembler with cut quadrature
cutfemx::CutFEMAssembler assembler(cut_mesh);

// Configure quadrature strategy
assembler.set_quadrature_strategy(cutfemx::QuadratureStrategy::SubelementTriangulation);
assembler.set_polynomial_degree(function_space->element()->degree() + 1);

// Assemble with custom quadrature
auto A = assembler.assemble_matrix(bilinear_form);
auto b = assembler.assemble_vector(linear_form);
```

## Advanced Topics

### Quadrature for Discontinuous Functions
```cpp
// Handle functions with jumps across interfaces
cutfemx::DiscontinuousQuadrature disc_quad(cut_mesh);

// Separate quadrature for each subdomain
auto inside_rule = disc_quad.generate_inside_rule(element_index, polynomial_degree);
auto outside_rule = disc_quad.generate_outside_rule(element_index, polynomial_degree);

// Integrate discontinuous function
double total_integral = 0.0;
total_integral += integrate_with_rule(inside_rule, inside_function);
total_integral += integrate_with_rule(outside_rule, outside_function);
```

### Level Set Gradient Integration
```cpp
// Special handling for level set gradient singularities
cutfemx::LevelSetGradientQuadrature ls_grad_quad(cut_mesh, level_set_function);

// Use higher-order quadrature near zero level set
ls_grad_quad.set_singularity_treatment(cutfemx::SingularityTreatment::HighOrderNearBoundary);
ls_grad_quad.set_boundary_neighborhood_size(2.0 * mesh_size);

auto rule = ls_grad_quad.generate_rule(element_index, polynomial_degree);
```

## Examples and Best Practices

### Complete Integration Example
```cpp
// See examples/integration/cut-element-integration.cpp
#include <cutfemx/examples/integration_example.h>

cutfemx::IntegrationExample example(mesh, level_set_function);

// Compare different quadrature strategies
example.compare_strategies({
    cutfemx::QuadratureStrategy::SubelementTriangulation,
    cutfemx::QuadratureStrategy::MomentFitting,
    cutfemx::QuadratureStrategy::Adaptive
});

// Generate convergence study
example.convergence_study(polynomial_degrees, mesh_refinements);
```

### Best Practices Summary
1. **Strategy Selection**: Use sub-element decomposition for robustness, moment fitting for efficiency
2. **Polynomial Degree**: Set quadrature degree ≥ 2 × finite element degree
3. **Geometric Tolerance**: Use tight tolerances (1e-12) for geometric operations
4. **Caching**: Cache rules for elements that don't change between time steps
5. **Validation**: Always verify integration accuracy for new problems

## Next Steps

- **[Stabilization Techniques](stabilization.md)** - Learn about numerical stabilization
- **[Boundary Conditions](boundary-conditions.md)** - Apply boundary conditions using boundary integration
- **[Examples Gallery](../examples/)** - See quadrature in complete examples

---

*Accurate quadrature is the foundation of reliable cut finite element simulations. Choose the appropriate strategy based on your problem's requirements for accuracy and computational efficiency.*
