# Visualization

Visualization is part of the numerical workflow for unfitted methods. A
standard mesh plot may look correct while the selected cut domain, the runtime
quadrature points, or the inactive degrees of freedom are wrong. CutFEMx
therefore provides tools for inspecting the geometry used by assembly, not only
the final solution.

```{admonition} What to inspect first
:class: practice

Before interpreting a numerical solution, check the geometric objects that
define the discrete problem:

- the sign and location of the level-set field,
- the cells selected by `"phi<0"`, `"phi=0"`, and `"phi<=0"`,
- the reconstructed cut mesh,
- the runtime quadrature points and weights,
- the inactive-DOF indicator,
- the orientation of normals on embedded boundaries.

Most wrong CutFEM formulations first appear as an inconsistent selector,
inverted sign convention, missing cut cells, or incorrect normal orientation.
```

## Cut Mesh Visualization

Create a cut mesh for the selected region:

```python
cut_data = cutfemx.cut(phi)
cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
```

If `cut_mesh.mesh` is not `None`, it can be converted to a VTK grid:

```python
from dolfinx import plot
import pyvista

cells, types, x = plot.vtk_mesh(cut_mesh.mesh)
grid = pyvista.UnstructuredGrid(cells, types, x)
grid.cell_data["parent_index"] = cut_mesh.parent_index
grid.cell_data["is_cut_cell"] = cut_mesh.is_cut_cell
```

The parent index identifies which background cell generated each output cell.
The cut-cell marker distinguishes reconstructed pieces from uncut cells
included in a `full` cut mesh.

```{admonition} Diagnostic interpretation
:class: theory

For a domain mesh created with selector `"phi<0"`, the reconstructed geometry
should approximate

$$
\Omega_h=\{x:\phi_h(x)<0\}.
$$

For an interface mesh created with selector `"phi=0"`, the geometry should
approximate

$$
\Gamma_h=\{x:\phi_h(x)=0\}.
$$

If these objects do not match the intended geometry, the variational form will
also be assembled on the wrong domain.
```

## Transferred Functions

Finite element solutions are computed on the background mesh. To plot them on
a cut mesh, first transfer the background function:

```python
phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)
u_cut = cutfemx.fem.cut_function(uh, cut_mesh)
```

Then build a VTK grid from the transferred function space:

```python
cells, types, x = plot.vtk_mesh(u_cut.function_space)
solution_grid = pyvista.UnstructuredGrid(cells, types, x)
solution_grid.point_data["u"] = u_cut.x.array[: solution_grid.n_points]
```

The transferred function is the same finite element field evaluated on the
reconstructed geometry. It is useful for physical-domain plots and for checking
whether the solution behaves correctly near the cut boundary.

## Runtime Quadrature Points

Quadrature providers compute physical points lazily:

```python
rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order=4)
points = rules.with_physical_points().physical_points
```

For PyVista:

```python
point_cloud = pyvista.PolyData(points.T)
point_cloud.point_data["weight"] = rules.weights
```

Plotting the quadrature points is a useful check for runtime integration. For a
volume rule, the points should fill $K\cap\Omega_h$ inside cut cells. For an
interface rule, they should lie on the reconstructed zero contour. Large or
negative-looking weights, empty point clouds, or points on the wrong side of
the interface usually indicate a selector or host-entity mismatch.

## Active-Domain Indicator

The inactive-DOF indicator can be visualized on the background mesh:

```python
active_domain = cutfemx.fem.active_domain(a_form)
indicator = active_domain.indicator
```

This helps check the algebraic support of the assembled system. For a scalar
problem on $\Omega_h$, the active region should cover the support of basis
functions participating in physical and stabilization integrals. It may be
slightly larger than the physical domain when ghost penalties are present.

## Complete Examples

See `python/demo/demo_poisson.py`,
`python/demo/demo_boundary_sphere_perimeter.py`, and
`python/demo/demo_interface_poisson.py` for complete plotting routines. The
examples combine cut meshes, transferred functions, and quadrature point
clouds to inspect different parts of the CutFEM workflow.
