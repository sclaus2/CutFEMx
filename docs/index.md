# CutFEMx documentation

CutFEMx provides cut finite element tools for the FEniCSx 0.11 release stack.
The current Python workflow is centered around `cutfemx.cut(...)`,
`cutfemx.locate_entities(...)`, `cutfemx.runtime_quadrature(...)`, and
`cutfemx.create_cut_mesh(...)`.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

getting-started/index
user-guide/index
stl_distance
examples/index
```

## Getting started

- [Installation](getting-started/installation.md): build FEniCSx, runintgen,
  CutCells, CutFEMx, and the docs in an isolated environment.
- [Quickstart](getting-started/quickstart.md): solve a first cut Poisson
  problem.

## User guide

- [Level sets](user-guide/level-sets.md): define implicit domains with
  DOLFINx functions.
- [Element classification](user-guide/element-classification.md): use
  `cutfemx.cut(...)` and selector strings.
- [Cut meshes](user-guide/cut-meshes.md): create visualization meshes and
  transfer functions.
- [Runtime quadrature](user-guide/quadrature.md): build quadrature providers
  for cells, interfaces, facets, and skeletons.
- [Boundary conditions](user-guide/boundary-conditions.md): impose conditions
  on embedded boundaries with Nitsche terms.
- [Stabilization](user-guide/stabilization.md): use ghost-penalty facets around
  cut cells.
- [DOF constraints](user-guide/dof-constraints.md): deactivate inactive
  background degrees of freedom.
- [Visualization](user-guide/visualization.md): inspect cut meshes and runtime
  quadrature points.
