# Tutorials

These tutorials are structured as a software-demo book. Each chapter starts
from a runnable script in `python/demo`, explains the mathematical problem,
then follows the CutFEMx workflow used in the code.

The embedded panels are PyVista screenshots generated from the same DOLFINx
meshes and CutFEMx classification steps as the demo scripts. Each chapter uses
several focused panels so the geometry being discussed is visible next to the
corresponding code and equations.

```{toctree}
:maxdepth: 1
:caption: Tutorial Book

poisson
dg-poisson
interface-poisson
surface-poisson-dg
stokes
boundary-perimeter
stl-distance
elasticity
moving-poisson
```

## Suggested Session Flow

1. Start with [Cut Poisson](poisson.md) to introduce level sets, entity
   classification, runtime quadrature, and inactive-DOF deactivation.
2. Move to [DG Poisson](dg-poisson.md) to show why interior facets also need
   cut quadrature.
3. Use [Interface Poisson](interface-poisson.md) to show two cut phases coupled
   across one unfitted interface.
4. Use [Surface Poisson DG](surface-poisson-dg.md) to demonstrate integration
   directly on $\phi=0$.
5. Use [Stokes](stokes.md) and [Elasticity](elasticity.md) to show mixed and
   vector-valued problems.
6. Use [Exterior-Facet Runtime Quadrature](boundary-perimeter.md) for
   codimension-two quadrature on boundary facets.
7. Use [STL Distance](stl-distance.md) to see how STL geometry becomes a
   level-set field.
8. Finish with [Moving Poisson](moving-poisson.md) to show the same background
   mesh with rebuilt cut data over several geometry positions.
