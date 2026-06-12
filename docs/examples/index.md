# Examples

The examples in this section are narrative descriptions of selected scripts in
`python/demo`. They are not executed automatically during the documentation
build. The goal is to explain the variational formulation, the cut geometry,
and the CutFEMx API choices before pointing to the runnable script.

For a more presentation-oriented walkthrough with embedded PyVista screenshots,
see the [Tutorial book](../tutorials/index.md).

```{toctree}
:maxdepth: 1
:caption: Examples

poisson
dg-poisson
stokes
interface-poisson
stl-distance
boundary-perimeter
```

## Available Scripts

The complete demo scripts live in `python/demo`. The most useful entry points
for new users are:

- `demo_poisson.py`: continuous Galerkin Poisson problem on a flower-shaped
  cut domain.
- `demo_dg_poisson.py`: cut-domain SIPG formulation with runtime interior
  skeleton quadrature.
- `demo_stokes.py`: Taylor-Hood Stokes flow outside a circular cut obstacle.
- `demo_interface_poisson.py`: two-domain unfitted Poisson interface problem.
- `demo_boundary_sphere_perimeter.py`: runtime quadrature on cut exterior
  facets.
- `demo_stl_distance.py`: signed distance computation from an STL surface.
- `demo_reinit.py`: level-set redistancing.
