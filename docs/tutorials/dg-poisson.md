# DG Poisson

This tutorial follows `python/demo/demo_dg_poisson.py`. It extends the scalar
cut Poisson demo to a discontinuous Galerkin space, so the physical cells, the
embedded boundary, and the interior skeleton all become part of the cut
integration problem.
The cut DG formulation is closest to the stabilized cut discontinuous
Galerkin framework cited in the related literature below.

```{raw} html
<figure class="tutorial-figure">
  <img class="tutorial-image" src="../_static/tutorials/dg-poisson-scene.png" alt="PyVista DG Poisson overview">
  <figcaption>The DG problem is solved on an unfitted circular domain; active cells and the interior skeleton are clipped by the same level set.</figcaption>
</figure>
```

## Model Problem

The physical domain is the negative phase of an offset disk,

$$
\Omega=\{x\in[-1,1]^2:\phi(x)<0\},\qquad
\phi(x,y)=\sqrt{(x-0.08)^2+(y+0.04)^2}-0.61 .
$$

The demo solves

$$
-\Delta u=f\quad\text{in }\Omega,\qquad u=g\quad\text{on }\Gamma=\{\phi=0\},
$$

with the manufactured field

$$
u_\mathrm{ex}(x,y)=\sin(\pi x)\sin(\pi y).
$$

## Implementation Order

The demo runs in this order:

1. Define helpers for the circular level set, serial solve, XDMF interpolation,
   and output.
2. Build the triangular background mesh and interpolate the P1 level set.
3. Classify inside/active/cut cells, create cell and interface runtime rules,
   then cut the active interior skeleton.
4. Build `dx_omega`, `dS_omega`, `dx_gamma`, and `dS_ghost`.
5. Build the DG space, SIPG volume/skeleton terms, Nitsche boundary terms, and
   ghost penalty.
6. Assemble, deactivate inactive dofs from the CutFEMx form active domain, and
   solve the serial sparse system.
7. Interpolate exact/error fields, assemble the cut-domain $L^2$ error, print
   diagnostics, and write background plus cut-domain output.

## Geometry And Active Cells

The level set is cut once, and predicates select the cells required by the DG
formulation. Interior cells use ordinary cell integration, cut cells receive
runtime quadrature, and the union forms the active computational domain.

```{raw} html
<figure class="tutorial-figure">
  <img class="tutorial-image" src="../_static/tutorials/dg-poisson-geometry-scene.png" alt="PyVista DG Poisson geometry scene">
  <figcaption>Blue cells are fully inside the disk; orange cells are intersected by the embedded boundary.</figcaption>
</figure>
```

```python
cell_cut = cutfemx.cut(phi)
inside_cells = cutfemx.locate_entities(cell_cut, "phi<0")
active_cells = cutfemx.locate_entities(cell_cut, "phi<=0")
cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
```

## Runtime Quadrature

The DG form uses three kinds of cut quadrature: volume rules on
$K\cap\Omega$, interface rules on $K\cap\Gamma$, and skeleton rules on
$F\cap\Omega$ for active interior facets.

```{raw} html
<figure class="tutorial-figure">
  <iframe class="tutorial-frame" src="../_static/tutorials/dg-poisson-quadrature-view.html" title="Interactive DG Poisson quadrature view" loading="lazy" allowfullscreen></iframe>
  <figcaption>Blue points mark cut-volume rules, magenta points mark embedded-boundary rules, and orange points mark skeleton rules on the clipped interior facets.</figcaption>
</figure>
```

```python
cell_rules = cutfemx.runtime_quadrature(cell_cut, "phi<0", order)
interface_rules = cutfemx.runtime_quadrature(cell_cut, "phi=0", order)

skeleton_facets = cutfemx.interior_facets_for_cells(msh, active_cells)
skeleton_cut = cutfemx.cut(phi, skeleton_facets, facet_dim)

omega_interior_facets = cutfemx.locate_entities(skeleton_cut, "phi<0")
omega_cut_facet_rules = cutfemx.runtime_quadrature(skeleton_cut, "phi<0", order)
cut_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi=0")
```

These rule sets define the measures used by the UFL form:

```python
dx_omega = ufl.Measure(
    "dx", domain=msh, subdomain_id=0, subdomain_data=[inside_cells, cell_rules]
)
dx_gamma = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=interface_rules)
dS_omega = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[omega_interior_facets, omega_cut_facet_rules],
)
```

For the DG skeleton this measure has two pieces. `omega_interior_facets`
contains the full background interior facets that lie inside $\Omega$.
`omega_cut_facet_rules` contains runtime quadrature on the clipped pieces
$F\cap\Omega$ for facets cut by $\Gamma$; its parent map corresponds to the
`cut_skeleton_facets` diagnostic set.

## Active DG Skeleton

The skeleton terms are only integrated on the part of each active interior
facet that belongs to $\Omega$. Facets fully inside $\Omega$ contribute as
ordinary background facets. Intersected facets only contribute on the clipped
segment $F\cap\Omega$, so the background support facet and the integration
facet are not the same geometric object.

```{raw} html
<figure class="tutorial-figure">
  <img class="tutorial-image" src="../_static/tutorials/dg-poisson-skeleton-scene.png" alt="PyVista DG Poisson skeleton scene">
  <figcaption>Muted orange facets are integrated fully; gray facets are intersected support facets, and the darker orange subsegments are the clipped parts used by `dS_omega`.</figcaption>
</figure>
```

With $h$ the cell diameter and $\sigma$ the DG penalty, the symmetric interior
penalty part has the form

$$
\int_\Omega \nabla u\cdot\nabla v\,dx
-\int_{\mathcal F_h^\Omega}\{\nabla u\}\cdot [v n]\,dS
-\int_{\mathcal F_h^\Omega}\{\nabla v\}\cdot [u n]\,dS
+\int_{\mathcal F_h^\Omega}\frac{\sigma}{h}\,[u n]\cdot[v n]\,dS .
$$

```python
n_facet = ufl.FacetNormal(msh)
h = ufl.CellDiameter(msh)
h_avg = ufl.avg(h)
sigma = penalty * degree**2

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
jump_u = ufl.jump(u, n_facet)
jump_v = ufl.jump(v, n_facet)
a += -ufl.inner(ufl.avg(ufl.grad(u)), jump_v) * dS_omega
a += -ufl.inner(ufl.avg(ufl.grad(v)), jump_u) * dS_omega
a += sigma / h_avg * ufl.inner(jump_u, jump_v) * dS_omega
```

## Boundary And Ghost Stabilization

The Dirichlet condition on $\Gamma$ is imposed with Nitsche terms. A ghost
penalty then couples cut cells to neighboring active cells, reducing the
conditioning sensitivity caused by very small intersections.

```{raw} html
<figure class="tutorial-figure">
  <img class="tutorial-image" src="../_static/tutorials/dg-poisson-penalty-scene.png" alt="PyVista DG Poisson ghost penalty scene">
  <figcaption>Red facets indicate the ghost-penalty band near the embedded boundary.</figcaption>
</figure>
```

```python
n_gamma = cutfemx.normal(phi)
ghost_facets = cutfemx.ghost_penalty_facets(cell_cut, "phi<0")
dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)
```

After solving, the demo follows the scalar Poisson output pattern: interpolate
the exact DG field, form `u_error`, assemble the $L^2$ error on `dx_omega`,
then write both the background fields and their physical cut-domain
restrictions.

## Solution Output

The final field is visualized on the cut disk. The tutorial view is kept flat,
matching the two-dimensional PyVista interaction used for mesh inspection.

```{raw} html
<figure class="tutorial-figure">
  <img class="tutorial-image" src="../_static/tutorials/dg-poisson-solution-scene.png" alt="PyVista DG Poisson solution scene">
  <figcaption>The scalar solution is color-mapped on the physical disk without warping the geometry.</figcaption>
</figure>
```

The script writes background and cut-domain fields in `dg_poisson_xdmf/`.

## Related Literature

- C. Gürkan and A. Massing,
  ["A stabilized cut discontinuous Galerkin framework for elliptic boundary
  value and interface problems"](https://doi.org/10.1016/j.cma.2018.12.041),
  *Computer Methods in Applied Mechanics and Engineering* 348, 466-499,
  2019. This paper develops the unfitted SIPG/CutDG framework for elliptic
  boundary-value and interface problems, including Poisson problems and
  ghost-penalty stabilization.

## Run The Demo

```bash
python python/demo/demo_dg_poisson.py
```

## Full Source

The complete source remains available in the repository:
[python/demo/demo_dg_poisson.py](../../python/demo/demo_dg_poisson.py).
