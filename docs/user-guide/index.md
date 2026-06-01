# User Guide

The user guide describes the current CutFEMx workflow around level-set
geometry, entity classification, runtime quadrature, boundary terms,
stabilization, algebraic constraints, and visualization.

CutFEM methods keep the mesh simple and move the geometric complexity into
classification, quadrature, and stabilization. The background mesh
${\mathcal T}_h$ covers a larger computational box, while the physical domain
is recovered from a level-set function:

$$
\Omega = \{x \in \Omega_0 : \phi(x) < 0\},
\qquad
\Gamma = \{x \in \Omega_0 : \phi(x) = 0\}.
$$

```{admonition} Core idea
:class: theory

CutFEMx does not ask DOLFINx to assemble on a curved mesh. Instead, it keeps the
usual finite element space on the background mesh and replaces selected
integrals by cut-cell quadrature rules:

$$
\int_{\Omega} f(x)\,dx
\approx
\sum_{K \in {\mathcal T}_h}
\sum_{q \in Q(K \cap \Omega)}
w_q f(x_q).
$$

This is why the high-level workflow always separates geometry from analysis:
classify entities, build runtime quadrature, write the variational form on the
selected measures, stabilize the cut configuration, and constrain inactive
degrees of freedom.
```

```{toctree}
:maxdepth: 2
:caption: User guide

level-sets
element-classification
cut-meshes
quadrature
boundary-conditions
stabilization
dof-constraints
visualization
```

## Workflow

1. Define a DOLFINx level-set function that describes the physical geometry.
2. Classify background cells, facets, or selected entities with
   `cutfemx.cut(...)` and `cutfemx.locate_entities(...)`.
3. Build runtime quadrature providers for cut cells, embedded interfaces,
   exterior facets, or active skeleton facets.
4. Express the PDE in UFL using the corresponding standard and runtime
   measures.
5. Add weak boundary/interface terms, typically Nitsche terms, when the
   boundary is not fitted to the mesh.
6. Add ghost-penalty stabilization to control small cut cells and extension
   into the cut-cell band.
7. Assemble with CutFEMx and deactivate inactive degrees of freedom.
8. Inspect cut meshes, quadrature points, and active-domain indicators before
   trusting a numerical solution.
