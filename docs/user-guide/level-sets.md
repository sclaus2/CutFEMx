# Level Sets

In CutFEMx the geometry is not encoded in the mesh topology. It is encoded in a
scalar finite element function $\phi$, whose zero contour represents the
boundary or interface of interest. The sign of this function determines which
part of the background mesh belongs to the physical domain:

- `phi < 0`: inside the physical domain
- `phi = 0`: embedded interface or boundary
- `phi > 0`: outside the physical domain

```{admonition} Implicit geometry
:class: geometry

Let $\Omega_0$ denote the background domain. A level-set function
$\phi:\Omega_0\to\mathbb R$ defines the two open phases

$$
\Omega^- = \{x\in\Omega_0:\phi(x)<0\},
\qquad
\Omega^+ = \{x\in\Omega_0:\phi(x)>0\},
$$

separated by the embedded interface

$$
\Gamma = \partial \Omega^- \cap \Omega_0
       = \{x\in\Omega_0:\phi(x)=0\}.
$$

The convention $\phi<0$ for the physical side is not intrinsic; it is simply
the convention used by the selector strings in the examples. Reversing the sign
of $\phi$ reverses the meaning of `"phi<0"` and changes the orientation of the
level-set normal.
```

## Create A Level Set

```python
from mpi4py import MPI

import numpy as np
from dolfinx import fem, mesh

msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(32, 32),
    cell_type=mesh.CellType.triangle,
)

V_phi = fem.functionspace(msh, ("Lagrange", 2))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) - 0.5)
```

The same DOLFINx function can be used both in CutFEMx geometry routines and in
UFL expressions. In the example above, the zero contour is the circle of radius
`0.5`, and the negative phase is the disk.

## Selector Semantics

Selectors such as `"phi<0"` and `"phi=0"` operate on the geometric
classification produced by the cutting step:

```python
cut_data = cutfemx.cut(phi)

inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
interface_cells = cutfemx.locate_entities(cut_data, "phi=0")
active_cells = cutfemx.locate_entities(cut_data, "phi<=0")
```

```{admonition} Classification is not an epsilon band
:class: theory

The selector `"phi=0"` denotes entities intersected by the zero level set. It
should be read geometrically as

$$
K\cap\Gamma \ne \emptyset,
$$

not as the pointwise condition $|\phi|<\varepsilon$. In the current CutFEMx
selector path, cell classification is based on the signs of the level-set
degrees of freedom: an entity is inside if all relevant values are negative,
outside if all are positive, and intersected otherwise. Thus a vertex value
equal to zero also places the entity in the intersected class.

This is separate from tolerances used in distance-field construction or other
geometric predicates. There is no user-facing epsilon parameter in
`cutfemx.locate_entities(...)`.
```

## Normals

For boundary terms on $\Gamma$, the normal is obtained from the gradient of the
level-set field. CutFEMx exposes this as a lazy quadrature function:

$$
n_\Gamma = \frac{\nabla \phi}{|\nabla \phi|}
$$

at runtime quadrature points.

```{admonition} Normal orientation
:class: theory

On a fitted boundary the outward normal is a property of the mesh facet. On an
embedded boundary it is instead a property of the scalar field used to describe
the geometry. For the negative phase $\Omega^-$, the normal associated with the
choice above is

$$
n_\Gamma(x_q) =
\frac{\nabla \phi(x_q)}{\sqrt{\nabla \phi(x_q)\cdot\nabla \phi(x_q)}}.
$$

The evaluation is delayed until the runtime quadrature points $x_q$ on the cut
boundary are known. In the implementation the denominator is protected by a
small floor, `1e-14`, to avoid division by zero if the gradient vanishes at a
quadrature point. This safeguard does not change the selector semantics above.
```

```python
import cutfemx

n_gamma = cutfemx.normal(phi)
```

Use `sign=-1.0` to reverse the orientation:

```python
n_inward = cutfemx.normal(phi, sign=-1.0)
```

## Updating Moving Level Sets

When the level-set values change, the previously constructed cut data no longer
represents the current geometry. Update the DOLFINx function and refresh the
cut state before rebuilding entity arrays or quadrature providers:

```python
cut_data = cutfemx.cut(phi)

phi.interpolate(new_level_set_expression)
cut_data.update()
```

After `cut_data.update()`, recompute any entity arrays and runtime quadrature
rules that depend on the position of the zero contour.
