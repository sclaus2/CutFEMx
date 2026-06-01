# Quickstart

This quickstart shows the current CutFEMx workflow on a scalar Poisson problem
posed on an implicit disk. The physical domain is

$$
\Omega = \{x \in [-1, 1]^2 : \phi(x) < 0\},
\qquad
\Gamma = \{x : \phi(x) = 0\}.
$$

```{admonition} Continuous problem
:class: formulation

We solve a Dirichlet problem on the unfitted domain:

$$
-\Delta u = f \quad \text{in } \Omega,
\qquad
u = g \quad \text{on } \Gamma.
$$

For the manufactured solution

$$
u_{\mathrm{ex}}(x,y) = \sin(\pi x)\sin(\pi y),
$$

the source term and boundary data are

$$
f = -\Delta u_{\mathrm{ex}}
  = 2\pi^2\sin(\pi x)\sin(\pi y),
\qquad
g = u_{\mathrm{ex}}\vert_{\Gamma}.
$$
```

The important API pattern is:

1. Interpolate a level-set function into a DOLFINx space.
2. Build one reusable `CutData` object with `cutfemx.cut(phi)`.
3. Use the `CutData` object to locate entities, create runtime quadrature, and
   create a visualization mesh.
4. Compile UFL forms with `cutfemx.fem.form(...)` before assembly.

## Imports

```python
from mpi4py import MPI

import cutfemx
import numpy as np
import ufl
from dolfinx import fem, mesh
```

## Background Mesh And Level Set

The level set is negative inside the disk and positive outside.

For this example,

$$
\phi(x,y) = \sqrt{x^2+y^2} - R,
\qquad R = 0.5.
$$

Hence

$$
\Omega = \{(x,y) : x^2+y^2 < R^2\},
\qquad
\Gamma = \{(x,y) : x^2+y^2 = R^2\}.
$$

```python
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(21, 21),
    cell_type=mesh.CellType.triangle,
)

V_phi = fem.functionspace(msh, ("Lagrange", 2))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2) - 0.5)
```

## Cut The Mesh

`cutfemx.cut(phi)` returns a `CutData` object. All later classification and
quadrature calls use this object, not the raw level-set function.

```python
cut_data = cutfemx.cut(phi)

inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
interface_cells = cutfemx.locate_entities(cut_data, "phi=0")
active_cells = cutfemx.locate_entities(cut_data, "phi<=0")
```

Selectors use the CutCells expression grammar. The default level-set name is
`phi`; with multiple level sets the names are `phi`, `phi1`, `phi2`, and so on.

```{admonition} Active cells and cut cells
:class: theory

The background mesh is not fitted to $\Gamma$. The active cell set is

$$
{\mathcal T}_h^\Omega =
\{K \in {\mathcal T}_h : K \cap \Omega \ne \emptyset\}.
$$

The interface cell set is

$$
{\mathcal T}_h^\Gamma =
\{K \in {\mathcal T}_h : K \cap \Gamma \ne \emptyset\}.
$$

In the code, `"phi<=0"` selects the active region and `"phi=0"` selects the
cells intersected by the embedded boundary.
```

## Runtime Quadrature And Cut Mesh

Cut cells need quadrature rules that are only known after cutting. CutFEMx
returns `runintgen`-compatible providers:

```{admonition} Runtime quadrature
:class: formulation

On a cut mesh, the cell integrals are evaluated as

$$
\int_\Omega F(x)\,dx
\approx
\sum_{K \in {\mathcal T}_h^\Omega}
\sum_{q=1}^{N_K} w_{Kq} F(x_{Kq}),
$$

where the quadrature points $x_{Kq}$ and weights $w_{Kq}$ are generated from
the local geometry $K \cap \Omega$.

Boundary integrals use a separate set of rules:

$$
\int_\Gamma G(x)\,ds
\approx
\sum_{K \in {\mathcal T}_h^\Gamma}
\sum_{q=1}^{M_K} \omega_{Kq} G(\xi_{Kq}).
$$

These rules are attached to UFL measures through `subdomain_data`.
```

```python
order = 4
dx_omega_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
dx_gamma_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
```

For visualization, create a cut mesh and transfer functions from the background
mesh:

```python
cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)
```

`cut_mesh.mesh` is a DOLFINx mesh. `cut_mesh.parent_index` maps cells in the
visualization mesh back to parent background cells.

## Variational Form

The unknown still lives on the background mesh. Runtime measures restrict the
integrals to the selected cut domain and embedded boundary.

```{admonition} Symmetric Nitsche formulation
:class: formulation

The homogeneous fitted weak form

$$
\int_\Omega \nabla u_h\cdot\nabla v_h\,dx
=
\int_\Omega f v_h\,dx
$$

is modified because $\Gamma$ cuts through mesh cells. The symmetric Nitsche form
for imposing $u=g$ on $\Gamma$ is: find $u_h \in V_h$ such that

$$
a_h(u_h,v_h) = L_h(v_h)
\qquad \forall v_h \in V_h,
$$

with

$$
\begin{aligned}
a_h(u_h,v_h)
&=
\int_\Omega \nabla u_h\cdot\nabla v_h\,dx
- \int_\Gamma (\nabla u_h\cdot n_\Gamma)v_h\,ds \\
&\quad
- \int_\Gamma (\nabla v_h\cdot n_\Gamma)u_h\,ds
+ \int_\Gamma \frac{\gamma}{h}u_hv_h\,ds,
\end{aligned}
$$

and

$$
L_h(v_h)
=
\int_\Omega f v_h\,dx
- \int_\Gamma (\nabla v_h\cdot n_\Gamma)g\,ds
+ \int_\Gamma \frac{\gamma}{h}g v_h\,ds.
$$

The normal is computed from the level set,

$$
n_\Gamma = \frac{\nabla \phi}{\|\nabla \phi\|},
$$

and evaluated at the runtime boundary quadrature points.
```

```python
V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
h = ufl.CellDiameter(msh)

u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
f = 2.0 * np.pi**2 * u_exact
n_gamma = cutfemx.normal(phi)

dx_omega = ufl.Measure(
    "dx",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[inside_cells, dx_omega_rules],
)
dx_gamma = ufl.Measure(
    "dx",
    domain=msh,
    subdomain_id=1,
    subdomain_data=dx_gamma_rules,
)

gamma = 40.0

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
a += (
    -ufl.dot(ufl.grad(u), n_gamma) * v
    - ufl.dot(ufl.grad(v), n_gamma) * u
    + gamma / h * u * v
) * dx_gamma

L = f * v * dx_omega
L += (
    -ufl.dot(ufl.grad(v), n_gamma) * u_exact
    + gamma / h * u_exact * v
) * dx_gamma
```

## Stabilization

Small cut cells can degrade the conditioning of the discrete system. A standard
CutFEM remedy is to add a ghost-penalty term on interior facets near the
interface:

```{admonition} Ghost penalty
:class: formulation

For continuous first-order elements, a common stabilization is

$$
g_h(u_h,v_h)
=
\sum_{F \in {\mathcal F}_h^\Gamma}
\gamma_g h_F
\int_F
\left[\!\left[ \nabla u_h \right]\!\right]_n
\left[\!\left[ \nabla v_h \right]\!\right]_n\,ds,
$$

where ${\mathcal F}_h^\Gamma$ is a band of interior facets adjacent to cut
cells and

$$
\left[\!\left[ \nabla u_h \right]\!\right]_n
=
(\nabla u_h^+ - \nabla u_h^-)\cdot n_F.
$$

The complete stabilized form is

$$
a_h^{\mathrm{stab}}(u_h,v_h)
=
a_h(u_h,v_h) + g_h(u_h,v_h).
$$
```

The minimal code pattern is:

```python
ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
dS_ghost = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_id=2,
    subdomain_data=ghost_facets,
)

n = ufl.FacetNormal(msh)
h_avg = ufl.avg(h)
gamma_g = 0.1

a += (
    gamma_g
    * h_avg
    * ufl.inner(
        ufl.jump(ufl.grad(u), n),
        ufl.jump(ufl.grad(v), n),
    )
    * dS_ghost
)
```

## Assembly

Compile with `cutfemx.fem.form(...)`, then assemble through CutFEMx. Inactive
background degrees of freedom can be deactivated using the active-domain
information derived from the form.

```{admonition} Inactive degrees of freedom
:class: theory

The finite element space is defined on the background mesh, but only basis
functions with support on the active domain are part of the physical problem:

$$
{\mathcal I}_\Omega
=
\{i : \operatorname{supp}(\varphi_i)\cap\Omega \ne \emptyset\}.
$$

Degrees of freedom outside ${\mathcal I}_\Omega$ are constrained by replacing
their rows with identity equations,

$$
A_{ii}=1,\qquad A_{ij}=0\ (j\ne i),\qquad b_i=0.
$$
```

```python
a_form = cutfemx.fem.form(a)
L_form = cutfemx.fem.form(L)

b = cutfemx.fem.assemble_vector(L_form)
A = cutfemx.fem.assemble_matrix(a_form)

active_domain = cutfemx.fem.active_domain(a_form)
cutfemx.fem.deactivate_outside(A, b, active_domain)
```

The full runnable version, including linear solvers, error computation, ghost
penalty stabilization, and plotting, is in `python/demo/demo_poisson.py`.
