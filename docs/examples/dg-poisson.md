# Cut DG Poisson Problem

This example follows `python/demo/demo_dg_poisson.py`. It solves a manufactured
Poisson problem on an implicitly defined disk using a discontinuous Galerkin
space. Compared with the continuous Galerkin example, the essential new feature
is the active skeleton: DG methods contain facet terms, and those facets must
also be restricted to the physical domain.

```{admonition} Model problem
:class: formulation

Let $\Omega_0=[-1,1]^2$ be the background domain and let

$$
\Omega=\{x\in\Omega_0:\phi(x)<0\},
\qquad
\Gamma=\{x\in\Omega_0:\phi(x)=0\}.
$$

The demo solves

$$
-\Delta u=f\quad\text{in }\Omega,
\qquad
u=g\quad\text{on }\Gamma,
$$

with the manufactured solution

$$
u_{\mathrm{ex}}(x,y)=\sin(\pi x)\sin(\pi y),
\qquad
f=2\pi^2u_{\mathrm{ex}},
\qquad
g=u_{\mathrm{ex}}\vert_\Gamma .
$$
```

## DG Space And Active Mesh

The discrete space is a discontinuous polynomial space on the background mesh:

```python
V = fem.functionspace(msh, ("DG", args.degree))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
```

The active cell set is the same geometric object as in the continuous example:

$$
{\mathcal T}_h^\Omega
=
\{K\in{\mathcal T}_h:K\cap\Omega\ne\emptyset\}.
$$

The demo obtains it from the cell cut:

```python
cell_cut = cutfemx.cut(phi)
inside_cells = cutfemx.locate_entities(cell_cut, "phi<0")
cut_cells = cutfemx.locate_entities(cell_cut, "phi=0")
active_cells = cutfemx.locate_entities(cell_cut, "phi<=0")
```

For DG methods, active cells are not enough. The variational form also contains
interior-facet integrals. The active skeleton is

$$
{\mathcal F}_h^\Omega
=
\{F=K^+\cap K^-:
K^+\cap\Omega\ne\emptyset,\,
K^-\cap\Omega\ne\emptyset\}.
$$

CutFEMx first discovers those facets from the active cell set and then cuts
them as lower-dimensional entities:

```python
fdim = msh.topology.dim - 1
skeleton_facets = cutfemx.interior_facets_for_cells(msh, active_cells)
skeleton_cut = cutfemx.cut(phi, skeleton_facets, fdim)

inside_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi<0")
cut_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi=0")
```

```{admonition} Why the skeleton is cut
:class: theory

An interior facet can be intersected by the physical boundary. The SIPG terms
should then be integrated over $F\cap\Omega$, not over the whole background
facet $F$. This is a lower-dimensional analogue of cut-cell quadrature:

$$
\int_{\mathcal{F}_h^\Omega} H\,ds
=
\sum_{F\in{\mathcal F}_h^\Omega}
\int_{F\cap\Omega}H\,ds .
$$

The runtime interior-facet rules are what make this restriction visible to the
form assembler.
```

## Runtime Measures

The demo uses three cut measures:

```python
cell_rules = cutfemx.runtime_quadrature(cell_cut, "phi<0", args.order)
interface_rules = cutfemx.runtime_quadrature(cell_cut, "phi=0", args.order)
skeleton_rules = cutfemx.runtime_quadrature(skeleton_cut, "phi<0", args.order)
```

These rules are attached to ordinary UFL measures:

```python
dx_omega = ufl.Measure(
    "dx", domain=msh, subdomain_id=0, subdomain_data=[inside_cells, cell_rules]
)
dS_omega = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[inside_skeleton_facets, skeleton_rules],
)
dx_gamma = ufl.Measure(
    "dx", domain=msh, subdomain_id=1, subdomain_data=interface_rules
)
```

The roles are:

- `dx_omega`: volume integration over $\Omega$,
- `dS_omega`: active interior-skeleton integration over $F\cap\Omega$,
- `dx_gamma`: embedded-boundary integration over $\Gamma$.

The ghost penalty uses a standard DOLFINx interior-facet measure on a selected
facet band:

```python
ghost_facets = cutfemx.ghost_penalty_facets(cell_cut, "phi<0")
dS_ghost = ufl.Measure("dS", domain=msh, subdomain_id=2, subdomain_data=ghost_facets)
```

## SIPG Terms On The Active Skeleton

Let $n$ be the background facet normal. The demo writes the vector jumps as

```python
jump_u = ufl.jump(u, n)
jump_v = ufl.jump(v, n)
```

For scalar $u$, this corresponds to the normal jump

$$
\left[\!\left[u n\right]\!\right]
=
u^+n^+ + u^-n^- .
$$

The SIPG contribution on the active skeleton is

$$
\begin{aligned}
a_{\mathrm{SIPG}}(u_h,v_h)
&=
-\int_{\mathcal{F}_h^\Omega}
\{\!\{\nabla u_h\}\!\}\cdot
\left[\!\left[v_h n\right]\!\right]\,ds\\
&\quad
-\int_{\mathcal{F}_h^\Omega}
\{\!\{\nabla v_h\}\!\}\cdot
\left[\!\left[u_h n\right]\!\right]\,ds\\
&\quad
+\int_{\mathcal{F}_h^\Omega}
\frac{\sigma}{\{h\}}
\left[\!\left[u_h n\right]\!\right]\cdot
\left[\!\left[v_h n\right]\!\right]\,ds .
\end{aligned}
$$

The first term is consistency, the second is adjoint consistency, and the third
penalizes discontinuities across active facets. In the demo:

```python
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
a += -ufl.inner(ufl.avg(ufl.grad(u)), jump_v) * dS_omega
a += -ufl.inner(ufl.avg(ufl.grad(v)), jump_u) * dS_omega
a += sigma / h_avg * ufl.inner(jump_u, jump_v) * dS_omega
```

The penalty is scaled by the polynomial degree:

```python
sigma = args.penalty * args.degree**2
```

This is the usual SIPG scaling: higher-order polynomials require stronger
control of facet traces.

## Embedded Boundary Terms

The physical Dirichlet boundary is not a mesh boundary. It is the zero contour
of the level set. The demo therefore adds symmetric Nitsche terms on
$\Gamma$:

$$
\begin{aligned}
a_\Gamma(u_h,v_h)
&=
-\int_\Gamma(\nabla u_h\cdot n_\Gamma)v_h\,ds
-\int_\Gamma(\nabla v_h\cdot n_\Gamma)u_h\,ds\\
&\quad
+\int_\Gamma\frac{\sigma_\Gamma}{h}u_hv_h\,ds ,
\end{aligned}
$$

and the corresponding right-hand side

$$
L_\Gamma(v_h)
=
-\int_\Gamma(\nabla v_h\cdot n_\Gamma)g\,ds
+\int_\Gamma\frac{\sigma_\Gamma}{h}g v_h\,ds .
$$

The code mirrors these terms:

```python
n_gamma = cutfemx.normal(phi)
sigma_gamma = args.interface_penalty * args.degree**2

a += (
    -ufl.dot(ufl.grad(u), n_gamma) * v
    - ufl.dot(ufl.grad(v), n_gamma) * u
    + sigma_gamma / h * u * v
) * dx_gamma

L = f * v * dx_omega
L += (
    -ufl.dot(ufl.grad(v), n_gamma) * u_exact
    + sigma_gamma / h * u_exact * v
) * dx_gamma
```

```{admonition} Two different penalties
:class: theory

The SIPG penalty $\sigma$ controls jumps between neighbouring DG cells on the
active skeleton. The Nitsche penalty $\sigma_\Gamma$ controls the mismatch
$u_h-g$ on the embedded boundary. They appear similar because both are
trace-control terms, but they act on different geometric sets and enforce
different properties.
```

## Ghost Penalty

The SIPG terms control jumps across active facets, but they do not fully remove
the small-cut-cell conditioning problem. If a cell contributes only a tiny
portion $K\cap\Omega$, the corresponding basis functions can remain poorly
controlled. The demo therefore adds a gradient-jump ghost penalty on facets
around the cut cells:

$$
g_h(u_h,v_h)
=
\sum_{F\in{\mathcal F}_h^\Gamma}
\gamma_g h_F
\int_F
\left[\!\left[\nabla u_h\right]\!\right]_n
\left[\!\left[\nabla v_h\right]\!\right]_n\,ds .
$$

In code:

```python
if ghost_facets.size > 0 and args.ghost_penalty != 0.0:
    a += (
        args.ghost_penalty
        * h_avg
        * ufl.inner(
            ufl.jump(ufl.grad(u), n),
            ufl.jump(ufl.grad(v), n),
        )
        * dS_ghost
    )
```

## Assembly And Convergence Check

The demo assembles the runtime form, deactivates inactive degrees of freedom,
and solves a serial MatrixCSR system with SciPy:

```python
a_form = cutfemx.fem.form(a)
L_form = cutfemx.fem.form(L)
uh, A, b, active_domain = solve_runtime_system_scipy(a_form, L_form, V)
```

The error is integrated over the same cut volume measure:

$$
\|u_h-u_{\mathrm{ex}}\|_{L^2(\Omega)}^2
=
\int_\Omega (u_h-u_{\mathrm{ex}})^2\,dx .
$$

```python
error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_omega)
error_sq = cutfemx.fem.assemble_scalar(error_form)
```

This is an important point: the error norm is also a cut-domain integral, so it
must use the same runtime volume measure as the PDE.

Run the complete example with:

```bash
python python/demo/demo_dg_poisson.py
```
