# Boundary Conditions

Boundary conditions in CutFEMx follow the same distinction as the geometry:
conditions on fitted mesh facets can be imposed with the standard DOLFINx
machinery, while conditions on an embedded boundary must be imposed through
terms in the variational form. This page focuses on Dirichlet data on a
level-set boundary.

```{admonition} Model problem
:class: formulation

Let the physical domain be represented by a level-set function,

$$
\Omega=\{x\in\Omega_0:\phi(x)<0\},
\qquad
\Gamma=\{x\in\Omega_0:\phi(x)=0\}.
$$

For the scalar Poisson problem

$$
-\Delta u=f\quad\text{in }\Omega,
\qquad
u=g\quad\text{on }\Gamma,
$$

the boundary $\Gamma$ is not a union of facets of the background mesh
${\mathcal T}_h$. Consequently there is, in general, no set of boundary degrees
of freedom on which one can impose $u=g$ strongly. The boundary condition is
instead enforced weakly by adding consistent and coercive boundary terms on
the embedded interface.
```

## From Integration By Parts To Nitsche Terms

Assume first that $u$ is the exact solution and $v$ is a test function. By
integration by parts on the physical domain,

$$
\int_\Omega \nabla u\cdot\nabla v\,dx
-\int_\Gamma (\nabla u\cdot n_\Gamma)v\,ds
=
\int_\Omega f v\,dx .
$$

The boundary flux term is therefore required for consistency. If we simply
substitute $u=g$ on $\Gamma$, however, the resulting method would not be
symmetric and would not provide sufficient control of the trace of the discrete
solution on a cut boundary. The symmetric Nitsche method adds the adjoint
consistency term and a penalty term.

```{admonition} Symmetric Nitsche formulation
:class: theory

Find $u_h\in V_h$ such that

$$
a_h(u_h,v_h)=L_h(v_h)
\qquad\forall v_h\in V_h,
$$

with

$$
\begin{aligned}
a_h(u_h,v_h)
&=
\int_\Omega \nabla u_h\cdot\nabla v_h\,dx
-\int_\Gamma(\nabla u_h\cdot n_\Gamma)v_h\,ds\\
&\quad
-\int_\Gamma(\nabla v_h\cdot n_\Gamma)u_h\,ds
+\int_\Gamma\frac{\gamma}{h}u_hv_h\,ds,
\end{aligned}
$$

and

$$
L_h(v_h)
=
\int_\Omega f v_h\,dx
-\int_\Gamma(\nabla v_h\cdot n_\Gamma)g\,ds
+\int_\Gamma\frac{\gamma}{h}g v_h\,ds.
$$

The first boundary term is the physical flux term obtained from integration by
parts. The second boundary term is the adjoint consistency term. The penalty
term controls the boundary mismatch $u_h-g$ in a mesh-dependent norm.
```

## Normal Orientation

The normal on an embedded boundary is computed from the level set. For the
negative phase convention used here,

$$
n_\Gamma = \frac{\nabla\phi}{|\nabla\phi|}.
$$

In CutFEMx this is represented by a lazy quadrature function:

```python
n_gamma = cutfemx.normal(phi)
```

If the opposite orientation is required, use

```python
n_gamma = cutfemx.normal(phi, sign=-1.0)
```

The sign matters in the flux terms. For symmetric Dirichlet Nitsche terms, a
consistent reversal of the normal changes both flux terms and leaves the
method equivalent when the same convention is used throughout.

## Penalty Scaling

The penalty parameter has the form $\gamma/h$. Here $h$ is a local cell size
and $\gamma$ is a dimensionless number chosen large enough to recover
coercivity of the discrete bilinear form.

```{admonition} Why the penalty is scaled by cell size
:class: theory

The trace inequality on a cut cell has the same dimensional structure as on a
fitted cell:

$$
\|v_h\|_{L^2(\Gamma\cap K)}^2
\lesssim
h_K^{-1}\|v_h\|_{L^2(K\cap\Omega)}^2
+h_K\|\nabla v_h\|_{L^2(K\cap\Omega)}^2 .
$$

The factor $h_K^{-1}$ in the boundary penalty balances the volume energy
$\|\nabla v_h\|_{L^2(\Omega)}^2$. In practice, small cut cells also require
ghost-penalty stabilization so that this estimate is robust with respect to
how $\Gamma$ intersects the background mesh.
```

In UFL, the local size is usually represented by

```python
h = ufl.CellDiameter(msh)
gamma = 40.0
```

The appropriate value of `gamma` depends on the polynomial degree, the
stabilization, and the geometry. The examples use conservative values rather
than optimized constants.

## Runtime Measures

The weak boundary terms must be integrated over two different geometric sets:
the physical volume $\Omega$ and the embedded boundary $\Gamma$. Both are
obtained from the same cut data:

```python
cut_data = cutfemx.cut(phi)

inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
omega_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order=4)
gamma_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order=4)
```

The volume measure combines standard cells completely inside the domain with
runtime quadrature on cut cells. The boundary measure contains only the
embedded interface quadrature:

```python
dx_omega = ufl.Measure(
    "dx",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[inside_cells, omega_rules],
)

dx_gamma = ufl.Measure(
    "dx",
    domain=msh,
    subdomain_id=1,
    subdomain_data=gamma_rules,
)
```

Although `dx_gamma` is written as a UFL `"dx"` measure, its quadrature provider
represents the lower-dimensional embedded boundary inside the cut cells. The
runtime quadrature data carries the physical points and weights for
$\Gamma\cap K$.

## UFL Form

With the measures and normal in place, the mathematical form translates
directly to UFL:

```python
n_gamma = cutfemx.normal(phi)
h = ufl.CellDiameter(msh)
gamma = 40.0

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_omega
a += (
    -ufl.dot(ufl.grad(u), n_gamma) * v
    - ufl.dot(ufl.grad(v), n_gamma) * u
    + gamma / h * u * v
) * dx_gamma

L = f * v * dx_omega
L += (
    -ufl.dot(ufl.grad(v), n_gamma) * g
    + gamma / h * g * v
) * dx_gamma
```

The same pattern also appears in vector-valued problems. For example, in the
Stokes demo the no-slip condition $u=0$ on a cut obstacle is imposed by
replacing the scalar flux $\nabla u\cdot n_\Gamma$ with the stress traction
$\sigma(u,p)n_\Gamma$.

## Assembly And Updates

Compile and assemble the form with CutFEMx:

```python
a_form = cutfemx.fem.form(a)
L_form = cutfemx.fem.form(L)
```

The active-domain information is derived from the compiled form and should be
used to deactivate degrees of freedom whose basis functions have no support in
the physical domain.

For moving boundaries, the geometric data must be refreshed before
reassembling:

```python
phi.interpolate(new_level_set_expression)
cut_data.update()

inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
omega_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order=4)
gamma_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order=4)
```

The old entity arrays and quadrature providers describe the old zero contour.
After `cut_data.update()`, rebuild all measures that depend on the cut
geometry.
