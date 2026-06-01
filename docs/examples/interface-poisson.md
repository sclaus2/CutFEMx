# Two-Domain Poisson Interface Problem

This example follows `python/demo/demo_interface_poisson.py`. It is the first
example in which the unfitted boundary is not an exterior boundary of the
physical domain, but an internal material interface. The level set partitions
the background mesh into two phases, each with its own diffusion coefficient.
The unknown is represented by two background finite element fields, one per
phase, and the two fields communicate only through the interface terms.

```{admonition} Interface model
:class: formulation

Let

$$
\Omega_1=\{x:\phi(x)<0\},
\qquad
\Omega_2=\{x:\phi(x)>0\},
\qquad
\Gamma=\{x:\phi(x)=0\}.
$$

The model problem is

$$
-\nabla\cdot(\kappa_i\nabla u_i)=f_i
\quad\text{in }\Omega_i,\qquad i=1,2,
$$

with transmission conditions

$$
u_1-u_2=0,
\qquad
\kappa_1\nabla u_1\cdot n_\Gamma
-\kappa_2\nabla u_2\cdot n_\Gamma=0
\quad\text{on }\Gamma .
$$

The demo uses a manufactured pair of quadratic solutions chosen so that the
transmission conditions hold for different diffusivities $\kappa_1$ and
$\kappa_2$. This gives a controlled test of both the phase quadrature and the
Nitsche coupling across $\Gamma$.
```

## Geometry And Phase Spaces

The background mesh is the square $\Omega_0=[-1,1]^2$. A circular level set
defines the material interface:

```python
msh = mesh.create_rectangle(
    comm,
    ((-1.0, -1.0), (1.0, 1.0)),
    (n, n),
    cell_type=mesh.CellType.triangle,
)

V_phi = fem.functionspace(msh, ("Lagrange", 1))
phi = fem.Function(V_phi, name="phi")
phi.interpolate(
    lambda x: np.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2)
    - radius
)
```

The two subdomains are selected by the sign of $\phi$. The field $u_1$ is
attached to the negative phase and $u_2$ to the positive phase. The background
finite element spaces are ordinary continuous Lagrange spaces:

```python
V1 = fem.functionspace(msh, ("Lagrange", 1))
V2 = fem.functionspace(msh, ("Lagrange", 1))
W_block = ufl.MixedFunctionSpace(V1, V2)
u1, u2 = ufl.TrialFunctions(W_block)
v1, v2 = ufl.TestFunctions(W_block)
```

```{admonition} Why two fields are used
:class: theory

For an interface problem the solution is generally continuous, but its normal
derivative is not. The flux condition is

$$
\kappa_1 \nabla u_1\cdot n_\Gamma
=\kappa_2 \nabla u_2\cdot n_\Gamma ,
$$

where $n_\Gamma$ points from $\Omega_1$ to $\Omega_2$. A single globally
continuous finite element field would implicitly tie the traces on the two
sides together and would make it harder to express phase-dependent fluxes. By
using two background fields, CutFEMx keeps the two traces independent and then
imposes the physical transmission law weakly on $\Gamma$.
```

## Phase Quadrature

The same `CutData` object provides quadrature for both phases and for the
interface:

```python
cut_data = cutfemx.cut(phi)

inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
outside_cells = cutfemx.locate_entities(cut_data, "phi>0")

inside_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
outside_rules = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
```

The measures are

```python
dx1 = ufl.Measure("dx", domain=msh, subdomain_id=1, subdomain_data=[inside_cells, inside_rules])
dx2 = ufl.Measure("dx", domain=msh, subdomain_id=2, subdomain_data=[outside_cells, outside_rules])
dgamma = ufl.Measure("dx", domain=msh, subdomain_id=3, subdomain_data=interface_rules)
```

Mathematically these measures represent

$$
\int_{\Omega_1}(\cdot)\,dx,\qquad
\int_{\Omega_2}(\cdot)\,dx,\qquad
\int_\Gamma(\cdot)\,ds .
$$

The UFL measure type is still `"dx"` for the interface because the runtime
quadrature points are hosted by the cut cells. The geometric dimension is
encoded in the quadrature provider, not in the string name of the UFL measure.

## Manufactured Solution

The demo chooses

$$
u_1(x)=r^2(x),\qquad
u_2(x)=\frac{\kappa_1}{\kappa_2}r^2(x)
+R^2\left(1-\frac{\kappa_1}{\kappa_2}\right),
$$

where

$$
r^2(x)=(x_0-c_0)^2+(x_1-c_1)^2 .
$$

On the circle $r=R$, the two traces agree:

$$
u_1=R^2=u_2 .
$$

Moreover,

$$
\kappa_1\nabla u_1\cdot n_\Gamma
=
\kappa_1 2R,
\qquad
\kappa_2\nabla u_2\cdot n_\Gamma
=
\kappa_2\frac{\kappa_1}{\kappa_2}2R
=
\kappa_1 2R .
$$

The manufactured pair therefore satisfies the interface conditions exactly.
The corresponding right-hand sides are constants:

```python
f1 = fem.Constant(msh, default_scalar_type(-4.0 * kappa_1))
f2 = fem.Constant(msh, default_scalar_type(-4.0 * kappa_1))
```

## Bulk Terms

Away from the interface, the weak form is the standard diffusion form in each
phase:

$$
\sum_{i=1}^2
\int_{\Omega_i}
\kappa_i\nabla u_i\cdot\nabla v_i\,dx .
$$

In code:

```python
a = kappa_1 * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1
a += kappa_2 * ufl.inner(ufl.grad(u2), ufl.grad(v2)) * dx2

L = f1 * v1 * dx1 + f2 * v2 * dx2
```

The important CutFEM point is that neither $V_1$ nor $V_2$ is defined on a
fitted submesh. Both spaces live on the background mesh, and the measures
select the phase where each field is physically active.

## Nitsche Coupling

The interface is coupled by a weighted symmetric Nitsche form. With

$$
[u]=u_1-u_2,
\qquad
\{\kappa\nabla u\cdot n\}_w
=
w_1\kappa_1\nabla u_1\cdot n_\Gamma
+w_2\kappa_2\nabla u_2\cdot n_\Gamma,
$$

the interface contribution is

$$
\int_\Gamma
\left(
-\{\kappa\nabla u\cdot n\}_w[v]
-\{\kappa\nabla v\cdot n\}_w[u]
+\eta_\Gamma [u][v]
\right)\,ds.
$$

The demo uses harmonic scaling

$$
\kappa_h=\frac{2\kappa_1\kappa_2}{\kappa_1+\kappa_2},
\qquad
\eta_\Gamma=\gamma_\Gamma\frac{\kappa_h}{h},
$$

with weights $w_1=\kappa_2/(\kappa_1+\kappa_2)$ and
$w_2=\kappa_1/(\kappa_1+\kappa_2)$.

The code makes the jump and weighted flux explicit:

```python
kappa_h = 2.0 * kappa_1 * kappa_2 / (kappa_1 + kappa_2)
eta_interface = gamma_interface * kappa_h / h
w1 = kappa_2 / (kappa_1 + kappa_2)
w2 = kappa_1 / (kappa_1 + kappa_2)

def jump_pair(side1, side2):
    return side1 - side2

def avg_flux_pair(side1, side2):
    return w1 * kappa_1 * ufl.dot(ufl.grad(side1), n_gamma) + w2 * kappa_2 * ufl.dot(
        ufl.grad(side2), n_gamma
    )

jump_u = jump_pair(u1, u2)
jump_v = jump_pair(v1, v2)
flux_u = avg_flux_pair(u1, u2)
flux_v = avg_flux_pair(v1, v2)

a += (-flux_u * jump_v - flux_v * jump_u + eta_interface * jump_u * jump_v) * dgamma
```

```{admonition} Consistency of the interface terms
:class: formulation

If the exact solution satisfies $[u]=0$ and flux continuity, the interface
contribution vanishes in the residual after integration by parts in each
phase. The first term supplies the physical flux, the second is the symmetric
adjoint-consistency term, and the final term penalizes the trace mismatch
$[u_h]$. The penalty is scaled with the harmonic diffusivity so that the
interface condition remains balanced when one phase is much more conductive
than the other.
```

```{admonition} Block structure
:class: theory

The two phase fields live on different background spaces, but their interface
terms produce off-diagonal matrix blocks. The demo extracts UFL blocks,
assembles each block with CutFEMx, deactivates inactive degrees of freedom in
both phase spaces, and solves the resulting serial block system.
```

## Outer Boundary Condition

The exterior boundary belongs only to the outside phase in this example. The
script imposes the manufactured value of $u_2$ weakly on $\partial\Omega_0$:

$$
\begin{aligned}
a_{\partial\Omega_0}(u_2,v_2)
&=
-\int_{\partial\Omega_0}\kappa_2(\nabla u_2\cdot n)v_2\,ds
-\int_{\partial\Omega_0}\kappa_2(\nabla v_2\cdot n)u_2\,ds\\
&\quad
+\int_{\partial\Omega_0}\eta_{\partial\Omega}u_2v_2\,ds,
\end{aligned}
$$

with

$$
\eta_{\partial\Omega}=\gamma_{\partial\Omega}\frac{\kappa_2}{h}.
$$

In UFL:

```python
eta_boundary = gamma_boundary * kappa_2 / h

a += (
    -kappa_2 * ufl.dot(ufl.grad(u2), n_facet) * v2
    - kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2
    + eta_boundary * u2 * v2
) * ds_outer

L += (
    -kappa_2 * ufl.dot(ufl.grad(v2), n_facet) * u2_exact
    + eta_boundary * u2_exact * v2
) * ds_outer
```

This keeps the example algebraically uniform: both the internal interface and
the outer Dirichlet boundary are handled by Nitsche terms.

## Ghost Penalties

Each phase also receives its own ghost penalty on the cut-cell band associated
with its selector:

```python
ghost_facets_1 = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
ghost_facets_2 = cutfemx.ghost_penalty_facets(cut_data, "phi>0")
```

The corresponding stabilization is

$$
g_i(u_i,v_i)
=
\sum_{F\in{\mathcal F}_{h,i}^{\Gamma}}
\gamma_g\kappa_i h_F
\int_F
\left[\!\left[\nabla u_i\right]\!\right]_n
\left[\!\left[\nabla v_i\right]\!\right]_n\,ds,
\qquad i=1,2 .
$$

The factor $\kappa_i$ gives the ghost penalty the same physical scaling as the
diffusion operator in phase $i$. This is important when the diffusivities are
not comparable.

## Assembly And Diagnostics

The mixed UFL form is split into scalar blocks before assembly:

```python
a_blocks = ufl.extract_blocks(a)
L_blocks = ufl.extract_blocks(L)
a_forms = tuple(tuple(cutfemx.fem.form(a_blocks[i][j]) for j in range(2)) for i in range(2))
L_forms = tuple(cutfemx.fem.form(L_blocks[i]) for i in range(2))
```

The active domain is computed separately for the two diagonal blocks:

```python
domain1 = cutfemx.fem.active_domain(a_forms[0][0])
domain2 = cutfemx.fem.active_domain(a_forms[1][1])
u1_h, u2_h = solve_block_system(a_forms, L_forms, domain1, domain2, V1, V2)
```

The demo reports the phase errors and the remaining jump on the interface:

$$
\|u_{1,h}-u_{1,\mathrm{ex}}\|_{L^2(\Omega_1)},\qquad
\|u_{2,h}-u_{2,\mathrm{ex}}\|_{L^2(\Omega_2)},\qquad
\|u_{1,h}-u_{2,h}\|_{L^2(\Gamma)} .
$$

These three quantities test different parts of the method: volume quadrature in
the two phases, interface quadrature on $\Gamma$, and the strength of the
transmission coupling.

Run the complete example with:

```bash
python python/demo/demo_interface_poisson.py
```
