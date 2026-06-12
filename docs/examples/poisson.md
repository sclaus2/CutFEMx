# Cut Poisson Problem

This example follows `python/demo/demo_poisson.py`. It illustrates the basic
CutFEM construction for a scalar elliptic problem posed on a domain that is not
resolved by the computational mesh. The physical boundary is represented
implicitly by a level-set function, while the finite element space remains
defined on a fixed background mesh.

```{admonition} Problem statement
:class: formulation

Let $\Omega_0 = [-1,1]^2$ be the background domain. With
$\rho=\sqrt{x^2+y^2}$ and $\theta=\operatorname{atan2}(y,x)$, define the
six-petal flower level set

$$
r_\Gamma(\theta)=R_0+A\cos(m\theta),
\qquad
\phi(x,y)=\rho-r_\Gamma(\theta),
$$

where $R_0=0.46$, $A=0.15$, and $m=6$.

The physical domain and embedded boundary are

$$
\Omega = \{(x,y)\in\Omega_0:\phi(x,y)<0\},
\qquad
\Gamma = \{(x,y)\in\Omega_0:\phi(x,y)=0\}.
$$

We consider the manufactured Dirichlet problem

$$
-\Delta u = f \quad \text{in }\Omega,
\qquad
u = g \quad \text{on }\Gamma,
$$

with

$$
u_{\mathrm{ex}}(x,y)=\sin(\pi x)\sin(\pi y),
\qquad
f=2\pi^2 u_{\mathrm{ex}},
\qquad
g=u_{\mathrm{ex}}\vert_\Gamma .
$$
```

## Geometry And Active Mesh

Let $\mathcal T_h$ be a triangulation of the background domain
$\Omega_0$. The computational mesh does not conform to $\Gamma$; instead,
CutFEMx classifies mesh entities according to the sign of the level-set field.
The level-set field is represented as an ordinary finite element function. The
demo uses a linear level-set field on triangles, matching the currently
reliable quadrature path.

```python
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    points=((-1.0, -1.0), (1.0, 1.0)),
    n=(21, 21),
    cell_type=mesh.CellType.triangle,
)

V_phi = fem.functionspace(msh, ("Lagrange", 1))
phi = fem.Function(V_phi, name="phi")

def flower_level_set(x):
    theta = np.arctan2(x[1], x[0])
    boundary_radius = 0.46 + 0.15 * np.cos(6 * theta)
    return np.sqrt(x[0] ** 2 + x[1] ** 2) - boundary_radius

phi.interpolate(flower_level_set)
```

The call to `cutfemx.cut` constructs the geometric information required to
identify the active cells, the cut cells, and the embedded interface:

```python
cut_data = cutfemx.cut(phi)

inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
interface_cells = cutfemx.locate_entities(cut_data, "phi=0")
active_cells = cutfemx.locate_entities(cut_data, "phi<=0")
```

Here the selector `"phi<0"` identifies cells that contain a non-empty part of
the physical domain, while `"phi=0"` identifies cells intersected by the
embedded boundary. In the demo, a cut mesh is also created for visualization:

```python
cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi < 0", mode="full")
phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)
```

The cut mesh is not the mesh on which the linear system is assembled. It is a
geometric representation of $\Omega_h$ used to inspect the level set,
quadrature points, and computed solution.

```{admonition} Why three cell sets appear
:class: theory

The discrete problem is posed on the active mesh, not on the whole background
mesh. In geometric terms,

$$
{\mathcal T}_h^\Omega =
\{K\in{\mathcal T}_h:K\cap\Omega\ne\emptyset\}.
$$

Cells for which $K\subset\Omega$ can be integrated with standard quadrature.
Cells intersected by $\Gamma$ require cut quadrature. Cells outside
$\Omega$ remain part of the background mesh data structure, but their degrees
of freedom are algebraically constrained after assembly.
```

The active mesh viewpoint is central to CutFEM. The finite element basis is
still the background basis, but only the active part of each basis function is
physically meaningful:

$$
V_h^\Omega
=
\{v_h\vert_{\Omega}:v_h\in V_h(\Omega_0)\}.
$$

This is why the code can use an ordinary DOLFINx function space:

```python
V = fem.functionspace(msh, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
```

The restriction to $\Omega$ is carried by the measures and by the active-domain
constraint, not by constructing a fitted finite element space.

## Cut Quadrature

On a fitted mesh, the measure in the variational form is independent of the
solution geometry. Here the relevant integration domains are $K\cap\Omega$ and
$K\cap\Gamma$, and therefore the quadrature rule is local to the cut
configuration of each cell.

```python
order = 4
dx_omega_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
dx_gamma_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
```

These rules are then attached to UFL measures. The form itself keeps the usual
finite element notation; the non-standard geometry enters through the measure:

```python
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
```

The measure named `dx_omega` represents volume integration over the flower
domain.
The measure named `dx_gamma` is written as a UFL `"dx"` measure because the
interface quadrature is hosted by cut cells, but geometrically it represents
integration over the curve $\Gamma$.

```{admonition} Quadrature on the cut domain
:class: formulation

For a volume integrand $F$, the numerical integration has the form

$$
\int_\Omega F\,dx
\approx
\sum_{K\in{\mathcal T}_h^\Omega}
\sum_{q=1}^{N_K} w_{Kq}F(x_{Kq}).
$$

For a boundary integrand $G$, integration over the zero level set is replaced by

$$
\int_\Gamma G\,ds
\approx
\sum_{K\in{\mathcal T}_h^\Gamma}
\sum_{q=1}^{M_K}\omega_{Kq}G(\xi_{Kq}).
$$

Thus the mathematical form remains recognizable as a standard weak form, while
the measure stores the geometry-dependent quadrature points and weights.
```

For this example, the two quadrature providers are reused in three places:

- the stiffness and load integrals over $\Omega$,
- the Nitsche terms over $\Gamma$,
- the error norm over $\Omega$.

Using the same runtime measure for the PDE and for the diagnostic error is
important. Otherwise the reported convergence would not measure the same
domain on which the equation was solved.

## Weak Imposition Of The Boundary Condition

Since $\Gamma$ is not composed of mesh facets, a strong imposition of
$u=g$ is not available through the usual boundary-degree-of-freedom machinery.
The Dirichlet condition is therefore imposed weakly by a symmetric Nitsche
method. The normal is obtained from the level set,
$n_\Gamma=\nabla\phi/|\nabla\phi|$, evaluated at the quadrature points on
$\Gamma$:

```python
n_gamma = cutfemx.normal(phi)
h = ufl.CellDiameter(msh)
gamma = 40.0
```

```{admonition} Symmetric Nitsche form
:class: formulation

For test and trial functions in the background finite element space restricted
to the active mesh, the discrete bilinear form reads

$$
\begin{aligned}
a_h(u_h,v_h)
&=
\int_\Omega \nabla u_h\cdot\nabla v_h\,dx
-\int_\Gamma (\nabla u_h\cdot n_\Gamma)v_h\,ds \\
&\quad
-\int_\Gamma(\nabla v_h\cdot n_\Gamma)u_h\,ds
+\int_\Gamma \frac{\gamma}{h}u_hv_h\,ds
+g_h(u_h,v_h).
\end{aligned}
$$

and the corresponding linear form is

$$
L_h(v_h)
=
\int_\Omega fv_h\,dx
-\int_\Gamma(\nabla v_h\cdot n_\Gamma)g\,ds
+\int_\Gamma\frac{\gamma}{h}gv_h\,ds.
$$

The consistency terms are obtained by integrating the diffusion operator by
parts on the physical domain. The penalty term provides coercive control of the
trace of $u_h-g$ on the embedded boundary; the parameter $\gamma$ must be
chosen large enough with respect to the polynomial order and the cut geometry.
```

The UFL expression follows this variational form directly:

```python
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

The sign of the normal is determined by the level set. For this flower level
set, the same expression gives the outward normal of the negative phase:

$$
n_\Gamma = \frac{\nabla\phi}{|\nabla\phi|}.
$$

This is the outward normal of $\Omega=\{\phi<0\}$, so the signs in the Nitsche
terms match the integration-by-parts identity

$$
\int_\Omega -\Delta u\,v\,dx
=
\int_\Omega \nabla u\cdot\nabla v\,dx
-\int_\Gamma (\nabla u\cdot n_\Gamma)v\,ds .
$$

## Ghost Penalty

Small intersections $K\cap\Omega$ can lead to poorly conditioned systems. A
ghost penalty extends control from well-resolved cells to their cut-cell
neighbours by penalizing jumps of normal derivatives across an interior facet
band $\mathcal F_h^\Gamma$:

$$
g_h(u_h,v_h)
=
\sum_{F\in{\mathcal F}_h^\Gamma}
\gamma_g h_F
\int_F
\left[\!\left[\nabla u_h\right]\!\right]_n
\left[\!\left[\nabla v_h\right]\!\right]_n\,ds .
$$

The demo obtains the facet band from the same cut data:

```python
ghost_facets = cutfemx.ghost_penalty_facets(cut_data, "phi<0")
dS_ghost = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_id=2,
    subdomain_data=ghost_facets,
)
```

and adds the stabilization only if the band is non-empty:

```python
if ghost_facets.size > 0:
    a_cut += (
        gamma_g
        * h_avg
        * ufl.inner(
            ufl.jump(ufl.grad(u), n_facet),
            ufl.jump(ufl.grad(v), n_facet),
        )
        * dS_ghost
    )
```

```{admonition} What the ghost penalty controls
:class: theory

The Nitsche penalty controls the trace of $u_h-g$ on $\Gamma$. It does not by
itself control basis functions that live mostly outside $\Omega$ but have a
small intersection with the active domain. The ghost penalty transfers
$H^1$-control across a narrow band of interior facets adjacent to the cut
boundary. This restores a mesh-size robust algebraic problem while preserving
the consistency of the method, because the exact smooth solution has small
normal-derivative jumps across background facets.
```

## Algebraic Constraint And Solve

After compilation and assembly, degrees of freedom whose basis functions have
no support on the active domain are constrained. This gives a linear system for
the physical unknowns while keeping the background finite element space intact:

```python
a_form = cutfemx.fem.form(a)
L_form = cutfemx.fem.form(L)

b = cutfemx.fem.assemble_vector(L_form)
A = cutfemx.fem.assemble_matrix(a_form)

active_domain = cutfemx.fem.active_domain(a_form)
cutfemx.fem.deactivate_outside(A, b, active_domain)
```

In mathematical terms, the background vector is decomposed into active and
inactive degrees of freedom. The inactive block corresponds to basis functions
whose support does not intersect the physical domain. These unknowns are not
part of the PDE, so CutFEMx replaces their equations by a benign algebraic
constraint. This avoids singular rows while keeping the global background
numbering intact.

The demo can solve the runtime system with either SciPy or PETSc through its
`solve_runtime_system` helper. The important documentation point is that the
form is compiled as a CutFEMx form before assembly:

```python
a_cut = cutfemx.fem.form(a_cut)
L_cut = cutfemx.fem.form(L_cut)
uh = solve_runtime_system(a_cut, L_cut, V, solver)
```

## Error And Visualization

The $L^2$ error is evaluated on the physical flower domain:

$$
\|u_h-u_{\mathrm{ex}}\|_{L^2(\Omega)}^2
=
\int_\Omega (u_h-u_{\mathrm{ex}})^2\,dx .
$$

The code uses the same cut volume measure as the variational form:

```python
error_form = cutfemx.fem.form((uh - u_exact) ** 2 * dx_omega)
error_sq = cutfemx.fem.assemble_scalar(error_form)
error = np.sqrt(comm.allreduce(error_sq, op=MPI.SUM))
```

For visualization, the background solution is transferred to the cut mesh:

```python
u_cut = cutfemx.fem.cut_function(uh, cut_mesh)
```

This separates computation from presentation: assembly uses runtime
quadrature on the background mesh, while plots use an explicit cut mesh that
shows the geometry seen by the method.

Run the complete example with:

```bash
python python/demo/demo_poisson.py
```
