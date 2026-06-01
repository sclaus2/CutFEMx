# Cut Stokes Flow

This example follows `python/demo/demo_stokes.py`. It solves a steady Stokes
problem in a channel with a circular obstacle represented by a level set. The
mesh is a rectangle; the circular obstacle is not meshed. Instead, the fluid
domain is selected by cut integration and the no-slip condition on the obstacle
is imposed weakly.

```{admonition} Stokes problem on an unfitted fluid domain
:class: formulation

Let

$$
\Omega_f=\{x\in\Omega_0:\phi(x)>0\},
\qquad
\Gamma_c=\{x\in\Omega_0:\phi(x)=0\},
$$

where $\Gamma_c$ is the boundary of the circular obstacle. The steady Stokes
problem is

$$
-\nabla\cdot\sigma(u,p)=f,
\qquad
\nabla\cdot u=0
\quad\text{in }\Omega_f,
$$

with stress tensor

$$
\sigma(u,p)=\nu(\nabla u+\nabla u^T)-pI.
$$

The demo prescribes an inflow profile and wall velocity on fitted channel
boundaries, fixes pressure on the outflow to remove the nullspace, and imposes
$u=0$ on the cut obstacle by Nitsche's method.
```

## Geometry And Fluid Selector

The background mesh is a channel:

```python
msh = mesh.create_rectangle(
    comm=comm,
    points=((-3.0, -1.0), (5.0, 1.0)),
    n=(4 * N, N),
    cell_type=mesh.CellType.triangle,
)
```

The level set is negative inside the obstacle and positive in the fluid:

```python
def circle(x):
    return np.sqrt(x[0]**2 + x[1]**2) - 0.3

level_set = fem.Function(Q)
level_set.interpolate(circle)
cut_data = cutfemx.cut(level_set)
```

The fluid is therefore selected with `"phi>0"`:

```python
active_entities = cutfemx.locate_entities(cut_data, "phi>0")
intersected_entities = cutfemx.locate_entities(cut_data, "phi=0")
```

The cut cells are removed from the standard-cell list so that they are not
integrated twice:

```python
uncut_cells = np.setdiff1d(active_entities, intersected_entities)
```

```{admonition} Sign convention
:class: theory

In the scalar Poisson examples, the physical domain is usually the negative
phase. In this Stokes example, the obstacle is the negative phase and the fluid
is the positive phase:

$$
\Omega_f=\{x:\phi(x)>0\}.
$$

This is not a change in CutFEMx semantics; it is a modeling choice. The
selector must always match the physical phase being solved.
```

## Runtime Measures

The fluid volume and the obstacle boundary are represented by two runtime
objects:

```python
fluid_quad = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
interface_quad = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
```

They are attached to measures:

```python
dxq = ufl.Measure(
    "dx",
    subdomain_id=0,
    subdomain_data=[uncut_cells, fluid_quad],
    domain=msh,
)
dsq = ufl.Measure(
    "dx",
    subdomain_id=1,
    subdomain_data=interface_quad,
    domain=msh,
)
```

Here `dxq` represents integration over $\Omega_f$. The measure named `dsq` is
also written as a UFL `"dx"` measure because the interface is cell-hosted
runtime quadrature; geometrically it represents integration over
$\Gamma_c$.

## Taylor-Hood Space

The demo uses a Taylor-Hood pair on the background mesh:

```python
P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(msh, TH)

w = ufl.TrialFunction(W)
(u, p) = ufl.split(w)
(v, q) = ufl.split(ufl.TestFunction(W))
```

Thus $u_h$ is continuous vector-valued $P_2$ and $p_h$ is continuous scalar
$P_1$. The mixed weak form is written on the background space but integrated
only over the cut fluid domain.

## Bulk Stokes Form

For test functions $(v,q)$, the bulk terms used in the demo are

$$
a_{\Omega_f}((u,p),(v,q))
=
\int_{\Omega_f}\sigma(u,p):\varepsilon(v)\,dx
+\int_{\Omega_f}(\nabla\cdot u)q\,dx,
$$

with

$$
\varepsilon(v)=\frac{1}{2}(\nabla v+\nabla v^T).
$$

The code defines the stress and bulk form as

```python
def sigma(u, p):
    return nu * (grad(u) + grad(u).T) - p * Identity(dim)

a = inner(sigma(u, p), sym(grad(v))) * dxq + div(u) * q * dxq
L = inner(f, v) * dxq
```

The body force in the demo is zero:

```python
f = Constant(msh, default_scalar_type((0.0, 0.0)))
```

## No-Slip Condition On The Cut Obstacle

The obstacle boundary is not a mesh boundary, so the no-slip condition
$u=0$ cannot be imposed by locating velocity degrees of freedom on
$\Gamma_c$. The demo uses a symmetric Nitsche form based on the traction
$\sigma(u,p)n_\Gamma$.

The level-set normal points from the negative phase to the positive phase. Since
the fluid is the positive phase and the obstacle is negative, the script uses
the opposite sign for the fluid-domain boundary normal:

```python
n_K = cutfemx.normal(level_set)
n_gamma = -n_K
```

The Nitsche contribution is

$$
\begin{aligned}
a_{\Gamma_c}((u,p),(v,q))
&=
-\int_{\Gamma_c}(\sigma(u,p)n_\Gamma)\cdot v\,ds
-\int_{\Gamma_c}(\sigma(v,q)n_\Gamma)\cdot u\,ds\\
&\quad
+\int_{\Gamma_c}\frac{\gamma_u}{h}u\cdot v\,ds .
\end{aligned}
$$

In code:

```python
a += -dot(dot(sigma(u, p), n_gamma), v) * dsq
a += -dot(dot(sigma(v, q), n_gamma), u) * dsq
a += (gamma_u / h) * inner(u, v) * dsq
```

```{admonition} Traction consistency
:class: theory

For Stokes flow, the natural boundary flux is not $\nabla u\cdot n$ but the
traction $\sigma(u,p)n$. Replacing the scalar diffusion flux by the traction is
what makes the Nitsche term consistent with the momentum equation. The adjoint
term uses $\sigma(v,q)n$ and gives the symmetric formulation used in the demo.
```

## Fitted Boundary Conditions

The channel inflow and walls are fitted to the background mesh. They can
therefore be imposed with ordinary DOLFINx Dirichlet boundary conditions on the
velocity subspace:

```python
inflow_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(x[0], -3.0)
)
wall_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(np.abs(x[1]), 1.0)
)

inflow_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inflow_facets)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
```

The inflow profile is parabolic:

$$
u_{\mathrm{in}}(x,y)=(1-y^2,0).
$$

The pressure is fixed on the outflow to remove the nullspace:

```python
p_dofs = fem.locate_dofs_topological((W.sub(1), Q), fdim, outflow_facets)
bcs.append(fem.dirichletbc(p_zero, p_dofs, W.sub(1)))
```

Thus the example deliberately mixes two boundary-condition mechanisms:
standard strong conditions on fitted boundaries, and weak Nitsche conditions on
the embedded obstacle.

## Velocity Ghost Penalty

Small cut cells near the obstacle can degrade the conditioning of the velocity
block. The demo adds a gradient-jump ghost penalty on the cut-boundary facet
band:

$$
g_u(u,v)
=
\sum_{F\in{\mathcal F}_h^\Gamma}
\gamma_g h_F
\int_F
\left[\!\left[\nabla u\right]\!\right]_n:
\left[\!\left[\nabla v\right]\!\right]_n\,ds .
$$

In UFL:

```python
gp_ids = cutfemx.ghost_penalty_facets(cut_data, "phi>0")
dS_ghost = ufl.Measure("dS", subdomain_id=0, subdomain_data=gp_ids, domain=msh)

a += avg(gamma_g) * avg(h) * inner(
    jump(grad(u), n),
    jump(grad(v), n),
) * dS_ghost
```

## Pressure Stabilization

The pressure is also stabilized on the active interior facets:

$$
g_p(p,q)
=
\sum_{F\in{\mathcal F}_h^f}
\gamma_p h_F^3
\int_F
\left[\!\left[\nabla p\right]\!\right]_n
\left[\!\left[\nabla q\right]\!\right]_n\,ds .
$$

The demo obtains the active interior facets from the active fluid cells:

```python
interior_entities = np.union1d(active_entities, intersected_entities)
interior_ids = cutfemx.interior_facets_for_cells(msh, interior_entities)
dS_interior = ufl.Measure("dS", subdomain_id=1, subdomain_data=interior_ids, domain=msh)
```

and adds

```python
a += avg(gamma_p) * avg(h)**3 * inner(
    jump(grad(p), n),
    jump(grad(q), n),
) * dS_interior
```

The factor $h_F^3$ reflects the scaling of a pressure-gradient jump penalty in
this continuous pressure space. Its role is to suppress pressure modes that can
otherwise be weakly controlled near the cut obstacle and on the active
skeleton.

## Assembly And Output

The demo assembles a CutFEMx runtime form, applies fitted boundary conditions,
deactivates inactive degrees of freedom, and solves a serial MatrixCSR system:

```python
a_cut = cut_form(a)
L_cut = cut_form(L)

A = assemble_matrix(a_cut, bcs=bcs)
b = assemble_vector(L_cut)
deactivate_outside(A, b, active_domain(a_cut))
```

The velocity and pressure are then written to XDMF files:

```python
with io.XDMFFile(msh.comm, "stokes_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_out)
with io.XDMFFile(msh.comm, "stokes_p.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(p_vis)
```

The output is on the background mesh in this demo. For physical-domain output,
the solution can additionally be transferred to a cut mesh with
`cutfemx.fem.cut_function(...)`, as described in the visualization section.

Run the complete example with:

```bash
python python/demo/demo_stokes.py
```

