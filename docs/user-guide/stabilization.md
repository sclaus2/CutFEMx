# Stabilization

CutFEM discretizations are posed on a background space, but the physical
integrals may involve arbitrarily small pieces of some cells. Without
additional stabilization, the stiffness matrix can become badly conditioned
when the interface passes close to a vertex or along a cell edge. The usual
CutFEM remedy is a ghost penalty on a narrow band of interior facets around
the cut boundary.

```{admonition} Why cut cells need stabilization
:class: theory

Consider a basis function whose support intersects the physical domain only in
a small sliver $K\cap\Omega$. Its energy contribution from the physical volume
may be very small even though the algebraic degree of freedom remains present
in the background space. As the relative volume

$$
\frac{|K\cap\Omega|}{|K|}
$$

tends to zero, the condition number can deteriorate in a way that is unrelated
to the mesh size $h$.

Ghost penalties extend control from well-resolved cells into neighbouring cut
cells by penalizing jumps of derivatives across interior facets. For continuous
first-order elements, a common term is

$$
g_h(u_h,v_h)
=
\sum_{F\in{\mathcal F}_h^\Gamma}
\gamma_g h_F
\int_F
\left[\!\left[\nabla u_h\right]\!\right]_n
\left[\!\left[\nabla v_h\right]\!\right]_n\,ds .
$$
```

## Facet Band

The ghost penalty is not applied everywhere. It is applied on a local band of
interior facets surrounding the cut cells:

$$
{\mathcal F}_h^\Gamma
=
\{F=K^+\cap K^-:
K^+\in{\mathcal T}_h^\Gamma
\ \text{or}\
K^-\in{\mathcal T}_h^\Gamma\}.
$$

CutFEMx constructs a depth-1 band with

```python
cell_cut = cutfemx.cut(phi)
ghost_facets = cutfemx.ghost_penalty_facets(cell_cut, "phi<0")
```

The returned array contains local interior facet ids. Use it with a standard
DOLFINx interior-facet measure:

```python
dS_ghost = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_id=2,
    subdomain_data=ghost_facets,
)
```

```{admonition} Choosing the ghost band
:class: practice

The default depth-1 band is the usual choice for scalar elliptic examples. It
controls the extension of finite element functions from well-resolved cells
into neighbouring cut cells while keeping the stabilization local. Increasing
the depth enlarges the stabilized patch, but it also adds more artificial
coupling and should be justified by the method being used.
```

## Continuous Galerkin Penalty

For continuous Lagrange elements, the function itself is continuous across
interior facets, so the lowest useful penalty acts on the normal jump of the
gradient:

$$
\left[\!\left[\nabla u_h\right]\!\right]_n
=
(\nabla u_h^+ - \nabla u_h^-)\cdot n_F .
$$

The corresponding UFL code is

```python
n = ufl.FacetNormal(msh)
h = ufl.CellDiameter(msh)
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

The factor $h_F$ balances the dimensions of the gradient-jump term with the
volume energy. The parameter $\gamma_g$ is usually much smaller than the
Nitsche penalty parameter, because it is not a boundary condition; it is an
extension control.

## Relation To Nitsche Terms

Nitsche terms impose boundary or interface conditions. Ghost penalties do not
impose physical data. Their role is algebraic and analytical: they make norms
on cut domains comparable to norms on neighbouring background cells. In a
typical cut Poisson discretization, the stabilized bilinear form is

$$
a_h^{\mathrm{stab}}(u_h,v_h)
=
a_h^{\mathrm{Nitsche}}(u_h,v_h)
+g_h(u_h,v_h).
$$

The two terms should therefore be tuned separately. A large Nitsche penalty
does not replace ghost stabilization, and a ghost penalty does not enforce
$u=g$ on an embedded boundary.

## DG And Mixed Problems

For DG methods, the active interior skeleton also carries consistency and
penalty terms. Those terms control jumps of the DG solution across active
facets, while the ghost penalty controls small-cut-cell extension. In mixed
problems such as Stokes, additional pressure ghost penalties may be used to
control pressure modes near the cut boundary:

$$
g_p(p_h,q_h)
=
\sum_{F\in{\mathcal F}_h^f}
\gamma_p h_F^3
\int_F
\left[\!\left[\nabla p_h\right]\!\right]_n
\left[\!\left[\nabla q_h\right]\!\right]_n\,ds .
$$

See `python/demo/demo_dg_poisson.py` and `python/demo/demo_stokes.py` for
complete examples of these variants.
