# Exterior-Facet Runtime Quadrature

This example follows `python/demo/demo_boundary_sphere_perimeter.py`. It is a
geometric quadrature test rather than a PDE solve. The purpose is to show that
the same cut-geometry machinery used for cut cells can also be applied to
facets of the background mesh and to lower-dimensional intersections of those
facets. This is the geometric setting needed for surface loads, contact
patches, immersed boundary measurements, and boundary-restricted diagnostics.

```{admonition} Geometry
:class: geometry

Let the background mesh cover the unit cube

$$
B=[0,1]^3 .
$$

A sphere with centre $c$ and radius $r$ is represented implicitly by

$$
\phi(x)=\|x-c\|^2-r^2 .
$$

The centre is chosen so that the zero level set intersects only one boundary
face of the box. Restricted to that face, the negative phase is a disk and the
zero level set is its boundary circle:

$$
D=\{x\in\partial B:\phi(x)<0\},
\qquad
C=\{x\in\partial B:\phi(x)=0\}.
$$
```

## What Is Being Tested

Most CutFEM examples cut volume cells. Here the entities being cut are
exterior facets. In three dimensions a background cell has dimension three,
but an exterior facet has dimension two. Cutting such a facet can produce two
different geometric objects:

$$
F\cap D
\quad\text{and}\quad
F\cap C,
$$

where $F$ is a boundary triangle. The first object is a surface patch and the
second is a curve segment. The example checks that CutFEMx can integrate both
objects with the same runtime quadrature interface.

```{admonition} Codimension hierarchy
:class: theory

The sphere is a surface in $\mathbb R^3$, but the demo does not integrate over
the whole sphere. It first restricts the geometry to the boundary face
$\partial B$ and then cuts that face by $\phi=0$. Thus the curve $C$ has
codimension two in the ambient three-dimensional space:

$$
C=\partial B\cap\{x:\phi(x)=0\}.
$$

This is why the perimeter integral is a line integral even though the
background mesh is three-dimensional.
```

## Facet Cutting

The relevant mesh entities are exterior facets. Their dimension is passed
explicitly to `cutfemx.cut`:

```python
fdim = msh.topology.dim - 1
msh.topology.create_entities(fdim)
msh.topology.create_connectivity(fdim, msh.topology.dim)

exterior_facets = mesh.exterior_facet_indices(msh.topology)
cut_data = cutfemx.cut(phi, exterior_facets, fdim)
```

Once the facets have been cut, the same selector language separates the
negative part of the face from the curve where the level set vanishes:

```python
negative_facets = cutfemx.locate_entities(cut_data, "phi<0")
cut_facets = cutfemx.locate_entities(cut_data, "phi=0")
```

The selectors have the same meaning as in the volume examples, but now they
act on a two-dimensional parent entity. For a boundary facet $F$,
`"phi<0"` means $F\cap D$ and `"phi=0"` means $F\cap C$.

## Runtime Measures On Facets

The geometric quantities of interest are

$$
|D|=\int_D 1\,ds,
\qquad
|C|=\int_C 1\,d\ell .
$$

Accordingly, the demo constructs one quadrature provider for the surface patch
and one for the curve:

```python
curve_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", args.order)
patch_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", args.order)
```

The `runintgen.dsq` measure is the runtime counterpart of an exterior-facet
measure. It is used for both integrations, with the geometric dimension encoded
in the quadrature provider:

```python
one = fem.Constant(msh, np.float64(1.0))

cut_patch_area = _assemble_scalar(
    one * dsq(domain=msh, quadrature_provider=patch_rules)
)
perimeter = _assemble_scalar(
    one * dsq(domain=msh, quadrature_provider=curve_rules)
)
```

```{admonition} Surface and curve integrals
:class: formulation

The surface patch area is approximated by

$$
|D|
=
\int_D 1\,ds
\approx
\sum_{F\in{\mathcal F}_{h,\partial}}
\sum_{q=1}^{N_F} w_{Fq}.
$$

The perimeter is approximated by the embedded curve rule

$$
|C|
=
\int_C 1\,d\ell
\approx
\sum_{F\in{\mathcal F}_{h,\partial}^C}
\sum_{q=1}^{M_F}\omega_{Fq}.
$$

The two expressions look similar, but their weights have different geometric
dimension. The patch weights approximate area, whereas the curve weights
approximate length.
```

## Standard And Runtime Contributions

The disk $D$ can contain complete exterior facets as well as cut pieces. The
demo therefore also assembles the contribution from facets that are entirely
inside the negative phase with an ordinary DOLFINx exterior-facet measure:

```python
standard_tags = mesh.meshtags(
    msh, fdim, negative_facets, np.full(negative_facets.size, 1, dtype=np.int32)
)
standard_area = _assemble_scalar(
    one * ufl.Measure("ds", domain=msh, subdomain_data=standard_tags)(1)
)
```

The reported area is

$$
|D|_h
=
\int_{D_h^{\mathrm{uncut}}}1\,ds
+
\int_{D_h^{\mathrm{cut}}}1\,ds .
$$

In code this is printed as

```python
standard_area + cut_patch_area
```

The perimeter has no standard-facet contribution because the curve
$C=\{\phi=0\}$ is produced only on facets intersected by the zero level set.

## Cut Meshes For Inspection

The script also constructs cut meshes for visualization:

```python
curve_mesh = cutfemx.create_cut_mesh(cut_data, "phi=0", mode="cut_only")
patch_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="cut_only")
```

These meshes are not needed for the scalar integrals. They are diagnostic
objects: `patch_mesh` shows the disk-like cut surface on the cube boundary,
whereas `curve_mesh` shows the circular boundary of that disk. This mirrors
the PDE examples, where the computation is assembled on runtime quadrature but
the cut mesh is useful for visual inspection.

## Exact Values

For this configuration the exact answer is known. If the sphere centre is
$c=(c_x,c_y,c_z)$ and the cut is taken on the face $x=0$, then the circle
radius on that face is

$$
\rho=\sqrt{r^2-c_x^2}.
$$

Thus the exact perimeter and disk area are

$$
|C|=2\pi\rho,
\qquad
|D|=\pi\rho^2.
$$

The demo reports the assembled values beside these closed-form quantities:

```python
exact_perimeter = 2.0 * np.pi * circle_radius
exact_area = np.pi * circle_radius**2
```

This makes the example a compact verification of exterior-facet cut
quadrature. The perimeter tests the induced curve rule, while the disk area
tests the combination of standard exterior-facet integration and runtime patch
integration.

Run the complete example with:

```bash
python python/demo/demo_boundary_sphere_perimeter.py
```
