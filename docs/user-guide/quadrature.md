# Runtime Quadrature

Runtime quadrature is the bridge between CutCells geometry and UFL forms. A
standard form compiler assumes that quadrature rules are known from the
reference cell type. In CutFEM, the integration region depends on where the
level set cuts each entity, so some quadrature rules must be generated after
the geometry is known.

```{admonition} Cut-cell quadrature
:class: formulation

For a domain $\Omega_h$ represented on a background mesh, a volume integral is
assembled cell by cell as

$$
\int_{\Omega_h} f(x)\,dx
=
\sum_{K\in{\mathcal T}_h^\Omega}
\int_{K\cap\Omega_h}f(x)\,dx .
$$

On uncut cells, $K\cap\Omega_h=K$ and standard cell quadrature is used. On cut
cells, CutCells constructs a rule

$$
\int_{K\cap\Omega_h}f(x)\,dx
\approx
\sum_{q=1}^{N_K}w_{Kq}f(x_{Kq}).
$$

For an embedded interface or boundary,

$$
\int_{\Gamma_h}g(x)\,ds
\approx
\sum_{K\in{\mathcal T}_h^\Gamma}
\sum_{q=1}^{M_K}\omega_{Kq}g(\xi_{Kq}).
$$

The points and weights are part of the runtime data associated with the cut
geometry.
```

## Providers

Use `cutfemx.runtime_quadrature(cut_data, selector, order)` to create a
`runintgen`-compatible quadrature provider:

```python
cell_cut = cutfemx.cut(phi)

inside_cells = cutfemx.locate_entities(cell_cut, "phi<0")
omega_rules = cutfemx.runtime_quadrature(cell_cut, "phi<0", order=4)
gamma_rules = cutfemx.runtime_quadrature(cell_cut, "phi=0", order=4)
```

A provider stores the parent entity map, offsets into the point and weight
arrays, and the reference coordinates used by the runtime kernels. Physical
coordinates are computed lazily when they are needed for plotting or
diagnostics:

```python
points = omega_rules.with_physical_points().physical_points
weights = omega_rules.weights
```

```{admonition} Reference and physical points
:class: theory

Form evaluation uses reference coordinates on the host entity. For cell-hosted
rules, the quadrature points are mapped through the parent cell geometry. For
facet-hosted rules, they are mapped through the corresponding facet chart and
then into physical space. The runtime provider keeps this parent-entity
information so that coefficient evaluation and basis tabulation are performed
on the correct background cell.
```

## Cell Measures

For a mixed standard/runtime volume integral, pass both ordinary entity ids and
runtime rules as `subdomain_data`:

```python
dx_omega = ufl.Measure(
    "dx",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[inside_cells, omega_rules],
)
```

The standard entity array covers uncut cells. The runtime provider covers cut
cells. Together they represent the full selected phase without forcing every
cell to use generated quadrature.

For an embedded interface from a cell cut, use the interface rules directly:

```python
dx_gamma = ufl.Measure(
    "dx",
    domain=msh,
    subdomain_id=1,
    subdomain_data=gamma_rules,
)
```

Although this is written as a UFL `"dx"` measure, the provider represents a
lower-dimensional object embedded in the cell. The runtime weights already
carry the appropriate physical measure factor for the selected geometry.

```{admonition} Mixed standard/runtime integration
:class: practice

Combining ordinary entity ids with runtime rules lets DOLFINx use standard
quadrature on cells that are fully inside the domain and specialized quadrature
only on cells crossed by the interface. This avoids paying the cut-quadrature
cost where the geometry is already simple.
```

## Exterior Facet Cuts

Boundary integrals on fitted exterior facets use ordinary UFL `ds` measures.
If the physical boundary itself cuts those exterior facets, the facets must be
cut explicitly:

```python
from dolfinx import mesh

fdim = msh.topology.dim - 1
exterior_facets = mesh.exterior_facet_indices(msh.topology)
facet_cut = cutfemx.cut(phi, exterior_facets, fdim)

negative_facets = cutfemx.locate_entities(facet_cut, "phi<0")
facet_rules = cutfemx.runtime_quadrature(facet_cut, "phi<0", order=4)

ds_omega = ufl.Measure(
    "ds",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[negative_facets, facet_rules]
)
```

The runtime rules are carried directly by the ordinary UFL `ds` measure. In
three dimensions, `"phi<0"` selects surface patches on boundary facets, while
`"phi=0"` would select curves on those facets.

## Interior Facet Cuts

Interior-facet quadrature is needed for discontinuous Galerkin methods, active
skeleton terms, and some stabilization terms. First identify the relevant
interior facets, then cut those facets:

```python
active_cells = cutfemx.locate_entities(cell_cut, "phi<=0")
skeleton_facets = cutfemx.interior_facets_for_cells(msh, active_cells)
skeleton_cut = cutfemx.cut(phi, skeleton_facets, fdim)

inside_skeleton_facets = cutfemx.locate_entities(skeleton_cut, "phi<0")
skeleton_rules = cutfemx.runtime_quadrature(skeleton_cut, "phi<0", order=4)

dS_omega = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_id=0,
    subdomain_data=[inside_skeleton_facets, skeleton_rules],
)
```

The runtime rules are carried directly by the ordinary UFL `dS` measure. It
represents two-sided interior-facet integration on the selected part of each
facet.

```{admonition} Skeleton terms
:class: formulation

For a DG method, an interior-facet term has the structure

$$
\sum_{F\in{\mathcal F}_h^\Omega}
\int_{F\cap\Omega_h}
H(u_h^+,u_h^-,v_h^+,v_h^-)\,ds .
$$

If $F$ is cut by the physical boundary, the integration region is
$F\cap\Omega_h$, not the full background facet. Runtime interior-facet
quadrature is what makes this distinction visible to the form assembler.
```

## Choosing The Order

The quadrature order should be high enough for the polynomial degree of the
finite element space and for the geometric reconstruction used by the level
set. The examples typically use `order=4` for first- and second-order tutorial
problems. For convergence studies, the quadrature order should be increased
until quadrature error is not the dominant error source.
