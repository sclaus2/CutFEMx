# Element Classification

Element classification is the first geometric step in a CutFEM computation. It
answers a simple question for each background entity: which part of the
implicit geometry does this entity see? The answer determines where standard
quadrature is sufficient, where runtime quadrature is required, and which
degrees of freedom belong to the active problem.

```{admonition} Classification induced by a level set
:class: theory

Let $\phi$ be a scalar finite element function on the background mesh
${\mathcal T}_h$. A cell $K\in{\mathcal T}_h$ is classified relative to the
negative phase $\Omega^-=\{x:\phi(x)<0\}$ and the interface
$\Gamma=\{x:\phi(x)=0\}$.

The three basic geometric cases are

$$
K\subset\Omega^-,
\qquad
K\subset\Omega^+,
\qquad
K\cap\Gamma\ne\emptyset .
$$

The active mesh for a problem posed on $\Omega^-$ is

$$
{\mathcal T}_h^\Omega
=
\{K\in{\mathcal T}_h:K\cap\Omega^-\ne\emptyset\}.
$$

This includes both cells completely inside the physical domain and cells cut by
the interface. Cells outside the active mesh remain part of the background data
structure, but they do not contribute physical integrals.
```

## Construct Cut Data

The public entry point is `cutfemx.cut(...)`. It stores the CutCells geometric
state and retains the Python level-set function so that the cut state can later
be refreshed.

```python
import cutfemx

cut_data = cutfemx.cut(phi)
```

Conceptually, this operation builds a local geometric model of the intersection
between the background entities and the zero set of $\phi$. Later operations do
not re-read the raw level-set function independently; they query this `CutData`
object.

## Locate Cells

Selectors are strings describing a phase or an interface:

```python
inside_cells = cutfemx.locate_entities(cut_data, "phi<0")
outside_cells = cutfemx.locate_entities(cut_data, "phi>0")
interface_cells = cutfemx.locate_entities(cut_data, "phi=0")
active_cells = cutfemx.locate_entities(cut_data, "phi<=0")
```

For a cell cut, the returned arrays contain local background cell indices. The
meaning of each selector is geometric:

- `"phi<0"` selects cells lying in the negative phase.
- `"phi>0"` selects cells lying in the positive phase.
- `"phi=0"` selects cells intersected by the zero level set.
- `"phi<=0"` selects the active negative-phase domain, including cut cells.

```{admonition} Selector language
:class: practice

Selectors are geometric predicates, not Python expressions evaluated pointwise
inside a form. For a single level set, `"phi=0"` should be read as

$$
K\cap\Gamma\ne\emptyset,
$$

not as a numerical tolerance band. For several level sets, predicates can be
combined to describe Boolean regions, for example a material domain inside one
interface and outside another.
```

## Standard Cells And Cut Cells

Classification separates the integration problem into two parts. On cells
fully contained in a selected phase, ordinary FEniCSx quadrature is sufficient.
On cut cells, the physical integration domain is only a subset of the cell:

$$
\int_\Omega f\,dx
=
\sum_{K\subset\Omega}\int_K f\,dx
+
\sum_{K\cap\Gamma\ne\emptyset}\int_{K\cap\Omega}f\,dx .
$$

This decomposition is why most CutFEMx measures accept both an array of
standard entities and a runtime quadrature provider. The classification step
identifies the first sum; runtime quadrature supplies the second.

## Cut Selected Entities

The same classification idea applies to entities other than cells. To cut
exterior facets, interior facets, or another lower-dimensional set, pass the
candidate entity array and the entity dimension:

```python
from dolfinx import mesh

fdim = msh.topology.dim - 1
msh.topology.create_entities(fdim)
msh.topology.create_connectivity(fdim, msh.topology.dim)

exterior_facets = mesh.exterior_facet_indices(msh.topology)
facet_cut = cutfemx.cut(phi, exterior_facets, fdim)

negative_facets = cutfemx.locate_entities(facet_cut, "phi<0")
cut_facets = cutfemx.locate_entities(facet_cut, "phi=0")
```

For selected-entity cuts, `locate_entities` returns local indices in the same
entity dimension as the supplied candidates. In three dimensions, cutting
facets produces surface patches for `"phi<0"` and curve segments for
`"phi=0"`. This is the mechanism behind runtime exterior-facet and
interior-facet quadrature.

```{admonition} Host entity
:class: theory

The host entity determines the dimension of the geometric object being cut. If
cells are cut in a three-dimensional mesh, `"phi=0"` represents a surface
inside each cut cell. If exterior facets are cut, `"phi=0"` represents a curve
on those boundary facets. The same selector can therefore refer to different
geometric measures depending on the host entity.
```

## Multiple Level Sets

Pass a sequence of DOLFINx functions to cut several level sets together:

```python
multi_cut = cutfemx.cut([phi_fluid, phi_solid])
fluid_without_solid = cutfemx.locate_entities(multi_cut, "phi<0 and phi1>0")
```

The first unnamed level set is called `phi`, then `phi1`, `phi2`, and so on.
For example, the region inside one level set and outside another is

$$
\Omega
=
\{x:\phi_0(x)<0\}
\cap
\{x:\phi_1(x)>0\}.
$$

For named DOLFINx functions, selector names are taken from the function names
when they are valid identifiers and unique. This is useful for readable
multi-material selectors, but names should be chosen before constructing the
`CutData` object.
