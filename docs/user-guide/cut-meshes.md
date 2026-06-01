# Cut Meshes

CutFEM computations are assembled on the background mesh. The physical domain
is represented by selectors and runtime quadrature, not by replacing the mesh
used for the finite element space. Nevertheless, it is often useful to build an
explicit mesh of the selected geometry for visualization, export, and
diagnostics. This is the role of `cutfemx.create_cut_mesh(...)`.

```{admonition} Geometry mesh versus computation mesh
:class: geometry

Let ${\mathcal T}_h$ be the background mesh and let

$$
\Omega_h=\{x:\phi_h(x)<0\}.
$$

The discrete finite element space $V_h$ is defined on ${\mathcal T}_h$. A cut
mesh is a derived geometric object representing pieces such as
$K\cap\Omega_h$ or $K\cap\Gamma_h$. It is not the mesh on which the unknown was
solved.

For a cut cell $K$, CutCells reconstructs the selected region by simpler
subcells,

$$
K\cap\Omega_h \approx \bigcup_i K_i^{\mathrm{cut}}.
$$

This reconstruction is intended for output and inspection. The variational
problem itself is still represented by the background function space and the
runtime quadrature rules.
```

## Creating A Cut Mesh

Start from existing cut data and a selector:

```python
cut_data = cutfemx.cut(phi)
cut_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
```

The selector determines which geometric part is reconstructed. The returned
`cutfemx.CutMesh` stores both the visualization mesh and its relation to the
original background mesh:

- `cut_mesh.mesh`: DOLFINx mesh for visualization or output
- `cut_mesh.parent_index`: parent background cell for each output cell
- `cut_mesh.is_cut_cell`: marker for reconstructed cut output cells

The parent map is important because finite element functions are still known on
the background mesh. It tells CutFEMx which background cell should be used when
evaluating a function on each cell of the cut mesh.

## Full And Cut-Only Modes

Use `mode="full"` when the object of interest is a full physical region. The
output contains ordinary background cells that lie completely in the selected
region together with reconstructed subcells in cut cells:

```python
omega_mesh = cutfemx.create_cut_mesh(cut_data, "phi<0", mode="full")
```

This is the natural choice for plotting a solution on $\Omega_h$. It gives a
visible domain rather than only the interface reconstruction.

Use `mode="cut_only"` when the object of interest is the cut geometry itself,
for example an embedded boundary or a local interface:

```python
gamma_mesh = cutfemx.create_cut_mesh(cut_data, "phi=0", mode="cut_only")
```

If `mode` is omitted, CutFEMx chooses a mode from the selector, but explicit
mode selection is clearer in tutorials and diagnostics.

```{admonition} Interpreting mode
:class: practice

The two modes correspond to different mathematical objects:

$$
\texttt{full}: \Omega_h,
\qquad
\texttt{cut\_only}: \Gamma_h\ \text{or another local intersection}.
$$

Use `full` for a domain visualization and `cut_only` for an interface
visualization. Mixing these two interpretations is a common source of confusing
plots.
```

## Transfer Functions

A background function can be evaluated on the cut mesh by using the parent-cell
map. This produces a new DOLFINx function whose function space lives on the
cut visualization mesh:

```python
phi_cut = cutfemx.fem.cut_function(phi, cut_mesh)
```

For a solved background function:

```python
u_cut = cutfemx.fem.cut_function(uh, cut_mesh)
```

Mathematically, this is the restriction of the background finite element
function to the reconstructed geometry:

$$
u_h^{\mathrm{cut}}(x)=u_h(x),
\qquad x\in\Omega_h^{\mathrm{cut}}.
$$

The operation is not a projection or a new solve. It is evaluation of the same
finite element field on a different geometric representation.

## What To Check

Cut meshes are particularly useful for checking whether the geometric part of a
problem is correct before interpreting PDE results:

- Does `"phi<0"` reconstruct the expected physical domain?
- Does `"phi=0"` lie on the intended embedded boundary?
- Are the cut cells concentrated where the level set crosses the mesh?
- Does the transferred level-set function vanish on the interface mesh?
- Are there empty cut meshes caused by an incorrect selector or sign
  convention?

The transferred function can be passed to `dolfinx.plot.vtk_mesh(...)` or
written with DOLFINx IO facilities when the selected mesh is non-empty.
