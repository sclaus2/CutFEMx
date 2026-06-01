# STL Distance Computation

This example follows `python/demo/demo_stl_distance.py`. It computes a signed
distance field from an STL surface and writes the result to XDMF. The example
is not a CutFEM variational solve; it produces a level-set function that can be
used later as the geometry input for cut integration. In a typical unfitted
workflow, this is the bridge between a triangulated CAD or scan surface and the
level-set representation expected by the CutFEMx cutting routines.

```{admonition} Signed distance field
:class: formulation

Given a surface $\Sigma\subset\mathbb R^3$, the signed distance function is

$$
\phi(x)=s(x)\operatorname{dist}(x,\Sigma),
\qquad
\operatorname{dist}(x,\Sigma)=\inf_{y\in\Sigma}|x-y|,
$$

where $s(x)$ determines on which side of the surface the point lies. Away from
the medial axis, the distance function satisfies the Eikonal equation

$$
|\nabla\phi|=1.
$$

For CutFEM, this field is useful because its zero contour is an implicit
surface and its gradient provides a geometrically meaningful normal direction.
```

## Role In A CutFEM Workflow

The cut examples above start from analytic level sets such as a circle. An STL
file instead provides only a triangulated surface. To use this geometry in a
CutFEM computation one first constructs a scalar field $\phi_h$ on a
background mesh such that

$$
\Gamma_h=\{x:\phi_h(x)=0\}
$$

approximates the STL surface. The sign convention then determines which phase
is interpreted as solid, fluid, or exterior. Once this field is available, the
same commands used in the PDE examples can be applied:

```python
cut_data = cutfemx.cut(phi_h)
fluid_rules = cutfemx.runtime_quadrature(cut_data, "phi>0", order)
surface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)
```

Thus the distance demo is a geometry preprocessing example rather than an
assembly example. Its output is precisely the input needed by the cut-cell,
interface, and embedded-boundary examples.

## Background Mesh From The STL Box

The demo first computes the STL bounding box on rank zero, broadcasts it, adds
padding, and creates a tetrahedral background mesh:

```python
from cutfemx.distance import compute_stl_bbox

min_c, max_c = compute_stl_bbox(stl_path)
min_pt_mesh = min_pt_stl - padding
max_pt_mesh = max_pt_stl + padding

mesh = create_box(
    comm,
    [min_pt_mesh, max_pt_mesh],
    N_mesh,
    CellType.tetrahedron,
    ghost_mode=GhostMode.shared_facet,
)
```

The number of cells in each coordinate direction is chosen from the padded box
side lengths:

```python
length_mesh[:] = max_pt_mesh - min_pt_mesh
min_length = np.min(length_mesh)
n_min = 5
N_mesh[:] = np.round(length_mesh / min_length * n_min).astype(int)
```

This gives an anisotropic Cartesian box while keeping the smallest direction
resolved by roughly `n_min` cells. The construction is deliberately simple:
local refinement near the STL will add the resolution needed for the geometry.

```{admonition} Why a padded box is used
:class: theory

The signed distance should be represented not only on the surface, but in a
neighbourhood of it. A padded background box gives room for both signs of the
distance field and avoids placing the artificial outer boundary exactly on the
STL. This is useful when the resulting level set is later used to form an
active mesh, because the cut cells and ghost-penalty band need nearby
background cells on both sides of the zero contour.
```

## Adapt Around The Surface

Before computing the distance, the mesh is refined near the STL surface:

```python
refined_mesh = adapt_mesh_to_stl(
    mesh,
    stl_path,
    nlevels=3,
    k_ring=1,
    aabb_padding=0.0,
    ghost_mode=GhostMode.shared_facet,
)
```

The refinement is geometric: cells whose bounding boxes are close to the STL
surface are marked, and a small ring of neighbouring cells is included. This
improves the resolution of the zero level set without refining the whole
background box.

Mathematically, this replaces a coarse background mesh ${\mathcal T}_h$ by a
locally refined mesh ${\mathcal T}_{h,\Sigma}$ whose small cells are
concentrated near $\Sigma$. The distance function is nonsmooth on the medial
axis and changes most rapidly in the normal direction near the surface.
Refining around $\Sigma$ therefore improves the geometry available to
subsequent cut quadrature without paying for a uniformly fine
three-dimensional mesh.

The parameters have geometric meanings:

- `nlevels` controls how many refinement sweeps are performed,
- `k_ring` expands the marked region by neighbouring cell layers,
- `aabb_padding` enlarges the search boxes used to detect cells close to the
  STL triangles.

## Distance And Sign

The signed distance is computed with

```python
from cutfemx.distance import SignMode, from_stl

dist = from_stl(refined_mesh, stl_path, sign_mode=SignMode.ComponentAnchor)
```

At each degree of freedom, the unsigned part approximates

$$
d(x_i,\Sigma)=\min_{y\in\Sigma}|x_i-y|.
$$

The sign is then attached according to the selected sign mode.

```{admonition} Sign construction
:class: theory

The unsigned distance is a nearest-surface quantity. The sign is a topological
or orientation choice. In this demo `SignMode.ComponentAnchor` assigns signs by
using the surface as a barrier between mesh components. This is a practical
default for scanned or triangulated surfaces where local triangle normals alone
may not provide a globally reliable inside/outside classification.
```

## Diagnostics

The demo reduces the range of the distance values over all MPI ranks:

```python
local_values = dist.x.array
local_min = np.min(local_values) if local_values.size else np.inf
local_max = np.max(local_values) if local_values.size else -np.inf
local_neg = np.count_nonzero(local_values < 0.0)
global_min = comm.allreduce(local_min, op=MPI.MIN)
global_max = comm.allreduce(local_max, op=MPI.MAX)
global_neg = comm.allreduce(local_neg, op=MPI.SUM)
```

The minimum and maximum indicate whether both sides of the surface are present
in the background box. The count of negative degrees of freedom is a practical
sanity check for the sign construction.

If the field is later used as a level set, one may additionally inspect the
Eikonal defect

$$
|\nabla\phi_h|-1
$$

away from the surface medial axis and away from the artificial box boundary. A
well-resolved signed distance field should have a small defect in the region
where the CutFEM computation will use normals and interface quadrature.

## Output

The distance is a DOLFINx function on the refined mesh. The demo writes it
together with the mesh:

```python
with XDMFFile(comm, "distance_from_stl.xdmf", "w") as xdmf:
    xdmf.write_mesh(refined_mesh)
    xdmf.write_function(dist)
```

The output can be inspected in ParaView. The zero isosurface of `dist` should
recover the STL geometry, while positive and negative contours show the chosen
phase convention. If the field will be used in another script, it can be read
back as a DOLFINx function or regenerated directly from the STL in the
preprocessing stage.

Run the complete example with:

```bash
python python/demo/demo_stl_distance.py
```
