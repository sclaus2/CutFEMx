# Distance Module: STL Distance and Level Set Reinitialization

This document details the algorithms and implementation strategies used by
`cutfemx.distance` for computing signed distance fields from STL surfaces and
reinitializing level set functions in a distributed MPI environment.

## 1. STL Surface Distribution

To handle large STL files efficiently in parallel, we distribute the surface facets among MPI ranks based on the partitioning of the background mesh.

### Algorithm: `distribute_stl_facets`

1.  **Domain Bounding Boxes**:
    *   Each MPI rank computes the Axis-Aligned Bounding Box (AABB) of its local mesh partition.
    *   A global AABB tree is constructed using `dolfinx::geometry::BoundingBoxTree::create_global_tree`, where each leaf corresponds to an MPI rank's domain.

2.  **Reading and Routing (Rank 0)**:
    *   Rank 0 reads the STL file (binary or ASCII).
    *   For each triangle in the STL:
        *   Its bounding box is checked against the global rank-AABB tree.
        *   The triangle is assigned to *all* ranks whose domain it potentially overlaps (using a padding tolerance).
    
3.  **Distribution**:
    *   Triangles are bucketed by destination rank.
    *   `MPI_Alltoallv` is used to send the triangle data (vertices and normals) to the target ranks.
    *   Each rank receives a local `TriSoup` containing only the surface subset relevant to its domain.

## 2. Cell-Triangle Mapping

To efficiently query distance from the surface, we build a local spatial index mapping background mesh cells to surface triangles.

### Algorithm: `build_cell_triangle_map`

1.  **AABB Tree Construction**:
    *   A generic AABB tree is built for the local `TriSoup` triangles.
    
2.  **Intersection Query (Broad Phase)**:
    *   For each cell in the local background mesh, we query the triangle AABB tree with the cell's bounding box.
    *   A list of *candidate* triangles is retrieved.

3.  **Robust Filtering (Narrow Phase)**:
    *   We verify exact intersection using **Robust Geometric Predicates** (Shewchuk's `orient3d`).
    *   **Intersection Test**: We check if a triangle intersects a cell (Tetrahedron or Hexahedron) by verifying:
        1.  Are any triangle vertices inside the cell?
        2.  Do any triangle edges intersect the cell faces?
        3.  Do any cell edges intersect the triangle face?
    *   These tests are performed purely using `orient3d` determinants (sign of volume), avoiding constructing intersection points to prevent floating-point consistency errors.
    *   If `CUTFEMX_USE_MULTIPRECISION_PSM` is enabled, these predicates use arbitrary-precision arithmetic (via Geogram) for exact results.
    
    This map guarantees that every cell intersecting the surface is correctly identified, which is critical for the "Nearfield" initialization (preventing holes in the zero level set).

## 3. Fast Marching Method (FMM)

We calculate the distance field by solving the Eikonal equation $|\nabla \phi| = 1$.

### Initialization (Nearfield)

The initialization phase computes exact Euclidean distances for vertices close to the surface.

1.  **Triangle Query**:
    *   We iterate over every cell $C$ in the mesh.
    *   From the `CellTriangleMap`, we retrieve the list of surface triangles $T_C$ intersecting $C$.

2.  **Point-Triangle Distance**:
    *   For each vertex $\mathbf{v} \in C$:
        *   We compute the squared Euclidean distance to every triangle $t \in T_C$.
        *   $d^2(\mathbf{v}, t) = \text{point\_triangle\_distance\_sq}(\mathbf{v}, \mathbf{a}, \mathbf{b}, \mathbf{c})$.
        *   This involves projecting $\mathbf{v}$ onto the triangle plane and clamping barycentric coordinates to the triangle bounds.
    *   We store the minimum distance: $d(\mathbf{v}) = \min_{t} d(\mathbf{v}, t)$.
    *   We also store the index of the **closest triangle**, which is required for sign determination (Method 1).

3.  **Fixing**:
    *   Vertices with a valid distance (found via the map) are marked as `KNOWN` / `ACTIVE`.
    *   These form the initial boundary condition for the FMM propagation.


### Propagation (Farfield)

We solve the Eikonal equation $|\nabla \phi| = 1$ on the vertices of the mesh. Each process runs a fast marching solve on its part of the mesh, ordered by a priority queue, and the processes then exchange the distances of the vertices they share. The solve is *label correcting*: a vertex whose distance improves after it was processed is processed again. This keeps the result exact on stencils that are not causal (obtuse simplices, distorted cells), where a pure fast marching method would accept wrong values, and it lets distances from neighbouring processes arrive in any order.

#### 1. Update Schemes (Geometric Upwind)

The core operation is computing a candidate distance $T(\mathbf{x})$ for a vertex $\mathbf{x}$ based on the values of its neighbors. We use geometric update stencils that correspond to characteristics flowing along edges, across faces, or through volumes.

*   **1D Update (Edge)**: Wavefront arrives along an edge from neighbor $\mathbf{x}_1$.
    $$ T_{new} = T(\mathbf{x}_1) + \|\mathbf{x} - \mathbf{x}_1\| $$

*   **2D Update (Triangle)**: Wavefront arrives across a triangle face formed by neighbors $\mathbf{x}_1, \mathbf{x}_2$.
    We minimize $T(\alpha) = (1-\alpha)T(\mathbf{x}_1) + \alpha T(\mathbf{x}_2) + \|\mathbf{x} - ((1-\alpha)\mathbf{x}_1 + \alpha \mathbf{x}_2)\|$.
    This is equivalent to solving a quadratic equation for the optimal entry point.

*   **3D Update (Tetrahedron)**: Wavefront arrives through a tetrahedron from a face $\{\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3\}$.
    The objective $T(\mathbf{p}) + \|\mathbf{x} - \mathbf{p}\|$ is convex in the entry point $\mathbf{p}$, so an interior stationary point is the minimum over the whole face. It is computed in closed form: the unit vector $\mathbf{g}$ from $\mathbf{p}$ to $\mathbf{x}$ has the in-plane part fixed by $\mathbf{g}\cdot(\mathbf{x}_i - \mathbf{x}_1) = T(\mathbf{x}_i) - T(\mathbf{x}_1)$ and the normal part by $\|\mathbf{g}\| = 1$. If $\mathbf{p}$ falls outside the face, the edge updates give the minimum.

Quadrilaterals and hexahedra are split into virtual triangles and tetrahedra (2 and 6, along the diagonal from vertex 0). For axis-aligned boxes these are right triangles and orthoschemes, so the stencils are causal.

#### 2. Local Solve (Priority Queue)

1.  **Initialization**: All vertices with known distances (Nearfield) are pushed to a min-heap keyed by distance.
2.  **Drain**: While the heap is not empty:
    *   Pop the vertex $u$ with the smallest key; skip it if its distance has improved since it was pushed.
    *   **Relax Neighbors**: For each vertex $v$ of a (virtual) simplex containing $u$:
        *   Compute candidate distance $D_{new}$ using the stencils above (using $u$ and the other vertices of the simplex).
        *   If $D_{new} < D(v) - \text{tol}$: set $D(v) = D_{new}$ and push $v$.

On causal stencils every vertex is popped once, as in the fast marching method.

#### 3. Parallel Communication (Rounds)

In a distributed setting, the wavefront must cross MPI partition boundaries. Boundary vertices exist as "Owned" on one rank and "Ghosts" on neighbors. The solve runs in rounds:

1.  **Local solve**: Every rank pops vertices with distances up to $D_{min} + \Delta$, where $D_{min}$ is the smallest pending distance on any rank.
2.  **Exchange** (DOLFINx neighbourhood scatter, only between ranks that share vertices):
    1.  Ghost distances are reduced onto their owners with $\min$.
    2.  Owners overwrite their ghosts, so all copies agree after every round.
    *   Every vertex whose distance improved in the exchange is pushed to the heap again.
3.  **Termination**: One global $\min$ reduction gives the next $D_{min}$; the solve stops when no rank has pending vertices.

The round width $\Delta$ (`FMMOptions::round_width`, by default 1/32 of the diagonal of the mesh bounding box) keeps a rank from computing far ahead of fronts that have not yet arrived from its neighbours, which would be recomputed when they do. The number of rounds is about the range of distances divided by $\Delta$ plus the number of partition boundaries the fronts cross; it does not grow with the mesh size or the number of ranks.

With `max_distance`, vertices are only pushed while their distance is at most `max_distance`, and all larger distances (including unreached vertices) are set to `max_distance`. For level-set reinitialization this computes a narrow band around the interface at a fraction of the cost.

## 4. Signed Distance Computation

Standard FMM computes $|\phi|$ (unsigned). We determine the sign using one of three strategies:

### Method 1: Local Normal Band (`SignMode::LocalNormalBand`)
*   **Idea**: For nearfield vertices, project the vector to the closest point onto the closest triangle's normal.
*   **Sign**: $\text{sgn} = \text{sgn}( (\mathbf{x} - \mathbf{p}_{closest}) \cdot \mathbf{n} )$.
*   **Propagation**: Propagate this sign to the farfield using BFS.
*   **Pros/Cons**: Very fast, but non-robust for complex geometries or poor quality surfaces.

### Method 2: Component Anchor (`SignMode::ComponentAnchor`) - **Default**
*   **Idea**: Utilize the connectivity of the background mesh. The surface cuts the mesh into connected components.
*   **Step 1**: Mark all "cut" facets (facets intersected by the surface).
*   **Step 2**: Identify an "outside" known point (e.g., a boundary vertex of the domain).
*   **Step 3**: Flood fill the "Outside" status via mesh topology. The flood fill is blocked by "cut" facets.
*   **Step 4**: Any region not reached is "Inside".
*   **Result**: Valid global sign even if the surface has minor gaps, as long as it topologically separates the domain.

### Method 3: Winding Number (`SignMode::WindingNumber`)
*   **Idea**: Compute the Generalized Winding Number $w(\mathbf{x}) \approx \frac{1}{4\pi} \sum_{T} \Omega_T(\mathbf{x})$ (solid angle).
*   **Algorithm**:
    *   Uses a Barnes-Hut Octree to approximate the sum for far triangles.
    *   If $w(\mathbf{x}) > 0.5$, point is Inside.
*   **Pros/Cons**: Robust for non-watertight "soup" meshes, but computationally more expensive ($O(N \log M)$).

## 5. Examples

### Python API

```python
from mpi4py import MPI
from dolfinx.mesh import create_box, CellType, GhostMode
from dolfinx.fem import FunctionSpace, Function
from cutfemx.distance import SignMode, from_stl

# 1. Setup Mesh
mesh = create_box(MPI.COMM_WORLD, [[0,0,0], [1,1,1]], [10,10,10], CellType.tetrahedron)

# 2. Compute Signed Distance (using default ComponentAnchor method)
dist_func = from_stl(mesh, "geometry.stl", sign_mode=SignMode.ComponentAnchor)

# 3. Use other sign methods if needed
dist_winding = from_stl(mesh, "geometry.stl", sign_mode=SignMode.WindingNumber)
```
