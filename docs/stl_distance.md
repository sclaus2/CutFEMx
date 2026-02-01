# STL Distance and Level Set Reinitialization

This document details the algorithms and implementation strategies used in CutFEMx for computing signed distance fields from STL surfaces and reinitializing level set functions in a distributed MPI environment.

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

We solve the Eikonal equation $|\nabla \phi| = 1$ using the **Fast Iterative Method (FIM)**. Unlike the Fast Marching Method (which uses a global priority queue and fixes one node per step), FIM allows multiple nodes to be updated simultaneously and re-updated if necessary, making it ideal for parallel architectures and distributed meshes.

#### 1. Update Schemes (Geometric Upwind)

The core operation is computing a candidate distance $T(\mathbf{x})$ for a vertex $\mathbf{x}$ based on the values of its neighbors. We use geometric update stencils that correspond to characteristics flowing along edges, across faces, or through volumes.

*   **1D Update (Edge)**: Wavefront arrives along an edge from neighbor $\mathbf{x}_1$.
    $$ T_{new} = T(\mathbf{x}_1) + \|\mathbf{x} - \mathbf{x}_1\| $$

*   **2D Update (Triangle)**: Wavefront arrives across a triangle face formed by neighbors $\mathbf{x}_1, \mathbf{x}_2$.
    We minimize $T(\alpha) = (1-\alpha)T(\mathbf{x}_1) + \alpha T(\mathbf{x}_2) + \|\mathbf{x} - ((1-\alpha)\mathbf{x}_1 + \alpha \mathbf{x}_2)\|$.
    This is equivalent to solving a quadratic equation for the optimal entry point.

*   **3D Update (Tetrahedron)**: Wavefront arrives through a tetrahedron from a face $\{\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3\}$.
    Solved using Projected Newton's Method to find the optimal entry point on the face $(u, v)$ minimizing the travel time.

For every active vertex, we query all incident cells (triangles in 2D, tetrahedra in 3D) and apply the highest-dimensional update possible, falling back to lower dimensions if causality conditions (gradient direction) are not met.

#### 2. Algorithm Flow (Active Set)

1.  **Initialization**: All vertices with known exact distances (Nearfield) are marked as `ACTIVE` and pushed to a queue.
2.  **Iteration**: While the queue is not empty:
    *   Pop vertex $u$. Mark as inactive.
    *   **Relax Neighbors**: For each neighbor $v$ of $u$:
        *   Compute candidate distance $D_{new}$ using the stencils above (using $u$ and mutual neighbors).
        *   If $D_{new} < D(v) - \epsilon$:
             *   Update $D(v) = D_{new}$.
             *   If $v$ is not `ACTIVE`, mark it `ACTIVE` and push to queue.

#### 3. Parallel Communication (Ghost Sync)

In a distributed setting, the wavefront must cross MPI partition boundaries.

*   **Shared State**: Boundary vertices exist as "Owned" on one rank and "Ghosts" on neighbors.
*   **Sync Interval**: Every $k$ iterations (default 2), we synchronize ghost values.
*   **Min-Reduction**:
    1.  Each rank sends its current values for shared vertices to neighbors.
    2.  Neighbors compute $D_{ghost} = \min(D_{local}, D_{received})$.
*   **Reactivation**:
    *   If a ghost value decreases significantly, it is added to the local `ACTIVE` queue.
    *   **Crucial Step**: We add the updated ghost vertices to the `ACTIVE` queue. In the next iteration, the solver "pops" these ghosts and relaxes their neighbors. (Implementation Note: We also proactively add the breakdown neighbors to the queue to accelerate this inward propagation).

#### 4. Example Scenario

Consider a 2D mesh distributed on 2 Ranks (Left and Right).
1.  **Init**: Source is in Rank Left. Rank Left computes distances up to the partition boundary. Rank Right has all values set to $\infty$.
2.  **Sync**: Rank Left sends finite values for boundary vertices to Rank Right.
3.  **Update**: Rank Right receives these values. The boundary ghosts on Rank Right drop from $\infty$ to real distances.
4.  **Reactivation**: Rank Right adds these ghosts and their inner neighbors to the active queue.
5.  **Propagation**: Rank Right's FIM solver picks up these active nodes and propagates the wavefront eastward through its domain.
6.  **Convergence**: Process stops when no values change on any rank.

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
from cutfemx.level_set import compute_distance_from_stl, SignMode

# 1. Setup Mesh
mesh = create_box(MPI.COMM_WORLD, [[0,0,0], [1,1,1]], [10,10,10], CellType.tetrahedron)

# 2. Compute Signed Distance (using default ComponentAnchor method)
dist_func = compute_distance_from_stl(mesh, "geometry.stl", sign_mode=SignMode.ComponentAnchor)

# 3. Use other sign methods if needed
dist_winding = compute_distance_from_stl(mesh, "geometry.stl", sign_mode=SignMode.WindingNumber)
```
