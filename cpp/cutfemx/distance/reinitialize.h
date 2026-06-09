#pragma once

// Level Set Reinitialization: Reconstruct signed distance function from level set
// Uses CutCells to extract the zero-contour, then FMM to compute distance

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include "fast_iterative.h"
#include "point_triangle_distance.h"
#include "vertex_map.h"
#include "parallel_exchange.h"

#include <cutfemx/cut/cut.h>
#include <cutfemx/distance/stl/cell_triangle_map.h>
#include <cutfemx/mesh/cut_mesh.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace cutfemx::distance {

/// Surface mesh structure for reinitialization
/// Supports both edges (2D) and triangles (3D)
template <typename Real>
struct SurfaceMesh
{
    // Vertex coordinates: layout [x0,y0,z0, x1,y1,z1, ...] (always 3D storage)
    std::vector<Real> X;
    
    // Element connectivity (edges or triangles)
    // For edges: length = 2*nE, indices into X vertices
    // For triangles: length = 3*nT, indices into X vertices
    std::vector<std::int32_t> conn;
    
    // Parent background cell for each surface element
    std::vector<std::int32_t> parent_cell;
    
    // Vertices per element (2 for edges, 3 for triangles)
    int verts_per_elem = 3;
    
    std::int32_t nV() const { return static_cast<std::int32_t>(X.size()/3); }
    std::int32_t nE() const { return static_cast<std::int32_t>(conn.size()/verts_per_elem); }
};

/// Extract surface mesh from level set zero-contour using CutCells
/// @param phi Level set function
/// @return SurfaceMesh containing the interface elements
template <typename Real>
SurfaceMesh<Real> extract_surface_from_levelset(
    const dolfinx::fem::Function<Real>& phi)
{
    SurfaceMesh<Real> surface;
    
    auto mesh = phi.function_space()->mesh();
    const int tdim = mesh->topology()->dim();
    const int gdim = mesh->geometry().dim();
    
    if (tdim != 2 && tdim != 3) {
        throw std::invalid_argument(
            "distance::reinitialize currently supports 2D and 3D meshes");
    }
    
    // Set vertices per element based on dimension
    surface.verts_per_elem = (tdim == 3) ? 3 : 2;  // triangles for 3D, edges for 2D
    
    // Build cut data through the public CutFEMx cut facade and use the new
    // cut-layer locator for the zero-contour cells.
    // "phi=0" marks cells where the level set changes sign
    auto phi_handle = std::shared_ptr<const dolfinx::fem::Function<Real>>(
        &phi, [](const dolfinx::fem::Function<Real>*) {});
    auto cut_data = cutfemx::cut<Real>(phi_handle);
    std::vector<std::int32_t> cut_cells =
        cutfemx::locate_entities<Real>(cut_data, "phi=0");
    
    if (cut_cells.empty()) {
        std::cout << "Warning: No cut cells found for reinitialization\n";
        return surface;
    }
    
    // The cut layer currently builds CutData with triangulate_cut_parts=true,
    // so the zero-contour mesh is intervals in 2D and triangles in 3D.
    cutfemx::mesh::CutMesh<Real> cut_mesh =
        cutfemx::create_cut_mesh<Real>(cut_data, "phi=0", "cut_only");
    
    if (!cut_mesh._cut_mesh || cut_mesh._parent_index.empty()) {
        return surface;
    }

    auto interface_mesh = cut_mesh._cut_mesh;
    const int interface_tdim = interface_mesh->topology()->dim();
    const int interface_gdim = interface_mesh->geometry().dim();
    if (interface_tdim != tdim - 1) {
        throw std::runtime_error(
            "cutfemx::create_cut_mesh returned a mesh with unexpected dimension");
    }
    if (interface_gdim != gdim) {
        throw std::runtime_error(
            "cutfemx::create_cut_mesh returned a mesh with unexpected geometry dimension");
    }

    auto cut_topology = interface_mesh->topology();
    interface_mesh->topology_mutable()->create_connectivity(interface_tdim, 0);
    auto cell_to_vertex = cut_topology->connectivity(interface_tdim, 0);
    auto cell_map = cut_topology->index_map(interface_tdim);
    auto vertex_map = cut_topology->index_map(0);
    if (!cell_to_vertex || !cell_map || !vertex_map) {
        throw std::runtime_error("Cut mesh topology is incomplete");
    }

    const std::int32_t num_interface_cells =
        static_cast<std::int32_t>(cell_map->size_local() + cell_map->num_ghosts());
    const std::int32_t num_interface_vertices =
        static_cast<std::int32_t>(vertex_map->size_local()
                                  + vertex_map->num_ghosts());
    if (cut_mesh._parent_index.size()
        != static_cast<std::size_t>(num_interface_cells)) {
        throw std::runtime_error(
            "Cut mesh parent-cell map does not match the cut-mesh cells");
    }

    surface.X.assign(static_cast<std::size_t>(3 * num_interface_vertices),
                     Real(0));
    const std::span<const Real> x = interface_mesh->geometry().x();
    auto x_dofmap = interface_mesh->geometry().dofmaps().front();
    for (std::int32_t c = 0; c < num_interface_cells; ++c) {
        auto cell_vertices = cell_to_vertex->links(c);
        if (cell_vertices.size()
            != static_cast<std::size_t>(surface.verts_per_elem)) {
            throw std::runtime_error(
                "Cut mesh cell arity does not match the zero-contour dimension");
        }

        for (std::size_t i = 0; i < cell_vertices.size(); ++i) {
            const std::int32_t vertex = cell_vertices[i];
            if (vertex < 0 || vertex >= num_interface_vertices) {
                throw std::runtime_error("Cut mesh cell has invalid vertex index");
            }

            const std::int32_t geom_dof = x_dofmap(c, i);
            for (int d = 0; d < gdim; ++d) {
                surface.X[static_cast<std::size_t>(3 * vertex + d)] =
                    x[static_cast<std::size_t>(3 * geom_dof + d)];
            }

            surface.conn.push_back(vertex);
        }
        surface.parent_cell.push_back(
            cut_mesh._parent_index[static_cast<std::size_t>(c)]);
    }
    
    return surface;
}

/// Build CellTriangleMap from SurfaceMesh (uses parent_cell info)
/// This is more efficient than BVH since we already know parent cells
template <typename Real>
CellTriangleMap build_map_from_surface(
    const dolfinx::mesh::Mesh<Real>& mesh,
    const SurfaceMesh<Real>& surface)
{
    CellTriangleMap map;
    
    const std::int32_t num_cells = mesh.topology()->index_map(mesh.topology()->dim())->size_local()
                                 + mesh.topology()->index_map(mesh.topology()->dim())->num_ghosts();
    
    map.offsets.resize(num_cells + 1, 0);
    
    // Count elements per cell
    for (std::int32_t parent : surface.parent_cell) {
        if (parent >= 0 && parent < num_cells) {
            map.offsets[parent + 1]++;
        }
    }
    
    // Prefix sum
    for (std::int32_t i = 1; i <= num_cells; ++i) {
        map.offsets[i] += map.offsets[i - 1];
    }
    
    // Fill data array
    map.data.resize(surface.nE());
    std::vector<std::int32_t> counters(num_cells, 0);
    
    for (std::int32_t e = 0; e < surface.nE(); ++e) {
        std::int32_t parent = surface.parent_cell[e];
        if (parent >= 0 && parent < num_cells) {
            std::int32_t idx = map.offsets[parent] + counters[parent]++;
            map.data[idx] = e;
        }
    }
    
    return map;
}

/// Compute point-to-surface-element distance
/// Handles both edges (2D) and triangles (3D)
template <typename Real>
Real point_surface_element_distance(
    const Real* P, 
    const SurfaceMesh<Real>& surface,
    std::int32_t elem_idx)
{
    int vpe = surface.verts_per_elem;
    
    if (vpe == 3) {
        // Triangle
        std::int32_t v0 = surface.conn[3 * elem_idx];
        std::int32_t v1 = surface.conn[3 * elem_idx + 1];
        std::int32_t v2 = surface.conn[3 * elem_idx + 2];
        
        const Real* A = &surface.X[3 * v0];
        const Real* B = &surface.X[3 * v1];
        const Real* C = &surface.X[3 * v2];
        
        return point_triangle_distance(P, A, B, C);
    } else {
        // Edge
        std::int32_t v0 = surface.conn[2 * elem_idx];
        std::int32_t v1 = surface.conn[2 * elem_idx + 1];
        
        const Real* A = &surface.X[3 * v0];
        const Real* B = &surface.X[3 * v1];
        
        return point_segment_distance(P, A, B);
    }
}

/// Initialize near-field distances from surface mesh
/// @param mesh Background mesh
/// @param surface Surface mesh (edges or triangles)
/// @param cell_map Mapping from cells to surface elements
/// @param dist Output distance array (vertex-indexed)
/// @param opt FMM options
template <typename Real>
void init_nearfield_from_surface(
    const dolfinx::mesh::Mesh<Real>& mesh,
    const SurfaceMesh<Real>& surface,
    const CellTriangleMap& cell_map,
    std::span<Real> dist,
    const FMMOptions& opt)
{
    // Build vertex-geometry mapping
    VertexMapCache<Real> vmap;
    vmap.build(mesh);
    
    auto topology = mesh.topology_mutable();
    topology->create_connectivity(topology->dim(), 0);
    auto cell_to_vertex = topology->connectivity(topology->dim(), 0);
    
    const std::int32_t num_cells = cell_map.num_cells();
    
    for (std::int32_t c = 0; c < num_cells; ++c) {
        std::int32_t start = cell_map.offsets[c];
        std::int32_t end = cell_map.offsets[c + 1];
        
        if (start == end) continue;  // No surface elements in this cell
        
        auto cell_verts = cell_to_vertex->links(c);
        
        for (std::int32_t v : cell_verts) {
            if (!vmap.has_geom(v)) continue;
            const Real* p = vmap.coords(v);
            
            // Find minimum distance to all surface elements in this cell
            for (std::int32_t i = start; i < end; ++i) {
                std::int32_t elem_idx = cell_map.data[i];
                Real d = point_surface_element_distance(p, surface, elem_idx);
                
                if (d < dist[v]) {
                    dist[v] = d;
                }
            }
        }
    }
}

/// Reinitialize level set function as signed distance function
/// Main entry point for reinitialization
/// @param phi Level set function to reinitialize (modified in-place)
/// @param opt FMM options
template <typename Real>
void reinitialize(
    dolfinx::fem::Function<Real>& phi,
    const FMMOptions& opt = FMMOptions{})
{
    auto mesh = phi.function_space()->mesh();
    const int tdim = mesh->topology()->dim();
    
    MPI_Comm comm = mesh->comm();
    int mpi_rank = dolfinx::MPI::rank(comm);
    
    if (mpi_rank == 0) {
        spdlog::info("Reinitializing level set ({}D)...", tdim);
    }
    
    // 1. Store original signs for all vertices
    std::span<const Real> phi_vals = phi.x()->array();
    std::vector<int> original_signs(phi_vals.size());
    for (std::size_t i = 0; i < phi_vals.size(); ++i) {
        original_signs[i] = (phi_vals[i] >= 0) ? 1 : -1;
    }
    
    // 2. Extract surface mesh from zero-contour
    auto surface = extract_surface_from_levelset(phi);
    
    if (surface.nE() == 0) {
        if (mpi_rank == 0) {
            std::cout << "Warning: Empty surface, skipping reinitialization\n";
        }
        return;
    }
    
    if (mpi_rank == 0) {
        spdlog::info("Extracted surface: {} elements, {} vertices", surface.nE(), surface.nV());
    }
    
    // 3. Build cell-element map from parent information
    auto cell_map = build_map_from_surface(*mesh, surface);
    
    // 4. Initialize distance array
    auto topology = mesh->topology();
    auto vertex_map = topology->index_map(0);
    std::int32_t num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
    
    std::vector<Real> dist(num_vertices, opt.inf_value);
    
    // 5. Compute near-field distances
    init_nearfield_from_surface(*mesh, surface, cell_map, std::span<Real>(dist), opt);
    
    // 6. Collect seeds and run FIM
    std::vector<std::int32_t> seeds;
    std::int32_t num_owned = vertex_map->size_local();
    for (std::int32_t v = 0; v < num_owned; ++v) {
        if (dist[v] < opt.inf_value) {
            seeds.push_back(v);
        }
    }
    
    if (mpi_rank == 0) {
        spdlog::info("Near-field seeds: {}", seeds.size());
    }
    
    // Run FIM to propagate distances
    compute_distance_fim(*mesh, std::span<Real>(dist), seeds, opt);
    
    // 7. Copy to function and apply original signs
    VertexMapCache<Real> vmap;
    vmap.build(*mesh, phi.function_space().get());
    
    std::span<Real> func_vals = phi.x()->array();
    for (std::int32_t v = 0; v < num_vertices; ++v) {
        std::int32_t dof = vmap.vert_to_dof[v];
        if (dof >= 0 && dof < (std::int32_t)func_vals.size()) {
            // Apply original sign
            Real s = (original_signs[dof] >= 0) ? Real(1) : Real(-1);
            func_vals[dof] = s * std::abs(dist[v]);
        }
    }
    
    // 8. Sync ghost values
    phi.x()->scatter_fwd();
    
    if (mpi_rank == 0) {
        spdlog::info("Reinitialization complete.");
    }
}

} // namespace cutfemx::distance
