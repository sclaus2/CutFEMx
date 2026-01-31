#pragma once

// Level Set Reinitialization: Reconstruct signed distance function from level set
// Uses CutCells to extract the zero-contour, then FMM to compute distance

#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/Mesh.h>

#include <cutcells/cut_mesh.h>
#include <cutcells/cell_flags.h>

#include "cut_entities.h"
#include "locate_entities.h"
#include "fmm.h"
#include "pt_tri_distance.h"
#include "geom_map.h"
#include "parallel_min_exchange.h"

#include <cutfemx/mesh/cell_triangle_map.h>

#include <vector>
#include <span>
#include <cmath>
#include <iostream>
#include <numeric>

namespace cutfemx::level_set {

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
    std::shared_ptr<dolfinx::fem::Function<Real>> phi_ptr)
{
    SurfaceMesh<Real> surface;
    
    auto mesh = phi_ptr->function_space()->mesh();
    const int tdim = mesh->topology()->dim();
    const int gdim = mesh->geometry().dim();
    
    // Set vertices per element based on dimension
    surface.verts_per_elem = (tdim == 3) ? 3 : 2;  // triangles for 3D, edges for 2D
    
    // Find cut cells using locate_entities
    // "phi=0" marks cells where the level set changes sign
    auto phi_const = std::const_pointer_cast<const dolfinx::fem::Function<Real>>(phi_ptr);
    std::vector<std::int32_t> cut_cells = locate_entities<Real>(
        phi_const, tdim, "phi=0");
    
    if (cut_cells.empty()) {
        std::cout << "Warning: No cut cells found for reinitialization\n";
        return surface;
    }
    
    // Call CutCells to extract interface
    auto cut_result = cut_entities<Real>(phi_const, cut_cells, tdim, "phi=0");
    
    // Build surface mesh from CutCells output
    // Each cut cell produces interface elements (edges or triangles)
    std::vector<Real> vertices;
    std::vector<std::int32_t> connectivity;
    std::vector<std::int32_t> parent_cells;
    
    int vertex_offset = 0;
    
    for (std::size_t i = 0; i < cut_result._cut_cells.size(); ++i) {
        const auto& cut_cell = cut_result._cut_cells[i];
        std::int32_t parent = cut_result._parent_map[i];
        
        // Skip empty cut cells
        if (cut_cell._vertex_coords.empty()) continue;
        
        // Get interface elements from cut cell
        // The CutCell stores sub-elements with types and connectivity
        int num_verts_in_cell = cut_cell._vertex_coords.size() / gdim;
        
        // Copy vertex coordinates (pad to 3D if needed)
        for (int v = 0; v < num_verts_in_cell; ++v) {
            for (int d = 0; d < gdim; ++d) {
                vertices.push_back(cut_cell._vertex_coords[v * gdim + d]);
            }
            // Pad with zeros if 2D
            for (int d = gdim; d < 3; ++d) {
                vertices.push_back(Real(0));
            }
        }
        
        // Process sub-elements
        // _connectivity is std::vector<std::vector<int>>
        // _types[t] corresponds to _connectivity[t]
        for (std::size_t t = 0; t < cut_cell._connectivity.size(); ++t) {
            auto type = cut_cell._types[t];
            
            // Only process interface elements (edges in 2D, triangles in 3D)
            bool is_interface = false;
            if (tdim == 3 && type == cutcells::cell::type::triangle) {
                is_interface = true;
            } else if (tdim == 2 && type == cutcells::cell::type::interval) {
                is_interface = true;
            }
            
            if (is_interface) {
                // Add connectivity with offset
                for (auto local_v : cut_cell._connectivity[t]) {
                    connectivity.push_back(vertex_offset + local_v);
                }
                parent_cells.push_back(parent);
            }
        }
        
        vertex_offset += num_verts_in_cell;
    }
    
    surface.X = std::move(vertices);
    surface.conn = std::move(connectivity);
    surface.parent_cell = std::move(parent_cells);
    
    return surface;
}

/// Build CellTriangleMap from SurfaceMesh (uses parent_cell info)
/// This is more efficient than BVH since we already know parent cells
template <typename Real>
mesh::CellTriangleMap build_map_from_surface(
    const dolfinx::mesh::Mesh<Real>& mesh,
    const SurfaceMesh<Real>& surface)
{
    mesh::CellTriangleMap map;
    
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
    const mesh::CellTriangleMap& cell_map,
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
        std::cout << "Reinitializing level set (" << (tdim == 3 ? "3D" : "2D") << ")...\n";
    }
    
    // 1. Store original signs for all vertices
    auto phi_vals = phi.x()->array();
    std::vector<int> original_signs(phi_vals.size());
    for (std::size_t i = 0; i < phi_vals.size(); ++i) {
        original_signs[i] = (phi_vals[i] >= 0) ? 1 : -1;
    }
    
    // 2. Extract surface mesh from zero-contour
    // Need non-const shared_ptr for the function
    auto phi_ptr = std::const_pointer_cast<dolfinx::fem::Function<Real>>(
        std::shared_ptr<dolfinx::fem::Function<Real>>(&phi, [](auto*){}) // non-owning
    );
    // Actually, just pass the address directly - require caller to manage lifetime
    auto surface = extract_surface_from_levelset(
        std::shared_ptr<dolfinx::fem::Function<Real>>(&phi, [](auto*){}));
    
    if (surface.nE() == 0) {
        if (mpi_rank == 0) {
            std::cout << "Warning: Empty surface, skipping reinitialization\n";
        }
        return;
    }
    
    if (mpi_rank == 0) {
        std::cout << "Extracted surface: " << surface.nE() << " elements, " 
                  << surface.nV() << " vertices\n";
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
        std::cout << "Near-field seeds: " << seeds.size() << "\n";
    }
    
    // Run FIM to propagate distances
    compute_distance_fim(*mesh, std::span<Real>(dist), seeds, opt);
    
    // 7. Copy to function and apply original signs
    VertexMapCache<Real> vmap;
    vmap.build(*mesh, phi.function_space().get());
    
    auto func_vals = phi.x()->mutable_array();
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
        std::cout << "Reinitialization complete.\n";
    }
}

} // namespace cutfemx::level_set
