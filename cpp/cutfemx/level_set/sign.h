#pragma once

// Sign determination for level sets from scanned (potentially non-watertight) surfaces
// Three methods from simple to complex:
//   1) LocalNormalBand - fast sign via closest triangle normal projection
//   2) ComponentAnchor - global sign via mesh connectivity barriers
//   3) WindingNumber   - robust global sign via generalized winding number

#include <dolfinx/mesh/Mesh.h>
#include <cutfemx/mesh/stl/stl_surface.h>
#include <cutfemx/mesh/stl/cell_triangle_map.h>
#include <cutfemx/level_set/pt_tri_distance.h>
#include <cutfemx/level_set/parallel_min_exchange.h>
#include "geom_map.h"
#include <dolfinx/common/log.h>
#include <sstream>

#include "sign_options.h"
#include "sign_region.h"
#include "winding.h"

#include <span>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <queue>

namespace cutfemx::level_set {

namespace sign_detail {

/// Compute triangle normal (not normalized)
template <typename Real>
void triangle_normal(const Real* v0, const Real* v1, const Real* v2, Real* normal)
{
    // Edge vectors
    Real e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    Real e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    
    // Cross product e1 × e2
    normal[0] = e1[1]*e2[2] - e1[2]*e2[1];
    normal[1] = e1[2]*e2[0] - e1[0]*e2[2];
    normal[2] = e1[0]*e2[1] - e1[1]*e2[0];
}

/// Propagate sign from seeded vertices to all vertices via BFS
/// Vertices with |phi| already set (non-zero) are seeds
template <typename Real>
void propagate_sign_bfs(
    std::span<Real> phi,
    const std::vector<bool>& has_sign,
    const dolfinx::mesh::Mesh<Real>& mesh,
    const std::vector<std::int32_t>& vert_to_dof)
{
    auto topology = mesh.topology();
    const int tdim = topology->dim();
    auto vertex_to_cell = topology->connectivity(0, tdim);
    auto cell_to_vertex = topology->connectivity(tdim, 0);
    auto vertex_map = topology->index_map(0);
    
    const std::int32_t num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
    
    // Track which vertices have sign assigned
    std::vector<bool> visited = has_sign;
    
    // BFS queue: start from vertices that have sign
    std::queue<std::int32_t> queue;
    for (std::int32_t v = 0; v < num_vertices; ++v) {
        if (has_sign[v]) {
            queue.push(v);
        }
    }
    
    // BFS propagation
    while (!queue.empty()) {
        std::int32_t v = queue.front();
        queue.pop();
        
        std::int32_t dof_v = vert_to_dof[v];
        if (dof_v < 0) continue;
        
        Real sign_v = (phi[dof_v] < 0) ? Real(-1) : Real(1);
        
        // Visit all neighbors through incident cells
        auto cell_links = vertex_to_cell->links(v);
        for (std::int32_t c : cell_links) {
            auto c_verts = cell_to_vertex->links(c);
            for (std::int32_t n : c_verts) {
                if (!visited[n]) {
                    visited[n] = true;
                    // Propagate sign: farfield vertex inherits sign from neighbor
                    std::int32_t dof_n = vert_to_dof[n];
                    if (dof_n >= 0) {
                        phi[dof_n] = sign_v * std::abs(phi[dof_n]);
                        queue.push(n);
                    }
                }
            }
        }
    }
}

} // namespace sign_detail

#include <dolfinx/fem/Function.h>



// ... (enums and structs remain same)

// ... (sign_detail remains same)

/// Apply sign to unsigned distance using Method 1 (LocalNormalBand)
/// 
/// For each nearfield vertex, compute sign based on dot product of displacement
/// with closest triangle normal. Then propagate sign to farfield vertices.
///
/// @param[in,out] phi  Distance function (input unsigned, output signed)
/// @param[in] soup     Surface triangle soup
/// @param[in] map      Cell-to-triangle mapping
/// @param[in] closest_tri  Closest triangle index per vertex (-1 if not set)
/// @param[in] opt      Sign options
template <typename Real>
void apply_sign_local_normal(
    dolfinx::fem::Function<Real>& phi,
    const ::cutfemx::mesh::TriSoup<Real>& soup,
    const ::cutfemx::mesh::CellTriangleMap& map,
    const std::vector<std::int32_t>& closest_tri,
    const SignOptions& opt = {})
{
    auto mesh = phi.function_space()->mesh();
    auto topology = mesh->topology();
    const int tdim = topology->dim();
    auto vertex_map = topology->index_map(0);
    auto cell_to_vertex = topology->connectivity(tdim, 0);
    
    // Get span for read/write access
    std::span<Real> phi_span = phi.x()->mutable_array();
    
    const std::int32_t num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
    
    // Build centralized mapping
    VertexMapCache<Real> vmap;
    vmap.build(*mesh, phi.function_space().get());
    
    // Alias for compatibility with existing code structure
    const auto& vert_to_dof = vmap.vert_to_dof;
    const auto& vert_to_geom = vmap.vert_to_geom;
    
    // Track which vertices have sign assigned
    std::vector<bool> has_sign(num_vertices, false);
    
    // Phase 1: Assign sign to nearfield vertices (those with closest_tri)
    for (std::int32_t v = 0; v < num_vertices; ++v)
    {
        if (closest_tri[v] < 0) continue;
        
        std::int32_t dof_v = vert_to_dof[v];
        if (dof_v < 0) continue;
        
        if (opt.band_max_dist > 0 && phi_span[dof_v] > opt.band_max_dist) continue;
        
        if (!vmap.has_geom(v)) continue;
        const Real* x_v = vmap.coords(v);
        
        std::int32_t t = closest_tri[v];
        std::int32_t i0 = soup.tri[3*t];
        std::int32_t i1 = soup.tri[3*t + 1];
        std::int32_t i2 = soup.tri[3*t + 2];
        
        const Real* v0 = &soup.X[3*i0];
        const Real* v1 = &soup.X[3*i1];
        const Real* v2 = &soup.X[3*i2];
        
        Real closest_pt[3];
        point_triangle_distance_sq(x_v, v0, v1, v2, closest_pt);
        
        Real disp[3] = {x_v[0] - closest_pt[0], x_v[1] - closest_pt[1], x_v[2] - closest_pt[2]};
        
        Real normal[3];
        sign_detail::triangle_normal(v0, v1, v2, normal);
        
        Real dot = detail::dot3(disp, normal);
        
        if (dot < 0) {
            phi_span[dof_v] = -phi_span[dof_v];
        }
        has_sign[v] = true;
    }
    
    // Phase 2: Propagate sign from nearfield to farfield via BFS
    if (opt.propagate_sign) {
        sign_detail::propagate_sign_bfs(phi_span, has_sign, *mesh, vert_to_dof);
        
        // Phase 3: MPI sync using native dolfinx scatter
        // This ensures ghosts have the correct sign from their owner
        phi.x()->scatter_fwd();
    }
}

/// Main API: Apply sign to unsigned distance
template <typename Real>
void apply_sign(
    dolfinx::fem::Function<Real>& phi,
    const ::cutfemx::mesh::TriSoup<Real>& soup,
    const ::cutfemx::mesh::CellTriangleMap& map,
    const std::vector<std::int32_t>& closest_tri,
    const SignOptions& opt = {})
{
    switch (opt.mode) {
        case SignMode::LocalNormalBand:
            apply_sign_local_normal(phi, soup, map, closest_tri, opt);
            break;
        case SignMode::ComponentAnchor:
        {
            // Method 2: 
            // 1. Mark cut facets
            auto mesh = phi.function_space()->mesh();
            auto cut_facets = mark_cut_facets(*mesh, map, soup);
            
            int local_cuts = 0; 
            for(bool b : cut_facets) if(b) local_cuts++;
            int global_cuts_pre = 0;
            MPI_Allreduce(&local_cuts, &global_cuts_pre, 1, MPI_INT, MPI_SUM, mesh->comm());
            if (dolfinx::MPI::rank(mesh->comm()) == 0) {
                 spdlog::info("Cut Facets Pre-Sync: {}", global_cuts_pre);
            }

            // Sync cut facets (OR)
            auto topology = mesh->topology();
            int tdim = topology->dim();
            int fdim = tdim - 1;
            or_exchange_bool(*topology->index_map(fdim), cut_facets);
            
            local_cuts = 0;
            for(bool b : cut_facets) if(b) local_cuts++;
            int global_cuts_post = 0;
            MPI_Allreduce(&local_cuts, &global_cuts_post, 1, MPI_INT, MPI_SUM, mesh->comm());
            if (dolfinx::MPI::rank(mesh->comm()) == 0) {
                 spdlog::info("Cut Facets Post-Sync: {}", global_cuts_post);
            }
            
            // 2. Apply sign via flooding
            apply_sign_flood_fill(phi, *mesh, cut_facets);
            break;
        }
        case SignMode::WindingNumber:
            apply_sign_winding_number(phi, soup, opt);
            break;
    }
}

} // namespace cutfemx::level_set
