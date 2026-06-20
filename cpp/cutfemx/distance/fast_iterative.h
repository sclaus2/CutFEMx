// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

// =========================================================================
// Fast Iterative Method (FIM) Orchestrator
// Robust for parallel execution on unstructured grids.
// Uses "Active Set" label-correcting approach rather than Dijkstra-like ordering.
// =========================================================================

#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/common/MPI.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <dolfinx/common/Timer.h>
#include <array>
#include <span>

#include "eikonal_update.h"
#include "parallel_exchange.h"
#include "point_triangle_distance.h"
#include "vertex_map.h"
#include "stl/cell_triangle_map.h"
#include "stl/distribute.h"

namespace cutfemx::distance {

enum class NearFieldMode { TRIANGLES, IMPLICIT };

// Renamed from FIMOptions to FMMOptions for backward compatibility with demo
struct FMMOptions {
    NearFieldMode mode = NearFieldMode::TRIANGLES;
    
    // FIM specific
    int max_iter = 1000;
    double tol = 1e-10; 
    double inf_value = 1e30;
    int sync_interval = 2; // Exchange every k iterations

    // Legacy / Compat fields (ignored by FIM or mapped)
    int band_layers = 2; 
    double restart_stride = 0.1;
    int local_relax_batch = 100;
};

using FIMOptions = FMMOptions;

template <typename Real>
struct FIMTransportPayload
{
    std::span<Real> speed;
    std::span<Real> normals;
    int normal_dim = 3;

    bool valid(std::int32_t num_vertices) const
    {
        return speed.size() >= static_cast<std::size_t>(num_vertices)
            && normals.size()
                   >= static_cast<std::size_t>(num_vertices)
                          * static_cast<std::size_t>(normal_dim)
            && normal_dim > 0;
    }
};

template <typename F>
void for_each_virtual_simplex(int tdim, std::span<const std::int32_t> c_verts,
                              F&& f)
{
    const int num_verts = static_cast<int>(c_verts.size());

    auto emit = [&](std::initializer_list<int> local)
    {
        std::array<std::int32_t, 4> simplex{-1, -1, -1, -1};
        int n = 0;
        for (int i : local)
            simplex[static_cast<std::size_t>(n++)] = c_verts[static_cast<std::size_t>(i)];
        f(std::span<const std::int32_t>(simplex.data(), static_cast<std::size_t>(n)));
    };

    if (tdim == 2 && num_verts == 3)
    {
        emit({0, 1, 2});
    }
    else if (tdim == 2 && num_verts == 4)
    {
        // Basix quadrilateral order is (0, 1, 2, 3). Split along 0--3.
        emit({0, 1, 3});
        emit({0, 3, 2});
    }
    else if (tdim == 3 && num_verts == 4)
    {
        emit({0, 1, 2, 3});
    }
    else if (tdim == 3 && num_verts == 8)
    {
        // Freudenthal split of a Basix-order hexahedron along body diagonal 0--7.
        emit({0, 1, 3, 7});
        emit({0, 1, 5, 7});
        emit({0, 2, 3, 7});
        emit({0, 2, 6, 7});
        emit({0, 4, 5, 7});
        emit({0, 4, 6, 7});
    }
}

template <typename Real>
void assign_payload_from_best_source(FIMTransportPayload<Real>* payload,
                                     std::int32_t target,
                                     std::span<const std::int32_t> sources,
                                     std::span<Real> dist)
{
    if (!payload)
        return;

    std::int32_t best = -1;
    Real best_value = std::numeric_limits<Real>::max();
    for (std::int32_t source : sources)
    {
        if (source >= 0 && dist[static_cast<std::size_t>(source)] < best_value)
        {
            best = source;
            best_value = dist[static_cast<std::size_t>(source)];
        }
    }
    if (best < 0)
        return;

    payload->speed[static_cast<std::size_t>(target)]
        = payload->speed[static_cast<std::size_t>(best)];
    for (int d = 0; d < payload->normal_dim; ++d)
    {
        payload->normals[static_cast<std::size_t>(target) * payload->normal_dim
                         + static_cast<std::size_t>(d)]
            = payload->normals[static_cast<std::size_t>(best) * payload->normal_dim
                               + static_cast<std::size_t>(d)];
    }
}

// Low-level FIM Solver
template <typename Real>
void compute_distance_fim(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<Real> dist,
    const std::vector<std::int32_t>& seeds,
    const FMMOptions& opt = {},
    FIMTransportPayload<Real>* payload = nullptr)
{
    // 0. Setup
    // ------------------------------------------------------------------------
    MPI_Comm comm = mesh.comm();
    const int mpi_rank = dolfinx::MPI::rank(comm);
    const int tdim = mesh.topology()->dim();
    
    // Topology maps
    auto topology = mesh.topology();
    // Ensure connectivity
    if (!topology->connectivity(0, tdim)) {
        mesh.topology_mutable()->create_connectivity(0, tdim);
        mesh.topology_mutable()->create_connectivity(tdim, 0);
    }
    auto vertex_to_cell = topology->connectivity(0, tdim);
    auto cell_to_vertex = topology->connectivity(tdim, 0);
    
    const std::int32_t num_vertices = vertex_to_cell->num_nodes(); // owned + ghosts
    const auto& vertex_map = topology->index_map(0);
    std::int32_t num_owned = vertex_map->size_local();
    
    // CRITICAL: Build mapping from topology vertex index to geometry DOF index
    // Use centralized cache
    VertexMapCache<Real> vmap;
    vmap.build(mesh);
    
    // Direct pointer access for coordinates via cache
    auto get_vertex_coords = [&vmap](std::int32_t v) -> const Real* {
        return vmap.coords(v);
    };
    
    // 1. Initialization (NearField)
    // ------------------------------------------------------------------------
    // Allow ALL vertices (Owned + Ghost) to be active.
    // This allows ghosts updated by local neighbors to immediately propagate
    // back to other local neighbors or ghosts.
    std::vector<bool> active_mask(num_vertices, false); 
    std::vector<std::int32_t> active_queue;
    active_queue.reserve(num_vertices / 4);
    
    // Initial sync to ensure consistency (e.g. if seeds are ghosts)
    std::vector<Real> dist_before(dist.begin(), dist.end());
    if (payload && payload->valid(num_vertices))
        min_exchange_vertices_payload(*vertex_map, dist, payload->speed,
                                      payload->normals, payload->normal_dim,
                                      static_cast<Real>(opt.inf_value));
    else
        min_exchange_vertices(*vertex_map, dist, static_cast<Real>(opt.inf_value));
    
    int initial_activated = 0;
    
    // Populate active set - scan ALL vertices (owned + ghost)
    // If a node is finite, it should be active to propagate.
    for (std::int32_t v = 0; v < num_vertices; ++v) {
        if (dist[v] < opt.inf_value) {
            if (!active_mask[v]) {
                active_mask[v] = true;
                active_queue.push_back(v);
                initial_activated++;
            }
        }
    }
    
    if (mpi_rank == 0) spdlog::info("FIM Init: {} active nodes.", initial_activated);
    
    // 2. FIM Iteration Loop
    // ------------------------------------------------------------------------
    dolfinx::common::Timer timer("FIM Iteration");
    
    for (int iter = 0; iter < opt.max_iter; ++iter)
    {
        // A. Local Relaxation Phase
        std::vector<std::int32_t> next_active_queue;
        next_active_queue.reserve(active_queue.size());
        
        // Track max change for convergence (only on OWNED nodes)
        Real local_max_change = 0.0;
        
        for (std::int32_t u : active_queue)
        {
            active_mask[u] = false; 
            
            auto cell_links = vertex_to_cell->links(u);
            const Real* x_u = get_vertex_coords(u);
            Real d_u = dist[u];
            
            for (std::int32_t c : cell_links)
            {
                auto c_verts = cell_to_vertex->links(c);

                for_each_virtual_simplex(
                    tdim, c_verts,
                    [&](std::span<const std::int32_t> simplex)
                    {
                        int u_idx = -1;
                        for (std::size_t k = 0; k < simplex.size(); ++k)
                        {
                            if (simplex[k] == u)
                            {
                                u_idx = static_cast<int>(k);
                                break;
                            }
                        }
                        if (u_idx == -1)
                            return;

                        auto accept_update =
                            [&](std::int32_t target, Real new_d,
                                std::span<const std::int32_t> sources)
                        {
                            if (new_d < dist[target] - 1e-13)
                            {
                                Real change = dist[target] - new_d;
                                dist[target] = new_d;
                                assign_payload_from_best_source(payload, target,
                                                                sources, dist);

                                if (target < num_owned
                                    && change > local_max_change)
                                    local_max_change = change;

                                if (!active_mask[target])
                                {
                                    active_mask[target] = true;
                                    next_active_queue.push_back(target);
                                }
                            }
                        };

                        if (simplex.size() == 3)
                        {
                            std::int32_t v = -1;
                            std::int32_t w = -1;
                            for (std::size_t i = 0; i < simplex.size(); ++i)
                            {
                                if (static_cast<int>(i) == u_idx)
                                    continue;
                                if (v < 0)
                                    v = simplex[i];
                                else
                                    w = simplex[i];
                            }

                            auto attempt_triangle_target =
                                [&](std::int32_t target, std::int32_t other)
                            {
                                std::array<std::int32_t, 2> sources{u, other};
                                Real new_d = opt.inf_value;
                                if (dist[other] < opt.inf_value)
                                    new_d = update_2pt(get_vertex_coords(target),
                                                       x_u, d_u,
                                                       get_vertex_coords(other),
                                                       dist[other]);
                                else
                                {
                                    sources[1] = u;
                                    new_d = update_1pt(get_vertex_coords(target),
                                                       x_u, d_u);
                                }
                                accept_update(target, new_d, sources);
                            };

                            attempt_triangle_target(v, w);
                            attempt_triangle_target(w, v);
                        }
                        else if (simplex.size() == 4)
                        {
                            for (std::size_t i = 0; i < simplex.size(); ++i)
                            {
                                if (static_cast<int>(i) == u_idx)
                                    continue;
                                const std::int32_t target = simplex[i];

                                std::array<std::int32_t, 2> others{-1, -1};
                                int oi = 0;
                                for (std::size_t j = 0; j < simplex.size(); ++j)
                                {
                                    if (static_cast<int>(j) != u_idx && j != i)
                                        others[static_cast<std::size_t>(oi++)]
                                            = simplex[j];
                                }
                                const std::int32_t v = others[0];
                                const std::int32_t w = others[1];

                                std::array<std::int32_t, 3> sources1{u, u, u};
                                accept_update(target,
                                              update_1pt(get_vertex_coords(target),
                                                         x_u, d_u),
                                              std::span<const std::int32_t>(
                                                  sources1.data(), 1));

                                if (dist[v] < opt.inf_value)
                                {
                                    std::array<std::int32_t, 2> sources{u, v};
                                    accept_update(
                                        target,
                                        update_2pt(get_vertex_coords(target),
                                                   x_u, d_u,
                                                   get_vertex_coords(v), dist[v]),
                                        sources);
                                }
                                if (dist[w] < opt.inf_value)
                                {
                                    std::array<std::int32_t, 2> sources{u, w};
                                    accept_update(
                                        target,
                                        update_2pt(get_vertex_coords(target),
                                                   x_u, d_u,
                                                   get_vertex_coords(w), dist[w]),
                                        sources);
                                }
                                if (dist[v] < opt.inf_value
                                    && dist[w] < opt.inf_value)
                                {
                                    std::array<std::int32_t, 3> sources{u, v, w};
                                    accept_update(
                                        target,
                                        update_3pt(get_vertex_coords(target),
                                                   x_u, d_u,
                                                   get_vertex_coords(v), dist[v],
                                                   get_vertex_coords(w), dist[w]),
                                        sources);
                                }
                            }
                        }
                    });
            } 
        } 
        
        active_queue = std::move(next_active_queue);
        
        // B. Synchronization Phase
        // FIX: Sync MORE frequently near convergence to fix boundary artifacts
        int current_sync_interval = opt.sync_interval;
        if (iter >= 50) current_sync_interval = 1;  // Sync every iteration late
        else if (iter >= 20) current_sync_interval = std::min(opt.sync_interval, 2);
        
        if (iter % current_sync_interval == 0)
        {
            std::vector<Real> dist_before_sync(dist.begin(), dist.end()); 
            if (payload && payload->valid(num_vertices))
                min_exchange_vertices_payload(*vertex_map, dist, payload->speed,
                                              payload->normals, payload->normal_dim,
                                              static_cast<Real>(opt.inf_value));
            else
                min_exchange_vertices(*vertex_map, dist, static_cast<Real>(opt.inf_value));
            
            int owned_cnt = 0;
            int ghost_cnt = 0;
            
            // Reactivate ANY vertex that improved (Owned or Ghost)
            // Even ghosts need reactivation because they might have been updated 
            // by the Owner to a better value than we computed locally,
            // and that better value needs to propagate to local neighbors.
            for (std::int32_t v = 0; v < num_vertices; ++v) {
                if (dist[v] < dist_before_sync[v] - 1e-13) {
                     if (v < num_owned) {
                          owned_cnt++;
                          Real chg = dist_before_sync[v] - dist[v];
                          if(chg > local_max_change) local_max_change = chg;
                     } else {
                          ghost_cnt++;
                     }
                     
                     // Activate the changed vertex itself
                     if (!active_mask[v]) {
                        active_mask[v] = true;
                        active_queue.push_back(v);
                     }
                     
                     // IMPORTANT: Also activate 1-ring neighbors through incident cells
                     // This ensures the improved value propagates aggressively
                     auto cell_links_v = vertex_to_cell->links(v);
                     for (std::int32_t c : cell_links_v) {
                         auto c_verts = cell_to_vertex->links(c);
                         for (std::int32_t n : c_verts) {
                             // Only activate owned neighbors (ghosts will be handled by their owner)
                             if (n < num_owned && !active_mask[n]) {
                                 active_mask[n] = true;
                                 active_queue.push_back(n);
                             }
                         }
                     }
                }
            }
            if ((owned_cnt + ghost_cnt > 0) && mpi_rank == 0 && iter%10==0) {
                spdlog::info("Iter {} Sync: Owned={} Ghost={}", iter, owned_cnt, ghost_cnt);
            }
        }
        
        // C. Convergence Check
        // ------------------------------------
        double global_max_change = 0;
        // Proper MPI type for Real (assuming double for now or using dolfinx helper)
        double local_max_double = static_cast<double>(local_max_change);
        MPI_Allreduce(&local_max_double, &global_max_change, 1, MPI_DOUBLE, MPI_MAX, comm);
        
        int local_q_size = active_queue.size();
        int global_q_size = 0;
        MPI_Allreduce(&local_q_size, &global_q_size, 1, MPI_INT, MPI_SUM, comm);
        
        if (global_max_change < opt.tol && global_q_size == 0) {
            if (mpi_rank == 0) spdlog::info("FIM Converged at iter {}", iter);
            break;
        }
    }
    
    // CRITICAL: Final owner→ghost copy to ensure perfect consistency for VTK output
    // The MIN-rule sync may leave ghosts with values slightly different from owners.
    // This strict copy ensures ghost values EXACTLY match their owners.
    if (payload && payload->valid(num_vertices))
        copy_owner_to_ghost_payload(*vertex_map, dist, payload->speed,
                                    payload->normals, payload->normal_dim);
    else
        copy_owner_to_ghost(*vertex_map, dist);
}

// Helper to get keys from map
inline std::vector<std::int32_t> get_cut_cells(const cutfemx::distance::CellTriangleMap& map) {
    std::vector<std::int32_t> cells;
    std::size_t n = map.num_cells();
    for(std::size_t i=0; i<n; ++i) {
        if(map.offsets[i+1] > map.offsets[i]) {
            cells.push_back((std::int32_t)i);
        }
    }
    return cells;
}

// Wrapper for backward compatibility with demo
#include <dolfinx/fem/Function.h>

// Wrapper for backward compatibility with demo (renamed to avoid conflict if needed, or overloaded)
template <typename Real>
void compute_unsigned_distance(
    dolfinx::fem::Function<Real>& dist_func, // Output function
    const ::cutfemx::distance::TriSoup<Real>& soup, // Changed from DistributedSTL to TriSoup
    const ::cutfemx::distance::CellTriangleMap& map,
    const FMMOptions& opt,
    std::vector<std::int32_t>* closest_tri_out = nullptr)  // Optional: output closest triangle per vertex
{
    auto mesh = dist_func.function_space()->mesh();
    const int tdim = mesh->topology()->dim();
    auto vertex_map = mesh->topology()->index_map(0);
    int num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
    
    // Internal vector for FMM (vertex-indexed)
    std::vector<Real> dist(num_vertices, opt.inf_value);
    
    // Track closest triangle if requested
    std::vector<std::int32_t> closest_tri(num_vertices, -1);
    
    // Build vertex -> geometry_dof and dof mapping
    VertexMapCache<Real> vmap;
    vmap.build(*mesh, dist_func.function_space().get());

    auto cell_to_vertex = mesh->topology()->connectivity(tdim, 0);
    
    // Correct iteration over CSR map
    std::size_t num_map_cells = map.num_cells();
    for (std::size_t c = 0; c < num_map_cells; ++c)
    {
        // Start/End indices in data array
        std::int32_t start = map.offsets[c];
        std::int32_t end = map.offsets[c+1];
        
        if (start == end) continue; // Empty cell
        
        auto cell_verts = cell_to_vertex->links((std::int32_t)c);
        
        for (std::int32_t v : cell_verts)
        {
            if (!vmap.has_geom(v)) continue;
            const Real* p = vmap.coords(v);
            
            for (std::int32_t k = start; k < end; ++k)
            {
                std::int32_t t_idx = map.data[k];
                
                // Access triangle data from TriSoup
                std::int32_t p0_idx = soup.tri[3*t_idx];
                std::int32_t p1_idx = soup.tri[3*t_idx+1];
                std::int32_t p2_idx = soup.tri[3*t_idx+2];
                
                const Real* a = &soup.X[3*p0_idx];
                const Real* b = &soup.X[3*p1_idx];
                const Real* c_coord = &soup.X[3*p2_idx];
                
                Real d = point_triangle_distance(p, a, b, c_coord);
                if (d < dist[v]) {
                    dist[v] = d;
                    closest_tri[v] = t_idx;  // Track closest triangle
                }
            }
        }
    }
    
    std::vector<std::int32_t> seeds;
    std::int32_t num_owned = vertex_map->size_local();
    for(std::int32_t v=0; v<num_owned; ++v) {
        if (dist[v] < opt.inf_value) seeds.push_back(v);
    }
    
    compute_distance_fim(*mesh, std::span<Real>(dist), seeds, opt);
    
    // Output closest triangle if requested
    if (closest_tri_out) {
        *closest_tri_out = std::move(closest_tri);
    }
    
    // Copy result to Function
    // 1. Use cached vertex -> DOF mapping
    
    // 2. Copy values
    std::span<Real> func_vals = dist_func.x()->array();
    for(std::int32_t v=0; v<num_vertices; ++v) {
        std::int32_t dof = vmap.vert_to_dof[v];
        if (dof >= 0 && dof < (std::int32_t)func_vals.size()) {
            func_vals[dof] = dist[v];
        }
    }
    
    // 3. Scatter ghost values (though FMM should have handled it, Function ensures consistency)
    dist_func.x()->scatter_fwd();
}

} // namespace cutfemx::distance
