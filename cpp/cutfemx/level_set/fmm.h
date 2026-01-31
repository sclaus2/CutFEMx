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

#include "eikonal_update.h"
#include "parallel_min_exchange.h"
#include "pt_tri_distance.h"
#include "geom_map.h"
#include "../mesh/cell_triangle_map.h"
#include "../mesh/distribute_stl.h" 

namespace cutfemx::level_set {

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

// Low-level FIM Solver
template <typename Real>
void compute_distance_fim(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<Real> dist,
    const std::vector<std::int32_t>& seeds,
    const FMMOptions& opt = {})
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
    
    if (mpi_rank == 0) std::cout << "FIM Init: " << initial_activated << " active nodes." << std::endl;
    
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
                int num_verts = c_verts.size();
                
                int u_idx = -1;
                for(int k=0; k<num_verts; ++k) if(c_verts[k]==u) { u_idx=k; break; }
                if (u_idx == -1) continue; 
                
                if (num_verts == 3) // Triangle
                {
                    std::int32_t v = c_verts[(u_idx + 1) % 3];
                    std::int32_t w = c_verts[(u_idx + 2) % 3];
                    
                    auto attempt_update = [&](std::int32_t target, const Real* x_tgt, 
                                            std::int32_t other, const Real* x_other, Real d_other)
                    {
                        Real new_d = opt.inf_value;
                        if (d_other < opt.inf_value)
                            new_d = update_2pt(x_tgt, x_u, d_u, x_other, d_other);
                        else
                            new_d = update_1pt(x_tgt, x_u, d_u);
                            
                        if (new_d < dist[target] - 1e-13) {
                            Real change = dist[target] - new_d;
                            dist[target] = new_d;
                            
                            // Only count convergence on owned nodes
                            if (target < num_owned) {
                                if (change > local_max_change) local_max_change = change;
                            }
                            
                            // Activate target (Owned or Ghost)
                            if (!active_mask[target]) {
                                active_mask[target] = true;
                                next_active_queue.push_back(target);
                            }
                        }
                    };
                    
                    attempt_update(v, get_vertex_coords(v), w, get_vertex_coords(w), dist[w]);
                    attempt_update(w, get_vertex_coords(w), v, get_vertex_coords(v), dist[v]);
                }
                else if (num_verts == 4) // Tet
                {
                     for(int i=0; i<4; ++i) {
                         if (i == u_idx) continue;
                         std::int32_t target = c_verts[i];
                         
                         int others[2]; int oi=0;
                         for(int j=0; j<4; ++j) if(j!=u_idx && j!=i) others[oi++] = c_verts[j];
                         
                         std::int32_t v = others[0];
                         std::int32_t w = others[1];
                         
                         Real new_d = update_1pt(get_vertex_coords(target), x_u, d_u);
                         if (dist[v] < opt.inf_value) new_d = std::min(new_d, update_2pt(get_vertex_coords(target), x_u, d_u, get_vertex_coords(v), dist[v]));
                         if (dist[w] < opt.inf_value) new_d = std::min(new_d, update_2pt(get_vertex_coords(target), x_u, d_u, get_vertex_coords(w), dist[w]));
                         if (dist[v] < opt.inf_value && dist[w] < opt.inf_value)
                            new_d = std::min(new_d, update_3pt(get_vertex_coords(target), x_u, d_u, get_vertex_coords(v), dist[v], get_vertex_coords(w), dist[w]));
                            
                         if (new_d < dist[target] - 1e-13) {
                             Real change = dist[target] - new_d;
                             dist[target] = new_d;
                             
                             if (target < num_owned) {
                                if (change > local_max_change) local_max_change = change;
                             }
                             
                             if (!active_mask[target]) {
                                active_mask[target] = true;
                                next_active_queue.push_back(target);
                            }
                         }
                     }
                }
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
                 std::cout << "Iter " << iter << " Sync: Owned=" << owned_cnt << " Ghost=" << ghost_cnt << std::endl;
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
            if (mpi_rank == 0) std::cout << "FIM Converged at iter " << iter << std::endl;
            break;
        }
    }
}

// Helper to get keys from map
inline std::vector<std::int32_t> get_cut_cells(const cutfemx::mesh::CellTriangleMap& map) {
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
    const ::cutfemx::mesh::TriSoup<Real>& soup, // Changed from DistributedSTL to TriSoup
    const ::cutfemx::mesh::CellTriangleMap& map,
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
    std::span<Real> func_vals = dist_func.x()->mutable_array();
    for(std::int32_t v=0; v<num_vertices; ++v) {
        std::int32_t dof = vmap.vert_to_dof[v];
        if (dof >= 0 && dof < (std::int32_t)func_vals.size()) {
            func_vals[dof] = dist[v];
        }
    }
    
    // 3. Scatter ghost values (though FMM should have handled it, Function ensures consistency)
    dist_func.x()->scatter_fwd();
}

} // namespace cutfemx::level_set
