#pragma once

// Method 2: Component Anchor (Flooding)
// Determine sign by flooding from the domain boundary.
// 1. Initialize all cells to negative (Inside).
// 2. Mark boundary cells as positive (Outside).
// 3. Flood positive sign through the mesh, blocked by cut facets.
// 4. Synchronize parallel ghosts to propagate flooding across partitions.

#include "sign_options.h"
#include <cutfemx/mesh/stl/cell_triangle_map.h>

#include <cutfemx/mesh/stl/tri_intersection.h>
#include <cutfemx/level_set/parallel_min_exchange.h>
#include "geom_map.h"
#include <dolfinx/common/log.h>
#include <sstream>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/fem/Function.h>
#include <vector>
#include <span>
#include <array>
#include <queue>

namespace cutfemx::level_set {

/// Mark facets crossed by the surface
/// A facet is marked if the surface explicitly intersects it.
/// @return vector of bools size = num_local_facets + num_ghost_facets
template <typename Real>
std::vector<bool> mark_cut_facets(
    const dolfinx::mesh::Mesh<Real>& mesh,
    const ::cutfemx::mesh::CellTriangleMap& cell_tri_map,
    const ::cutfemx::mesh::TriSoup<Real>& soup);

/// Apply sign using naive flooding from boundary
/// @param phi Unsigned distance function (input), Signed distance (output)
/// @param mesh The mesh
/// @param cut_facets Boolean mask of cut facets
template <typename Real>
void apply_sign_flood_fill(
    dolfinx::fem::Function<Real>& phi,
    const dolfinx::mesh::Mesh<Real>& mesh,
    const std::vector<bool>& cut_facets);

// ==========================================
// Implementation Details
// ==========================================

template <typename Real>
std::vector<bool> mark_cut_facets(
    const dolfinx::mesh::Mesh<Real>& mesh,
    const ::cutfemx::mesh::CellTriangleMap& cell_tri_map,
    const ::cutfemx::mesh::TriSoup<Real>& soup)
{
    const auto topology = mesh.topology(); // shared_ptr<const Topology>
    const int tdim = topology->dim();
    const int fdim = tdim - 1;

    // Ensure connectivity
    auto topology_mut = mesh.topology_mutable();
    topology_mut->create_connectivity(tdim, fdim);
    topology_mut->create_connectivity(fdim, 0);

    auto cell_to_facet = topology->connectivity(tdim, fdim);
    auto facet_to_vertex = topology->connectivity(fdim, 0);

    auto facet_map = topology->index_map(fdim);
    const std::int32_t num_facets = facet_map->size_local() + facet_map->num_ghosts();
    
    std::vector<bool> is_cut(num_facets, false);



    // Geometry access via centralized map
    VertexMapCache<Real> vmap;
    vmap.build(mesh); // No function space needed for geometry only
    
    const std::size_t num_cells = cell_tri_map.num_cells();
    
    // Check cell type
    const dolfinx::mesh::CellType cell_type = topology->cell_type(); 
    const bool is_tet = (cell_type == dolfinx::mesh::CellType::tetrahedron);
    
    for (std::int32_t c = 0; c < (std::int32_t)num_cells; ++c)
    {
        auto tris = cell_tri_map.triangles(c);
        if (tris.empty()) continue;

        auto facets = cell_to_facet->links(c);
        
        for (std::int32_t f : facets)
        {
            if (is_cut[f]) continue; // Already marked

            auto f_verts = facet_to_vertex->links(f);
            
            for (std::int32_t t_idx : tris)
            {
                std::int32_t i0 = soup.tri[3*t_idx+0];
                std::int32_t i1 = soup.tri[3*t_idx+1];
                std::int32_t i2 = soup.tri[3*t_idx+2];
                const Real* t0 = &soup.X[3*i0];
                const Real* t1 = &soup.X[3*i1];
                const Real* t2 = &soup.X[3*i2];

                bool intersect = false;
                if (is_tet) {
                    std::int32_t v0 = f_verts[0];
                    std::int32_t v1 = f_verts[1];
                    std::int32_t v2 = f_verts[2];
                    
                    if (!vmap.has_geom(v0) || !vmap.has_geom(v1) || !vmap.has_geom(v2)) continue;
                    
                    const Real* f0 = vmap.coords(v0);
                    const Real* f1 = vmap.coords(v1);
                    const Real* f2 = vmap.coords(v2);
                    
                    intersect = ::cutfemx::mesh::triangle_triangle_intersect(t0, t1, t2, f0, f1, f2);
                }
                else {
                    // Quad
                    std::int32_t v0 = f_verts[0];
                    std::int32_t v1 = f_verts[1];
                    std::int32_t v2 = f_verts[2];
                    std::int32_t v3 = f_verts[3];
                    
                    if (!vmap.has_geom(v0) || !vmap.has_geom(v1) || !vmap.has_geom(v2) || !vmap.has_geom(v3)) continue;
                    
                    const Real* f0 = vmap.coords(v0);
                    const Real* f1 = vmap.coords(v1);
                    const Real* f2 = vmap.coords(v2);
                    const Real* f3 = vmap.coords(v3);
                    
                    intersect = ::cutfemx::mesh::triangle_triangle_intersect(t0, t1, t2, f0, f1, f2) ||
                                ::cutfemx::mesh::triangle_triangle_intersect(t0, t1, t2, f0, f2, f3);
                }

                if (intersect) {
                    is_cut[f] = true;
                    break; 
                }
            }
        }
    }

    int cut_count = 0;
    for(bool b : is_cut) if(b) cut_count++;
    
    // Explicitly Synchronize Cuts (OR rule)
    cutfemx::level_set::or_exchange_bool(*facet_map, is_cut);

    // Logging removed or moved to debug level if needed
    return is_cut;
}

template <typename Real>
void apply_sign_flood_fill(
    dolfinx::fem::Function<Real>& phi,
    const dolfinx::mesh::Mesh<Real>& mesh,
    const std::vector<bool>& cut_facets)
{
    const auto topology = mesh.topology(); // shared_ptr<const Topology>
    const int tdim = topology->dim();
    const int fdim = tdim - 1;

    auto mesh_mut = mesh.topology_mutable();
    if (!topology->connectivity(tdim, fdim)) mesh_mut->create_connectivity(tdim, fdim);
    if (!topology->connectivity(fdim, tdim)) mesh_mut->create_connectivity(fdim, tdim);

    auto cell_to_facet = topology->connectivity(tdim, fdim);
    auto facet_to_cell = topology->connectivity(fdim, tdim);
    auto cell_map = topology->index_map(tdim);

    const std::int32_t num_local_cells = cell_map->size_local();
    const std::int32_t num_total_cells = num_local_cells + cell_map->num_ghosts();

    // 1. Initialize sign to Negative (-1)
    // We use int8: -1 (Inside), +1 (Outside)
    std::vector<std::int8_t> cell_sign(num_total_cells, -1);

    // 2. Identify Seeds (Boundary Cells)
    std::queue<std::int32_t> queue;
    std::vector<bool> visited(num_total_cells, false);

    for (std::int32_t c = 0; c < num_local_cells; ++c) {  // ONLY OWNED CELLS AS SEEDS
        auto facets = cell_to_facet->links(c);
        bool on_boundary = false;
        for (std::int32_t f : facets) {
            // Check if facet is boundary (only 1 cell neighbor = physical boundary)
            if (facet_to_cell->links(f).size() == 1) {
                on_boundary = true;
                break;
            }
        }
        
        if (on_boundary) {
            cell_sign[c] = 1;
            queue.push(c);
            visited[c] = true;
        }
    }
    
    // 3. Flood Fill Loop (Iterative with MPI)
    
    int mpi_rank = dolfinx::MPI::rank(mesh.comm());

    // Debug output
    long seed_count = queue.size();
    long total_seeds = 0;
    MPI_Allreduce(&seed_count, &total_seeds, 1, MPI_LONG, MPI_SUM, mesh.comm());
    if (mpi_rank == 0) {
        spdlog::info("DEBUG: Flood seeds (boundary cells): {}", total_seeds);
    }
    
    long cut_count = 0;
    for(bool b : cut_facets) if(b) cut_count++;
    long total_cuts = 0;
    MPI_Allreduce(&cut_count, &total_cuts, 1, MPI_LONG, MPI_SUM, mesh.comm());
    if (mpi_rank == 0) {
         spdlog::info("DEBUG: Cut facets (local+ghost per rank): {} Total (approx): {}", cut_count, total_cuts);
    }
    
    
    // We iterate until global convergence
    bool changed = true;
    int iter = 0;
    const int MAX_ITER = 10000;
    while (changed && iter < MAX_ITER) {
        // Local Propagate
        while(!queue.empty()) {
            std::int32_t c = queue.front();
            queue.pop();
            
            // If I am +1, infect neighbors
            if (cell_sign[c] != 1) continue; 
            
            auto facets = cell_to_facet->links(c);
            for (std::int32_t f : facets) {
                if (cut_facets[f]) continue; // Blocked by cut
                
                auto neighbors = facet_to_cell->links(f);
                for (std::int32_t n : neighbors) {
                    if (n == c) continue;
                    if (cell_sign[n] == -1) {
                        cell_sign[n] = 1;
                        visited[n] = true;
                        
                        // Rule: Only push OWNED cells to queue immediately.
                        // Ghosts must wait for MPI confirmation to avoid bypassing cuts.
                        if (n < num_local_cells) {
                            queue.push(n);
                        }
                    }
                }
            }
        }
        
        // Parallel Exchange
        // We need to propagate +1.
        // Use negation trick with min_exchange (which propagates minimizer).
        // +1 -> -1 (Minimizer)
        // -1 -> +1
        std::vector<double> exchange_buf(num_total_cells);
        for(int i=0; i<num_total_cells; ++i) exchange_buf[i] = -static_cast<double>(cell_sign[i]);
        
        // Sync
        cutfemx::level_set::min_exchange_vertices<double>(*cell_map, std::span<double>(exchange_buf));
        
        // Check for changes and update queue
        bool local_change = false;
        for(int c=0; c<num_total_cells; ++c) {
            std::int8_t new_sign = -static_cast<std::int8_t>(std::round(exchange_buf[c])); // -(-1) = +1
            if (new_sign != cell_sign[c]) {
                cell_sign[c] = new_sign;
                local_change = true;
                // If we flipped to +1, we become a seed for local propagation
                if (new_sign == 1 && !visited[c]) {
                    visited[c] = true;
                    queue.push(c);
                }
            }
        }
        
        // Global convergence check
        int global_change = 0;
        int local_change_int = local_change ? 1 : 0;
        MPI_Allreduce(&local_change_int, &global_change, 1, MPI_INT, MPI_LOR, mesh.comm());
        
        // if (dolfinx::MPI::rank(mesh.comm()) == 0 && iter % 100 == 0) std::cout << "DEBUG: Flood iter " << iter << " active." << std::endl;
        iter++;
        changed = (global_change > 0);
    }
    
    if (iter >= MAX_ITER && dolfinx::MPI::rank(mesh.comm()) == 0) {
        spdlog::warn("Flood fill max iterations reached! Flooding may be incomplete.");
    }
    
    // Debug: Final Positive Count
    long local_pos = 0;
    for(auto s : cell_sign) if(s == 1) local_pos++;
    long global_pos = 0;
    MPI_Allreduce(&local_pos, &global_pos, 1, MPI_LONG, MPI_SUM, mesh.comm());
    
    long total_cells_global = 0;
    long local_cells_count = (long)num_total_cells;
    MPI_Allreduce(&local_cells_count, &total_cells_global, 1, MPI_LONG, MPI_SUM, mesh.comm());
    
    if (mpi_rank == 0) {
         spdlog::info("DEBUG: Final Positive Cells: {} / {}", global_pos, total_cells_global);
    }

    // 4. Vote for Vertices
    if (!topology->connectivity(tdim, 0)) mesh_mut->create_connectivity(tdim, 0);
    auto cell_to_vert = topology->connectivity(tdim, 0);
    
    // We need map Vertex -> Dof
    const auto& dofmap = phi.function_space()->dofmap();
    auto vertex_map = topology->index_map(0);
    const std::int32_t num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
    
    // Simple Vertex Vote: Sum signs of incident cells
    std::vector<int> votes(num_vertices, 0);
    
    for (std::int32_t c = 0; c < num_total_cells; ++c) {
        std::int8_t s = cell_sign[c];
        auto verts = cell_to_vert->links(c);
        for (std::int32_t v : verts) {
             if (v < num_vertices) votes[v] += s;
        }
    }
    
    // CRITICAL: Synchronize votes across MPI ranks
    // Ghost vertices may have received votes from ONLY local cells, 
    // while the owner needs to aggregate votes from ALL cells sharing the vertex.
    // We use a SUM reduction: ghost → owner, then owner → ghost.
    {
        MPI_Comm comm = mesh.comm();
        const int mpi_size = dolfinx::MPI::size(comm);
        
        if (mpi_size > 1) {
            const std::int32_t size_local = vertex_map->size_local();
            const std::int32_t num_ghosts_v = vertex_map->num_ghosts();
            
            std::span<const int> ghost_owners = vertex_map->owners();
            std::span<const std::int64_t> ghosts_global = vertex_map->ghosts();
            
            // Count messages per destination
            std::vector<int> send_counts(mpi_size, 0);
            for (int owner : ghost_owners) send_counts[owner]++;
            
            std::vector<int> send_displs(mpi_size + 1, 0);
            for (int i = 0; i < mpi_size; ++i) send_displs[i + 1] = send_displs[i] + send_counts[i];
            
            // Pack ghost votes and global IDs
            std::vector<int> send_vals(num_ghosts_v);
            std::vector<std::int64_t> send_global_ids(num_ghosts_v);
            std::vector<int> cursors(mpi_size, 0);
            
            for (std::int32_t g = 0; g < num_ghosts_v; ++g) {
                int owner = ghost_owners[g];
                int idx = send_displs[owner] + cursors[owner]++;
                send_vals[idx] = votes[size_local + g];
                send_global_ids[idx] = ghosts_global[g];
            }
            
            // Exchange counts
            std::vector<int> recv_counts(mpi_size);
            MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
            
            std::vector<int> recv_displs(mpi_size + 1, 0);
            for (int i = 0; i < mpi_size; ++i) recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
            
            // Exchange values
            int total_recv = recv_displs[mpi_size];
            std::vector<int> recv_vals(total_recv);
            std::vector<std::int64_t> recv_global_ids(total_recv);

            MPI_Alltoallv(send_vals.data(), send_counts.data(), send_displs.data(), MPI_INT,
                          recv_vals.data(), recv_counts.data(), recv_displs.data(), MPI_INT, comm);
            MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                          recv_global_ids.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T, comm);
            
            // SUM received ghost votes into owned vertices
            auto local_range = vertex_map->local_range();
            for (int i = 0; i < total_recv; ++i) {
                std::int64_t global_id = recv_global_ids[i];
                if (global_id >= local_range[0] && global_id < local_range[1]) {
                    std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
                    votes[local_id] += recv_vals[i];
                }
            }
            
            // Now send owner's final vote back to ghosts
            std::vector<int> owned_reply(total_recv);
            for (int i = 0; i < total_recv; ++i) {
                std::int64_t global_id = recv_global_ids[i];
                if (global_id >= local_range[0] && global_id < local_range[1]) {
                    std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
                    owned_reply[i] = votes[local_id];
                }
            }
            
            std::vector<int> ghost_update(num_ghosts_v);
            MPI_Alltoallv(owned_reply.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                          ghost_update.data(), send_counts.data(), send_displs.data(), MPI_INT, comm);
            
            // Update ghost votes with owner's aggregated value
            std::fill(cursors.begin(), cursors.end(), 0);
            for (std::int32_t g = 0; g < num_ghosts_v; ++g) {
                int owner = ghost_owners[g];
                int idx = send_displs[owner] + cursors[owner]++;
                votes[size_local + g] = ghost_update[idx];
            }
        }
    }
    
    // 5. Update Phi
    // Need Vertex->Dof mapping.
    VertexMapCache<Real> vmap;
    vmap.build(mesh, phi.function_space().get());
    const std::vector<std::int32_t>& vert_to_dof = vmap.vert_to_dof;
    
    auto data = phi.x()->mutable_array();
    
    for (std::int32_t v = 0; v < num_vertices; ++v) {
        std::int32_t dof = vert_to_dof[v];
        if (dof < 0 || dof >= (int)data.size()) continue;
        
        // Majority vote
        Real s = (votes[v] >= 0) ? 1.0 : -1.0;
        data[dof] = s * std::abs(data[dof]);
    }
    
    // Final sync: Ensure ghost DOFs match owner DOFs
    phi.x()->scatter_fwd();
}

} // namespace cutfemx::level_set
