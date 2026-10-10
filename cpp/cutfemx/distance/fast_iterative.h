// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

// =========================================================================
// Distributed eikonal solver on vertex graphs
//
// Each process runs a heap-ordered (fast marching) label-correcting solve
// on its part of the mesh, then exchanges the distances of shared vertices
// with its neighbours. Vertices whose distance improved in the exchange go
// back on the heap, and the rounds repeat until no process has pending
// vertices. Each round only processes distances up to the smallest pending
// distance on any process plus a round width, so that a process does not
// compute far ahead of fronts that are still to arrive from its neighbours.
// The number of rounds depends on the range of distances over the round
// width and on how many partition boundaries the fronts cross, not on the
// mesh size.
//
// Label correcting (re-pushing improved vertices) keeps the solve exact on
// stencils that are not causal (obtuse simplices, distorted cells), where a
// pure fast marching method would accept wrong values.
// =========================================================================

#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Scatterer.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <dolfinx/common/Timer.h>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

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
    
    // Maximum number of local-solve-and-exchange rounds
    int max_iter = 1000;
    // Improvements of a vertex distance by less than tol are not propagated,
    // so the result is within tol of the converged discrete solution
    double tol = 1e-10;
    double inf_value = 1e30;
    // Distances are only computed up to max_distance; larger distances and
    // unreached vertices are set to max_distance
    double max_distance = std::numeric_limits<double>::infinity();
    // In parallel, each round only processes distances up to the smallest
    // pending distance on any process plus round_width. Smaller widths repeat
    // less work where fronts from different processes meet, at the cost of
    // more rounds (about the range of distances divided by round_width).
    // round_width <= 0 uses 1/32 of the diagonal of the mesh bounding box.
    double round_width = 0.0;

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

/// Order of transported payloads [speed, normal...] used to choose between
/// sources at the same distance. Components that agree to a relative 1e-8
/// count as equal, so rounding differences between processes do not decide.
template <typename Real>
bool payload_precedes(Real speed_a, const Real* normal_a, Real speed_b,
                      const Real* normal_b, int normal_dim)
{
    auto differ = [](Real a, Real b)
    {
        return std::abs(a - b)
               > Real(1e-8) * std::max({Real(1), std::abs(a), std::abs(b)});
    };
    if (differ(speed_a, speed_b))
        return speed_a < speed_b;
    for (int d = 0; d < normal_dim; ++d)
        if (differ(normal_a[d], normal_b[d]))
            return normal_a[d] < normal_b[d];
    return false;
}

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

// Distributed eikonal solve on the vertices of a mesh.
//
// dist holds the initial distances (inf_value where unknown) for owned and
// ghost vertices and is overwritten by the solution. Every finite vertex is
// a source. payload, if given, is copied from the vertex an update came from
// (velocity extension). Where updates give the same distance (within tol)
// with different payloads, the payload that comes first in payload_precedes
// wins, so the payload does not depend on the processing order or the
// partition. On
// return, ghost entries equal their owners'.
template <typename Real>
void compute_distance_fim(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<Real> dist,
    const std::vector<std::int32_t>& /*seeds*/,
    const FMMOptions& opt = {},
    FIMTransportPayload<Real>* payload = nullptr)
{
    dolfinx::common::Timer timer("FIM solve");

    MPI_Comm comm = mesh.comm();
    const int mpi_rank = dolfinx::MPI::rank(comm);
    const int mpi_size = dolfinx::MPI::size(comm);
    const int tdim = mesh.topology()->dim();

    auto topology = mesh.topology();
    if (!topology->connectivity(0, tdim)) {
        mesh.topology_mutable()->create_connectivity(0, tdim);
        mesh.topology_mutable()->create_connectivity(tdim, 0);
    }
    auto vertex_to_cell = topology->connectivity(0, tdim);
    auto cell_to_vertex = topology->connectivity(tdim, 0);

    const auto& vertex_map = topology->index_map(0);
    const std::int32_t num_owned = vertex_map->size_local();
    const std::int32_t num_vertices = num_owned + vertex_map->num_ghosts();
    if (dist.size() < static_cast<std::size_t>(num_vertices))
        throw std::invalid_argument("FIM distance array is too small.");
    if (payload && !payload->valid(num_vertices))
        throw std::invalid_argument("FIM payload arrays are too small.");

    VertexMapCache<Real> vmap;
    vmap.build(mesh);

    const Real inf_value = static_cast<Real>(opt.inf_value);
    const Real tol = static_cast<Real>(opt.tol);
    const Real max_distance
        = std::min(static_cast<Real>(opt.max_distance), inf_value);

    // Min-heap of (tentative distance, vertex). Entries whose distance is
    // larger than the vertex's current distance are stale and skipped.
    using Entry = std::pair<Real, std::int32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;

    auto payload_less = [&](Real speed_a, const Real* normal_a, Real speed_b,
                            const Real* normal_b)
    {
        return payload_precedes(speed_a, normal_a, speed_b, normal_b,
                                payload->normal_dim);
    };
    auto normal_of = [&](std::int32_t v)
    {
        return payload->normals.data()
               + static_cast<std::size_t>(v) * payload->normal_dim;
    };
    // Is distance d_a with the payload of vertex a better than distance d_b
    // with the payload of vertex b?
    auto better = [&](Real d_a, std::int32_t a, Real d_b, std::int32_t b)
    {
        if (d_a < d_b - tol)
            return true;
        if (!payload || d_a > d_b + tol)
            return false;
        return payload_less(payload->speed[a], normal_of(a), payload->speed[b],
                            normal_of(b));
    };
    // Is an exchanged block [distance, payload...] better than vertex v?
    auto block_better = [&](const Real* block, std::int32_t v)
    {
        if (block[0] < dist[v] - tol)
            return true;
        if (!payload || block[0] > dist[v] + tol)
            return false;
        return payload_less(block[1], block + 2, payload->speed[v], normal_of(v));
    };

    auto copy_payload = [&](std::int32_t from, std::int32_t to)
    {
        payload->speed[to] = payload->speed[from];
        std::copy_n(normal_of(from), payload->normal_dim, normal_of(to));
    };

    auto improve = [&](std::int32_t target, Real new_d,
                       std::span<const std::int32_t> sources)
    {
        // The payload comes from the best of the update's sources
        std::int32_t best = sources[0];
        for (std::int32_t v : sources)
            if (better(dist[v], v, dist[best], best))
                best = v;
        if (!better(new_d, best, dist[target], target))
            return;
        dist[target] = std::min(dist[target], new_d);
        if (payload)
            copy_payload(best, target);
        if (dist[target] <= max_distance)
            heap.emplace(dist[target], target);
    };

    // Relax every vertex of the virtual simplices around u from u.
    auto relax_from = [&](std::int32_t u)
    {
        const Real d_u = dist[u];
        const Real* x_u = vmap.coords(u);
        for (std::int32_t c : vertex_to_cell->links(u))
        {
            for_each_virtual_simplex(
                tdim, cell_to_vertex->links(c),
                [&](std::span<const std::int32_t> simplex)
                {
                    if (std::find(simplex.begin(), simplex.end(), u)
                        == simplex.end())
                        return;

                    // The other vertices of the simplex, in order
                    std::array<std::int32_t, 3> others{-1, -1, -1};
                    int num_others = 0;
                    for (std::int32_t v : simplex)
                        if (v != u)
                            others[static_cast<std::size_t>(num_others++)] = v;

                    for (int i = 0; i < num_others; ++i)
                    {
                        const std::int32_t target = others[static_cast<std::size_t>(i)];
                        const Real* x_t = vmap.coords(target);

                        // Remaining vertices of the face opposite target.
                        // Distances beyond max_distance are tentative and
                        // depend on the processing order, so they are not
                        // used.
                        std::array<std::int32_t, 2> face{-1, -1};
                        int num_face = 0;
                        for (int j = 0; j < num_others; ++j)
                        {
                            const std::int32_t v = others[static_cast<std::size_t>(j)];
                            if (j != i && dist[v] < inf_value
                                && dist[v] <= max_distance)
                                face[static_cast<std::size_t>(num_face++)] = v;
                        }

                        if (num_face == 2)
                        {
                            // The interior minimum is the minimum over the
                            // whole face when it exists.
                            const std::int32_t v = face[0];
                            const std::int32_t w = face[1];
                            const Real d = update_3pt_interior(
                                x_t, x_u, d_u, vmap.coords(v), dist[v],
                                vmap.coords(w), dist[w]);
                            if (d < inf_value)
                            {
                                const std::array<std::int32_t, 3> src{u, v, w};
                                improve(target, d, src);
                                continue;
                            }
                        }

                        bool edge_update = false;
                        for (int j = 0; j < num_face; ++j)
                        {
                            const std::int32_t v = face[static_cast<std::size_t>(j)];
                            const std::array<std::int32_t, 2> src{u, v};
                            improve(target,
                                    update_2pt(x_t, x_u, d_u, vmap.coords(v),
                                               dist[v]),
                                    src);
                            edge_update = true;
                        }
                        if (!edge_update)
                        {
                            const std::array<std::int32_t, 1> src{u};
                            improve(target, update_1pt(x_t, x_u, d_u), src);
                        }
                    }
                });
        }
    };

    std::int64_t num_relaxations = 0;
    auto drain = [&](Real threshold)
    {
        while (!heap.empty())
        {
            const auto [d, u] = heap.top();
            if (d > dist[u])
            {
                heap.pop();
                continue;
            }
            if (d > threshold)
                break;
            heap.pop();
            ++num_relaxations;
            relax_from(u);
        }
    };

    // Smallest pending distance on any process, infinity if none
    constexpr Real no_work = std::numeric_limits<Real>::infinity();
    auto global_min_pending = [&]() -> Real
    {
        while (!heap.empty() && heap.top().first > dist[heap.top().second])
            heap.pop();
        Real local = heap.empty() ? no_work : heap.top().first;
        Real global = local;
        if (mpi_size > 1)
            MPI_Allreduce(&local, &global, 1, dolfinx::MPI::mpi_t<Real>,
                          MPI_MIN, comm);
        return global;
    };

    // Exchange of shared vertices with neighbouring processes: ghost
    // distances are reduced onto their owners with min (carrying the payload
    // of the minimum), then owners overwrite their ghosts. Each vertex is
    // a block of [distance, payload...] values.
    const int width = 1 + (payload ? 1 + payload->normal_dim : 0);
    std::unique_ptr<dolfinx::common::Scatterer<>> scatterer;
    std::vector<Real> owned_buffer, ghost_buffer;
    if (mpi_size > 1)
    {
        scatterer = std::make_unique<dolfinx::common::Scatterer<>>(
            *vertex_map, width);
        owned_buffer.resize(scatterer->local_indices().size());
        ghost_buffer.resize(scatterer->remote_indices().size());
    }

    auto pack = [&](std::int32_t v, Real* block)
    {
        block[0] = dist[v];
        if (payload)
        {
            block[1] = payload->speed[v];
            for (int d = 0; d < payload->normal_dim; ++d)
                block[2 + d] = payload->normals[static_cast<std::size_t>(v)
                                                    * payload->normal_dim + d];
        }
    };
    auto unpack = [&](std::int32_t v, const Real* block)
    {
        dist[v] = block[0];
        if (payload)
        {
            payload->speed[v] = block[1];
            for (int d = 0; d < payload->normal_dim; ++d)
                payload->normals[static_cast<std::size_t>(v)
                                     * payload->normal_dim + d]
                    = block[2 + d];
        }
    };

    auto exchange = [&]()
    {
        const auto& owned_idx = scatterer->local_indices();
        const auto& ghost_idx = scatterer->remote_indices();
        const std::size_t num_owned_blocks = owned_idx.size() / width;
        const std::size_t num_ghost_blocks = ghost_idx.size() / width;
        auto owned_vertex = [&](std::size_t k)
        { return static_cast<std::int32_t>(owned_idx[k * width] / width); };
        auto ghost_vertex = [&](std::size_t k)
        { return num_owned + static_cast<std::int32_t>(ghost_idx[k * width] / width); };
        // Ghosts -> owners, min
        for (std::size_t k = 0; k < num_ghost_blocks; ++k)
            pack(ghost_vertex(k), ghost_buffer.data() + k * width);
        MPI_Request request = MPI_REQUEST_NULL;
        scatterer->scatter_rev_begin(ghost_buffer.data(), owned_buffer.data(),
                                     request);
        scatterer->scatter_end(request);
        for (std::size_t k = 0; k < num_owned_blocks; ++k)
        {
            const std::int32_t v = owned_vertex(k);
            const Real* block = owned_buffer.data() + k * width;
            if (block_better(block, v))
            {
                const Real d = std::min(dist[v], block[0]);
                unpack(v, block);
                dist[v] = d;
                if (d <= max_distance)
                    heap.emplace(d, v);
            }
        }

        // Owners -> ghosts, copy
        for (std::size_t k = 0; k < num_owned_blocks; ++k)
            pack(owned_vertex(k), owned_buffer.data() + k * width);
        request = MPI_REQUEST_NULL;
        scatterer->scatter_fwd_begin(owned_buffer.data(), ghost_buffer.data(),
                                     request);
        scatterer->scatter_end(request);
        for (std::size_t k = 0; k < num_ghost_blocks; ++k)
        {
            const std::int32_t v = ghost_vertex(k);
            const Real* block = ghost_buffer.data() + k * width;
            // Any decrease, even within tol, makes the pending heap entry of
            // v stale, so v must be pushed again.
            bool changed = block[0] < dist[v];
            if (payload)
                changed = changed || block[1] != payload->speed[v]
                          || !std::equal(block + 2, block + 2 + payload->normal_dim,
                                         normal_of(v));
            unpack(v, block);
            if (changed && block[0] <= max_distance)
                heap.emplace(block[0], v);
        }
    };

    // Every finite vertex is a source. Seeds on ghost vertices reach their
    // owners in the first exchange.
    for (std::int32_t v = 0; v < num_vertices; ++v)
        if (dist[v] <= max_distance)
            heap.emplace(dist[v], v);

    // A serial solve is a single fast marching pass.
    Real round_width = no_work;
    if (mpi_size > 1 && opt.round_width > 0)
        round_width = static_cast<Real>(opt.round_width);
    else if (mpi_size > 1)
    {
        // Bounding box as [min_x, min_y, min_z, -max_x, -max_y, -max_z]
        std::array<Real, 6> box;
        box.fill(std::numeric_limits<Real>::max());
        for (std::int32_t v = 0; v < num_vertices; ++v)
        {
            if (!vmap.has_geom(v))
                continue;
            const Real* x = vmap.coords(v);
            for (int k = 0; k < 3; ++k)
            {
                box[k] = std::min(box[k], x[k]);
                box[3 + k] = std::min(box[3 + k], -x[k]);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, box.data(), 6, dolfinx::MPI::mpi_t<Real>,
                      MPI_MIN, comm);
        Real diagonal_sq = 0;
        for (int k = 0; k < 3; ++k)
            diagonal_sq += (box[3 + k] + box[k]) * (box[3 + k] + box[k]);
        round_width = std::sqrt(diagonal_sq) / Real(32);
    }
    int round = 0;
    Real pending = global_min_pending();
    while (pending < no_work && round < opt.max_iter)
    {
        ++round;
        drain(pending + round_width);
        if (mpi_size > 1)
            exchange();
        pending = global_min_pending();
    }
    const bool converged = !(pending < no_work);

    // Vertices beyond the band may hold tentative distances (and payloads)
    // that were never propagated; they are not part of the solution.
    if (max_distance < inf_value)
    {
        for (std::int32_t v = 0; v < num_vertices; ++v)
        {
            if (dist[v] <= max_distance)
                continue;
            dist[v] = max_distance;
            if (payload)
            {
                payload->speed[v] = Real(0);
                for (int d = 0; d < payload->normal_dim; ++d)
                    payload->normals[static_cast<std::size_t>(v)
                                         * payload->normal_dim + d]
                        = Real(0);
            }
        }
    }

    std::int64_t total_relaxations = 0;
    MPI_Reduce(&num_relaxations, &total_relaxations, 1, MPI_INT64_T, MPI_SUM,
               0, comm);
    if (mpi_rank == 0)
    {
        if (converged)
            spdlog::info("FIM converged in {} rounds, {} vertex relaxations.",
                         round, total_relaxations);
        else
            spdlog::warn("FIM stopped after max_iter={} rounds without "
                         "converging ({} vertex relaxations).",
                         opt.max_iter, total_relaxations);
    }
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
