#pragma once

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/graph/AdjacencyList.h>

#include "cell_triangle_map.h"
#include "distribute_stl.h"

#include <vector>
#include <algorithm>
#include <set>
#include <span>

namespace cutfemx::mesh::stl {

/// Mark cut cells from CellTriangleMap
/// Returns list of local cell indices that intersect the STL
inline std::vector<std::int32_t> mark_cut_cells(const CellTriangleMap& m) {
    std::vector<std::int32_t> cut_cells;
    std::size_t n = m.num_cells();
    cut_cells.reserve(n / 10);
    for(std::size_t i = 0; i < n; ++i) {
        if(m.offsets[i+1] > m.offsets[i]) {
            cut_cells.push_back((std::int32_t)i);
        }
    }
    return cut_cells;
}

/// Expand cell set by k rings of neighbors via facets
template <typename Real>
inline std::vector<std::int32_t> expand_cells_k_ring(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<const std::int32_t> seed_cells,
    int k)
{
    if (k <= 0) return std::vector<std::int32_t>(seed_cells.begin(), seed_cells.end());

    auto topology = mesh.topology(); // Returns const Topology*
    const int tdim = topology->dim();
    const int fdim = tdim - 1;

    // We assume connectivity exists. Caller must ensure it.
    auto c2f = topology->connectivity(tdim, fdim);
    auto f2c = topology->connectivity(fdim, tdim);
    
    if (!c2f || !f2c) {
        throw std::runtime_error("Mesh connectivity (Cell <-> Facet) missing.");
    }

    std::vector<bool> visited(topology->index_map(tdim)->size_local() + topology->index_map(tdim)->num_ghosts(), false);
    std::vector<std::int32_t> current_layer(seed_cells.begin(), seed_cells.end());
    
    for(auto c : current_layer) {
        if(c < (std::int32_t)visited.size()) visited[c] = true;
    }
    
    for(int ring = 0; ring < k; ++ring) {
        std::vector<std::int32_t> next_layer;
        next_layer.reserve(current_layer.size() * 2);
        
        for(auto c : current_layer) {
            auto facets = c2f->links(c);
            for(auto f : facets) {
                auto neighbors = f2c->links(f);
                for(auto n : neighbors) {
                    if (n < (std::int32_t)visited.size() && !visited[n]) {
                        visited[n] = true;
                        next_layer.push_back(n);
                    }
                }
            }
        }
        
        if (next_layer.empty()) break;
        current_layer.insert(current_layer.end(), next_layer.begin(), next_layer.end());
    }
    
    return current_layer;
}

/// Convert cells to unique edges
template <typename Real>
inline std::vector<std::int32_t> cells_to_edges(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<const std::int32_t> cells)
{
    auto topology = mesh.topology();
    const int tdim = topology->dim();
    
    auto c2e = topology->connectivity(tdim, 1);
    if (!c2e) throw std::runtime_error("Mesh connectivity (Cell -> Edge) missing.");
    
    std::vector<std::int32_t> edges;
    edges.reserve(cells.size() * 6);
    
    for(auto c : cells) {
        auto cell_edges = c2e->links(c);
        edges.insert(edges.end(), cell_edges.begin(), cell_edges.end());
    }
    
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    
    return edges;
}

/// Helper: Refinement edges from STL
template <typename Real>
inline std::vector<std::int32_t> refinement_edges_from_stl(
    dolfinx::mesh::Mesh<Real>& mesh,
    const std::string& stl_path,
    double aabb_padding,
    int k_ring)
{
    // 0. Ensure connectivities (needs mutable mesh reference)
    int tdim = mesh.topology()->dim();
    mesh.topology_mutable()->create_connectivity(tdim, tdim-1);
    mesh.topology_mutable()->create_connectivity(tdim-1, tdim);
    mesh.topology_mutable()->create_connectivity(tdim, 1);
    
    // 1. Distribute STL
    DistributeSTLOptions opt;
    opt.aabb_padding = aabb_padding;
    auto soup = distribute_stl_facets(mesh.comm(), mesh, stl_path, opt);
    
    // 2. Build Map
    CellTriangleMapOptions map_opt;
    auto map = build_cell_triangle_map(mesh, soup, map_opt);
    
    // 3. Mark
    auto cut_cells = mark_cut_cells(map);
    
    // 4. Expand
    auto cells = expand_cells_k_ring(mesh, cut_cells, k_ring);
    
    // 5. Edges
    auto edges = cells_to_edges(mesh, cells);
    
    return edges;
}

} // namespace cutfemx::mesh::stl
