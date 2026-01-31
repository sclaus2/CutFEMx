#pragma once

// Band Selection: expand from seed cells using dolfinx connectivities
// Provides band cells and band vertices for FMM NearField seeding

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

#include <vector>
#include <span>
#include <algorithm>
#include <unordered_set>

namespace cutfemx::level_set {

/// Expand from seed cells outward using facet adjacency
/// @param mesh The background mesh
/// @param seed_cells Initial set of cells (e.g., cut cells)
/// @param layers Number of expansion layers (0 = seeds only)
/// @return Set of cells in the band (sorted, unique)
template <typename Real>
std::vector<std::int32_t> expand_band_cells(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<const std::int32_t> seed_cells,
    int layers)
{
    auto topology = mesh.topology_mutable();
    const int tdim = topology->dim();
    
    // Create required connectivities
    topology->create_connectivity(tdim, tdim - 1);  // cell -> facet
    topology->create_connectivity(tdim - 1, tdim);  // facet -> cell
    
    auto cell_to_facet = topology->connectivity(tdim, tdim - 1);
    auto facet_to_cell = topology->connectivity(tdim - 1, tdim);
    
    // Use set for efficient lookup
    std::unordered_set<std::int32_t> band_set(seed_cells.begin(), seed_cells.end());
    std::vector<std::int32_t> current_front(seed_cells.begin(), seed_cells.end());
    
    for (int layer = 0; layer < layers && !current_front.empty(); ++layer)
    {
        std::vector<std::int32_t> next_front;
        
        for (std::int32_t cell : current_front)
        {
            // Get facets of this cell
            auto facets = cell_to_facet->links(cell);
            
            for (std::int32_t facet : facets)
            {
                // Get cells adjacent via this facet
                auto neighbors = facet_to_cell->links(facet);
                
                for (std::int32_t neighbor : neighbors)
                {
                    if (neighbor != cell && band_set.find(neighbor) == band_set.end())
                    {
                        band_set.insert(neighbor);
                        next_front.push_back(neighbor);
                    }
                }
            }
        }
        
        current_front = std::move(next_front);
    }
    
    // Convert to sorted vector
    std::vector<std::int32_t> result(band_set.begin(), band_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

/// Get all vertices belonging to band cells
/// @param mesh The background mesh
/// @param band_cells Cells in the band
/// @return Sorted unique list of vertex indices
template <typename Real>
std::vector<std::int32_t> band_vertices(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::span<const std::int32_t> band_cells)
{
    auto topology = mesh.topology_mutable();
    const int tdim = topology->dim();
    
    // Create cell -> vertex connectivity
    topology->create_connectivity(tdim, 0);
    auto cell_to_vertex = topology->connectivity(tdim, 0);
    
    std::unordered_set<std::int32_t> vertex_set;
    
    for (std::int32_t cell : band_cells)
    {
        auto vertices = cell_to_vertex->links(cell);
        for (std::int32_t v : vertices)
        {
            vertex_set.insert(v);
        }
    }
    
    std::vector<std::int32_t> result(vertex_set.begin(), vertex_set.end());
    std::sort(result.begin(), result.end());
    return result;
}

/// Get cells incident to a vertex
/// @param mesh The background mesh
/// @param vertex Vertex index
/// @return Span of incident cell indices
template <typename Real>
std::span<const std::int32_t> vertex_to_cells(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::int32_t vertex)
{
    auto topology = mesh.topology_mutable();
    const int tdim = topology->dim();
    
    // Ensure connectivity exists
    topology->create_connectivity(0, tdim);
    auto v_to_c = topology->connectivity(0, tdim);
    
    return v_to_c->links(vertex);
}

} // namespace cutfemx::level_set
