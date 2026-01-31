#pragma once

// Vertex Graph: build vertex neighbor adjacency from mesh edges
// Used for FarField propagation (graph-based distance approximation)

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/Geometry.h>

#include <vector>
#include <span>
#include <cmath>

namespace cutfemx::level_set {

/// Vertex adjacency from mesh edges in CSR format
struct VertexGraph {
    std::vector<std::int32_t> offsets;   ///< Size = num_vertices + 1
    std::vector<std::int32_t> neighbors; ///< Neighbor indices (local)
    std::vector<double> weights;         ///< Edge weights (Euclidean distance)
    
    /// Number of vertices
    std::size_t num_vertices() const { 
        return offsets.empty() ? 0 : offsets.size() - 1; 
    }
    
    /// Get neighbors of vertex v
    std::span<const std::int32_t> get_neighbors(std::int32_t v) const {
        return std::span<const std::int32_t>(
            neighbors.data() + offsets[v],
            neighbors.data() + offsets[v + 1]);
    }
    
    /// Get edge weights for vertex v
    std::span<const double> get_weights(std::int32_t v) const {
        return std::span<const double>(
            weights.data() + offsets[v],
            weights.data() + offsets[v + 1]);
    }
};

/// Build vertex neighbor graph from mesh edge connectivity
/// @param mesh The background mesh
/// @return VertexGraph with neighbors and edge weights
template <typename Real>
VertexGraph build_vertex_graph(const dolfinx::mesh::Mesh<Real>& mesh)
{
    VertexGraph graph;
    
    // We infer the graph directly from Cell->Vertex connectivity.
    // This is robust for distributed meshes because we know cells (dim) and vertices (0)
    // are fully populated including ghosts, whereas intermediate entities (edges)
    // might not be generated for ghost cells.
    
    auto topology = mesh.topology_mutable();
    const int dim = topology->dim();
    const int tdim = dim; // alias
    
    // Ensure cell->vertex connectivity exists
    topology->create_connectivity(dim, 0);
    auto cell_to_vertex = topology->connectivity(dim, 0);
    
    // Get vertex coordinates
    const dolfinx::mesh::Geometry<Real>& geometry = mesh.geometry();
    std::span<const Real> x = geometry.x();
    
    // Get number of local + ghost vertices
    auto vertex_map = topology->index_map(0);
    const std::int32_t num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
    
    // Get number of local + ghost cells
    auto cell_map = topology->index_map(dim);
    const std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    
    // Temporary adjacency list (using vector for potential duplicates, will sort/unique later)
    std::vector<std::vector<std::int32_t>> adj(num_vertices);
    
    // Iterate ALL cells (including ghosts)
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
        auto verts = cell_to_vertex->links(c);
        
        // For a tetrahedron (or triangle), all vertices are connected to each other
        // (Clique connectivity)
        for (std::size_t i = 0; i < verts.size(); ++i)
        {
            std::int32_t u = verts[i];
            
            // Only store neighbors for vertices we track (local + ghost)
            // (Should be all of them, but safety check)
            if (u >= num_vertices) continue;
            
            for (std::size_t j = 0; j < verts.size(); ++j)
            {
                if (i == j) continue;
                std::int32_t v = verts[j];
                
                adj[u].push_back(v);
            }
        }
    }
    
    // Flatten into CSR format
    graph.offsets.resize(num_vertices + 1);
    graph.offsets[0] = 0;
    
    // Count unique neighbors and fill
    for (std::int32_t i = 0; i < num_vertices; ++i)
    {
        auto& neighbors = adj[i];
        
        // Sort and remove duplicates
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        
        graph.offsets[i + 1] = graph.offsets[i] + neighbors.size();
    }
    
    graph.neighbors.resize(graph.offsets[num_vertices]);
    graph.weights.resize(graph.offsets[num_vertices]);
    
    // Fill neighbors and compute weights
    for (std::int32_t i = 0; i < num_vertices; ++i)
    {
        Real ux = x[3*i + 0];
        Real uy = x[3*i + 1];
        Real uz = x[3*i + 2];
        
        const auto& neighbors = adj[i];
        std::int32_t start = graph.offsets[i];
        
        for (std::size_t k = 0; k < neighbors.size(); ++k)
        {
            std::int32_t v = neighbors[k];
            graph.neighbors[start + k] = v;
            
            Real vx = x[3*v + 0];
            Real vy = x[3*v + 1];
            Real vz = x[3*v + 2];
            
            Real dx = ux - vx;
            Real dy = uy - vy;
            Real dz = uz - vz;
            
            graph.weights[start + k] = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
    
    return graph;
}

} // namespace cutfemx::level_set
