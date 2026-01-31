#pragma once

// Geometry Mapping Utility: Centralized Vertex Mappings
// Avoids rebuilding the same expensive mappings in multiple files.

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <vector>
#include <span>

namespace cutfemx::level_set {

/// Cache for commonly needed vertex mappings
/// Build once, reuse in fmm.h, sign.h, winding.h, etc.
template <typename Real>
struct VertexMapCache {
    std::vector<std::int32_t> vert_to_geom;   // Topology vertex -> Geometry DOF
    std::vector<std::int32_t> vert_to_dof;    // Topology vertex -> FunctionSpace DOF
    std::vector<Real> vertex_coords;           // Contiguous xyz for all vertices
    std::int32_t num_vertices;
    std::int32_t num_owned;
    
    /// Build cache from mesh and function space
    void build(const dolfinx::mesh::Mesh<Real>& mesh,
               const dolfinx::fem::FunctionSpace<Real>* function_space = nullptr)
    {
        auto topology = mesh.topology();
        const int tdim = topology->dim();
        
        auto vertex_map = topology->index_map(0);
        num_vertices = vertex_map->size_local() + vertex_map->num_ghosts();
        num_owned = vertex_map->size_local();
        
        // Ensure connectivity
        if (!topology->connectivity(tdim, 0)) {
            mesh.topology_mutable()->create_connectivity(tdim, 0);
        }
        auto cell_to_vertex = topology->connectivity(tdim, 0);
        
        const Real* x = mesh.geometry().x().data();
        auto geom_dofmap = mesh.geometry().dofmap();
        
        // Build vert_to_geom and vertex_coords
        vert_to_geom.assign(num_vertices, -1);
        vertex_coords.resize(3 * num_vertices);
        
        for (std::int32_t c = 0; c < static_cast<std::int32_t>(geom_dofmap.extent(0)); ++c) {
            auto c_verts = cell_to_vertex->links(c);
            for (std::size_t i = 0; i < c_verts.size(); ++i) {
                std::int32_t v = c_verts[i];
                if (v < num_vertices) {
                    std::int32_t g = geom_dofmap(c, i);
                    vert_to_geom[v] = g;
                    vertex_coords[3*v]   = x[3*g];
                    vertex_coords[3*v+1] = x[3*g+1];
                    vertex_coords[3*v+2] = x[3*g+2];
                }
            }
        }
        
        // Build vert_to_dof if function space provided
        if (function_space) {
            vert_to_dof.assign(num_vertices, -1);
            auto dofmap = function_space->dofmap();
            const int num_cells = topology->index_map(tdim)->size_local() 
                                + topology->index_map(tdim)->num_ghosts();
            
            for (std::int32_t c = 0; c < num_cells; ++c) {
                auto c_verts = cell_to_vertex->links(c);
                auto c_dofs = dofmap->cell_dofs(c);
                for (std::size_t i = 0; i < std::min(c_verts.size(), c_dofs.size()); ++i) {
                    std::int32_t v = c_verts[i];
                    if (v < num_vertices) {
                        vert_to_dof[v] = c_dofs[i];
                    }
                }
            }
        }
    }
    
    /// Get coordinates for vertex v
    const Real* coords(std::int32_t v) const {
        return &vertex_coords[3*v];
    }
    
    /// Check if vertex has valid geometry mapping
    bool has_geom(std::int32_t v) const {
        return v < num_vertices && vert_to_geom[v] >= 0;
    }
    
    /// Check if vertex has valid DOF mapping
    bool has_dof(std::int32_t v) const {
        return v < num_vertices && !vert_to_dof.empty() && vert_to_dof[v] >= 0;
    }
};

} // namespace cutfemx::level_set
