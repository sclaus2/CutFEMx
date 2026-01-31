#pragma once

// Cell-Triangle Map: for each local mesh cell, find all surface triangles that intersect it
// Uses dolfinx bounding box trees for candidate generation and robust predicates for filtering

#include "stl_surface.h"
#include "tri_intersection.h"

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>

#include <vector>
#include <array>
#include <span>
#include <algorithm>
#include <cstdint>

namespace cutfemx::mesh {

/// Options for cell-triangle map computation
struct CellTriangleMapOptions {
    double aabb_padding = 0.0;  ///< Extra padding for AABB overlap detection
};

/// Result structure: CSR format map from cells to local triangle indices
struct CellTriangleMap {
    std::vector<std::int32_t> offsets;  ///< Size = num_local_cells + 1
    std::vector<std::int32_t> data;     ///< Triangle indices (local to this rank)
    
    /// Number of cells in the map
    std::size_t num_cells() const { return offsets.empty() ? 0 : offsets.size() - 1; }
    
    /// Get triangle indices for a given cell
    std::span<const std::int32_t> triangles(std::int32_t cell) const {
        return std::span<const std::int32_t>(
            data.data() + offsets[cell],
            data.data() + offsets[cell + 1]);
    }
};

namespace detail {

/// Compute AABB for a single triangle
template <typename Real>
std::array<Real, 6> triangle_aabb(const Real* v0, const Real* v1, const Real* v2)
{
    std::array<Real, 6> box;
    box[0] = std::min({v0[0], v1[0], v2[0]});
    box[1] = std::min({v0[1], v1[1], v2[1]});
    box[2] = std::min({v0[2], v1[2], v2[2]});
    box[3] = std::max({v0[0], v1[0], v2[0]});
    box[4] = std::max({v0[1], v1[1], v2[1]});
    box[5] = std::max({v0[2], v1[2], v2[2]});
    return box;
}

/// Check if two AABBs overlap (with optional padding)
template <typename Real>
bool aabb_overlap(const Real* a, const Real* b, Real padding = 0)
{
    return !(a[3] + padding < b[0] - padding ||
             a[0] - padding > b[3] + padding ||
             a[4] + padding < b[1] - padding ||
             a[1] - padding > b[4] + padding ||
             a[5] + padding < b[2] - padding ||
             a[2] - padding > b[5] + padding);
}

/// Get cell vertex coordinates
template <typename Real>
void get_cell_vertices(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::int32_t cell,
    std::vector<std::array<Real, 3>>& vertices)
{
    const dolfinx::mesh::Geometry<Real>& geometry = mesh.geometry();
    std::span<const Real> x = geometry.x();
    auto dofmap = geometry.dofmap();
    
    // dofmap is an mdspan, access via (cell, local_dof)
    const std::size_t num_dofs_per_cell = dofmap.extent(1);
    vertices.resize(num_dofs_per_cell);
    
    for (std::size_t i = 0; i < num_dofs_per_cell; ++i)
    {
        std::int32_t node = dofmap(cell, i);
        vertices[i][0] = x[3*node + 0];
        vertices[i][1] = x[3*node + 1];
        vertices[i][2] = x[3*node + 2];
    }
}

/// Compute AABB for a mesh cell
template <typename Real>
std::array<Real, 6> cell_aabb(
    const dolfinx::mesh::Mesh<Real>& mesh,
    std::int32_t cell)
{
    std::vector<std::array<Real, 3>> verts;
    get_cell_vertices(mesh, cell, verts);
    
    std::array<Real, 6> box;
    box[0] = box[1] = box[2] = std::numeric_limits<Real>::max();
    box[3] = box[4] = box[5] = std::numeric_limits<Real>::lowest();
    
    for (const auto& v : verts)
    {
        box[0] = std::min(box[0], v[0]);
        box[1] = std::min(box[1], v[1]);
        box[2] = std::min(box[2], v[2]);
        box[3] = std::max(box[3], v[0]);
        box[4] = std::max(box[4], v[1]);
        box[5] = std::max(box[5], v[2]);
    }
    return box;
}

} // namespace detail

/// Build cell-to-triangle map
/// @param mesh The background mesh (distributed)
/// @param soup The distributed surface triangles (local portion)
/// @param opt Options controlling the algorithm
/// @return CSR map from local cells to local triangle indices
template <typename Real>
CellTriangleMap build_cell_triangle_map(
    const dolfinx::mesh::Mesh<Real>& mesh,
    const TriSoup<Real>& soup,
    const CellTriangleMapOptions& opt = {})
{
    CellTriangleMap result;
    
    // Get mesh info
    auto topology = mesh.topology();
    const int tdim = topology->dim();
    auto cell_map = topology->index_map(tdim);
    
    // CRITICAL FIX: Include ghost cells for parallel FMM NearField initialization
    // Ghost cells may contain surface triangles that initialize owned vertices
    const std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    
    const std::int32_t num_tris = soup.nT();
    
    if (num_cells == 0 || num_tris == 0)
    {
        result.offsets.resize(num_cells + 1, 0);
        return result;
    }
    
    // Step 1: Build AABBs for all surface triangles
    std::vector<std::array<Real, 6>> tri_aabbs(num_tris);
    
    for (std::int32_t t = 0; t < num_tris; ++t)
    {
        // Get triangle vertex indices
        std::int32_t i0 = soup.tri[3*t + 0];
        std::int32_t i1 = soup.tri[3*t + 1];
        std::int32_t i2 = soup.tri[3*t + 2];
        
        // Get vertex coordinates
        const Real* v0 = &soup.X[3*i0];
        const Real* v1 = &soup.X[3*i1];
        const Real* v2 = &soup.X[3*i2];
        
        tri_aabbs[t] = detail::triangle_aabb(v0, v1, v2);
    }
    
    // Step 2: Build AABBs for ALL cells (local + ghost)
    std::vector<std::array<Real, 6>> cell_aabbs(num_cells);
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
        cell_aabbs[c] = detail::cell_aabb(mesh, c);
    }
    
    // Step 3: For each cell, find candidate triangles via AABB overlap
    // Then filter with robust predicates
    
    result.offsets.resize(num_cells + 1);
    result.offsets[0] = 0;
    
    // Get cell type
    dolfinx::mesh::CellType cell_type = topology->cell_type();
    
    std::vector<std::array<Real, 3>> cell_verts;
    
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
        const auto& c_aabb = cell_aabbs[c];
        detail::get_cell_vertices(mesh, c, cell_verts);
        
        // Build vertex pointer array for intersection tests
        std::vector<const Real*> vert_ptrs(cell_verts.size());
        for (std::size_t i = 0; i < cell_verts.size(); ++i)
            vert_ptrs[i] = cell_verts[i].data();
        
        for (std::int32_t t = 0; t < num_tris; ++t)
        {
            const auto& t_aabb = tri_aabbs[t];
            
            // Quick AABB rejection
            if (!detail::aabb_overlap(c_aabb.data(), t_aabb.data(), 
                                      static_cast<Real>(opt.aabb_padding)))
                continue;
            
            // Get triangle vertex coordinates
            std::int32_t i0 = soup.tri[3*t + 0];
            std::int32_t i1 = soup.tri[3*t + 1];
            std::int32_t i2 = soup.tri[3*t + 2];
            
            std::array<Real, 3> tv0 = {soup.X[3*i0], soup.X[3*i0+1], soup.X[3*i0+2]};
            std::array<Real, 3> tv1 = {soup.X[3*i1], soup.X[3*i1+1], soup.X[3*i1+2]};
            std::array<Real, 3> tv2 = {soup.X[3*i2], soup.X[3*i2+1], soup.X[3*i2+2]};
            const Real* tri_ptrs[3] = {tv0.data(), tv1.data(), tv2.data()};
            
            // Robust intersection test based on cell type
            bool intersects = false;
            
            if (cell_type == dolfinx::mesh::CellType::tetrahedron)
            {
                intersects = triangle_tet_intersect(tri_ptrs, vert_ptrs.data());
            }
            else if (cell_type == dolfinx::mesh::CellType::hexahedron)
            {
                intersects = triangle_hex_intersect(tri_ptrs, vert_ptrs.data());
            }
            else
            {
                // For other cell types, conservatively assume intersection
                // if AABBs overlap
                intersects = true;
            }
            
            if (intersects)
            {
                result.data.push_back(t);
            }
        }
        
        result.offsets[c + 1] = static_cast<std::int32_t>(result.data.size());
    }
    
    return result;
}

} // namespace cutfemx::mesh

