#pragma once
#include "stl_surface.h"
#include <string>
#include <dolfinx/mesh/Mesh.h>
#include <mpi.h>
#include "stl_reader.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stack>

namespace cutfemx::mesh {

struct DistributeSTLOptions
{
  double aabb_padding = 0.05;
  bool normalize_normals = true;
};

namespace {

// Overlap test, AABB [min, max]
template <typename T>
bool check_overlap_aabb(const T* a, const T b[6])
{
    // A: a[0..2] min, a[3..5] max
    // B: b[0..2] min, b[3..5] max
    
    // Check non-overlap in any dimension
    if (a[3] < b[0] || a[0] > b[3]) return false; // x
    if (a[4] < b[1] || a[1] > b[4]) return false; // y
    if (a[5] < b[2] || a[2] > b[5]) return false; // z
    
    return true;
}

} // namespace

template <typename Real>
TriSoup<Real> distribute_stl_facets(MPI_Comm comm,
                                   dolfinx::mesh::Mesh<Real>& mesh,
                                   const std::string& stl_path,
                                   const DistributeSTLOptions& opt)
{
    int rank = dolfinx::MPI::rank(comm);
    int size = dolfinx::MPI::size(comm);

    // 1. Compute global tree of rank AABBs using dolfinx
    int tdim = mesh.topology()->dim();

    // Workaround for dolfinx BoundingBoxTree constructor issue with const Mesh&
    // We compute the range of entities manually.
    mesh.topology_mutable()->create_entities(tdim);
    auto map = mesh.topology()->index_map(tdim);
    std::int32_t num_entities = map->size_local() + map->num_ghosts();
    std::vector<std::int32_t> entities(num_entities);
    std::iota(entities.begin(), entities.end(), 0);

    dolfinx::geometry::BoundingBoxTree<Real> local_tree(mesh, tdim, entities, opt.aabb_padding);
    
    // Compute global tree (each leaf is a rank)
    dolfinx::geometry::BoundingBoxTree<Real> global_tree = local_tree.create_global_tree(comm);

    // 2. Extract rank AABBs from global tree
    // We only need this on Rank 0 for routing? Yes.
    // Does create_global_tree broadcast the tree to everyone?
    // "Compute a global bounding tree (collective on comm)..."
    // Implementation does Allgather. So everyone has it.
    
    // Collect leaf boxes (Rank, AABB)
    struct RankBox {
        int rank;
        Real box[6]; // Flattened
    };
    std::vector<RankBox> rank_boxes;
    
    if (rank == 0) // Only needed on rank 0
    {
        std::int32_t root = global_tree.num_bboxes() - 1;
        if (root >= 0)
        {
            std::stack<std::int32_t> stack;
            stack.push(root);
            
            while (!stack.empty())
            {
                std::int32_t node = stack.top();
                stack.pop();
                
                std::array<std::int32_t, 2> children = global_tree.bbox(node);
                if (children[0] == children[1])
                {
                    // Leaf: value is rank
                    RankBox rb;
                    rb.rank = children[0];
                    std::array<Real, 6> b = global_tree.get_bbox(node);
                    std::copy(b.begin(), b.end(), rb.box);
                    rank_boxes.push_back(rb);
                }
                else
                {
                    stack.push(children[0]);
                    stack.push(children[1]);
                }
            }
        }
    }

    // 3. Read STL on rank 0
    std::vector<std::int32_t> tri_gid_read;
    std::vector<Real> facet_data_read;
    
    if (rank == 0)
    {
        read_stl_facets(stl_path, tri_gid_read, facet_data_read);
        
        // Optional normalization
        if (opt.normalize_normals)
        {
            std::size_t num_tris = tri_gid_read.size();
            for (std::size_t t = 0; t < num_tris; ++t)
            {
                Real* n = &facet_data_read[12 * t];
                Real len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                if (len > 1e-30)
                {
                    n[0] /= len;
                    n[1] /= len;
                    n[2] /= len;
                }
            }
        }
    }

    // 4. Determine routing (Rank 0 only)
    std::vector<std::int32_t> send_counts_T(size, 0); 
    std::vector<std::vector<std::int32_t>> send_gids_bucket(size);
    std::vector<std::vector<Real>> send_facets_bucket(size);
    
    if (rank == 0)
    {
        std::size_t num_tris = tri_gid_read.size();
        for (std::size_t t = 0; t < num_tris; ++t)
        {
            // Compute triangle AABB
            const Real* data = &facet_data_read[12 * t];
            // Vertices start at offset 3
            const Real* v0 = &data[3];
            const Real* v1 = &data[6];
            const Real* v2 = &data[9];
            
            Real tri_aabb[6];
            tri_aabb[0] = std::min({v0[0], v1[0], v2[0]});
            tri_aabb[1] = std::min({v0[1], v1[1], v2[1]});
            tri_aabb[2] = std::min({v0[2], v1[2], v2[2]});
            tri_aabb[3] = std::max({v0[0], v1[0], v2[0]});
            tri_aabb[4] = std::max({v0[1], v1[1], v2[1]});
            tri_aabb[5] = std::max({v0[2], v1[2], v2[2]});
            
            // Check overlap with each rank box
            // Note: rank_boxes is sparse if some ranks are skipped? No, create_global_tree should cover all if they sent something.
            // But if a rank has no mesh entities, it might send "max" box.
            // check_overlap will fail for inverted box.
            
            for (const auto& rb : rank_boxes)
            {
                if (check_overlap_aabb(tri_aabb, rb.box))
                {
                    int r = rb.rank;
                    // Safey check
                    if (r >= 0 && r < size) {
                        send_gids_bucket[r].push_back(tri_gid_read[t]);
                        send_facets_bucket[r].insert(send_facets_bucket[r].end(), data, data + 12);
                        send_counts_T[r]++;
                    }
                }
            }
        }
    }

    // 5. MPI Exchange (Same as before)
    std::vector<std::int32_t> recv_counts_T(size);
    MPI_Alltoall(send_counts_T.data(), 1, MPI_INT,
                 recv_counts_T.data(), 1, MPI_INT, comm);
                 
    int my_nT = std::accumulate(recv_counts_T.begin(), recv_counts_T.end(), 0);

    // Prepare send buffers
    std::vector<std::int32_t> send_gids_flat;
    std::vector<Real> send_facets_flat;
    std::vector<int> send_counts_gids(size), send_displs_gids(size);
    std::vector<int> send_counts_facets(size), send_displs_facets(size);
    
    if (rank == 0)
    {
        int offset_gids = 0;
        int offset_facets = 0;
        for (int r = 0; r < size; ++r)
        {
            send_counts_gids[r] = static_cast<int>(send_gids_bucket[r].size());
            send_displs_gids[r] = offset_gids;
            send_gids_flat.insert(send_gids_flat.end(), send_gids_bucket[r].begin(), send_gids_bucket[r].end());
            offset_gids += send_counts_gids[r];
            
            send_counts_facets[r] = static_cast<int>(send_facets_bucket[r].size());
            send_displs_facets[r] = offset_facets;
            send_facets_flat.insert(send_facets_flat.end(), send_facets_bucket[r].begin(), send_facets_bucket[r].end());
            offset_facets += send_counts_facets[r];
        }
    }
    
    // Prepare recv buffers
    std::vector<int> recv_counts_gids(size), recv_displs_gids(size);
    std::vector<int> recv_counts_facets(size), recv_displs_facets(size);
    
    int offset_gids_r = 0;
    int offset_facets_r = 0;
    for (int r=0; r<size; ++r)
    {
        recv_counts_gids[r] = recv_counts_T[r]; 
        recv_displs_gids[r] = offset_gids_r;
        offset_gids_r += recv_counts_gids[r];
        
        recv_counts_facets[r] = recv_counts_T[r] * 12;
        recv_displs_facets[r] = offset_facets_r;
        offset_facets_r += recv_counts_facets[r];
    }
    
    TriSoup<Real> soup;
    soup.tri_gid.resize(my_nT);
    std::vector<Real> recv_facets(my_nT * 12);
    
    MPI_Alltoallv(send_gids_flat.data(), send_counts_gids.data(), send_displs_gids.data(), MPI_INT,
                  soup.tri_gid.data(), recv_counts_gids.data(), recv_displs_gids.data(), MPI_INT, comm);
                  
    MPI_Alltoallv(send_facets_flat.data(), send_counts_facets.data(), send_displs_facets.data(), dolfinx::MPI::mpi_type<Real>(),
                  recv_facets.data(), recv_counts_facets.data(), recv_displs_facets.data(), dolfinx::MPI::mpi_type<Real>(), comm);

    // 6. Unpack
    soup.tri.resize(my_nT * 3);
    soup.X.resize(my_nT * 9);
    soup.N.resize(my_nT * 3);
    
    for (int t = 0; t < my_nT; ++t)
    {
        soup.tri[3*t + 0] = 3*t + 0;
        soup.tri[3*t + 1] = 3*t + 1;
        soup.tri[3*t + 2] = 3*t + 2;
        
        const Real* src = &recv_facets[12*t];
        
        soup.N[3*t+0] = src[0];
        soup.N[3*t+1] = src[1];
        soup.N[3*t+2] = src[2];
        
        soup.X[9*t + 0] = src[3];
        soup.X[9*t + 1] = src[4];
        soup.X[9*t + 2] = src[5];
        
        soup.X[9*t + 3] = src[6];
        soup.X[9*t + 4] = src[7];
        soup.X[9*t + 5] = src[8];
        
        soup.X[9*t + 6] = src[9];
        soup.X[9*t + 7] = src[10];
        soup.X[9*t + 8] = src[11];
    }

    return soup;
}

} // namespace cutfemx::mesh
