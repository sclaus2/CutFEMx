#pragma once

// Parallel Min-Exchange: synchronize distance values across MPI ranks
// Uses dolfinx IndexMap scatter functions for ghost synchronization

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>

#include <span>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <iostream>

namespace cutfemx::level_set {

/// Get MPI datatype for a given C++ type
template <typename T>
MPI_Datatype get_mpi_type() {
    if constexpr (std::is_same_v<T, double>)
        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, float>)
        return MPI_FLOAT;
    else
        static_assert(sizeof(T) == 0, "Unsupported type for MPI");
}

/// Synchronize distance array: ghost→owner min reduce, then owner→ghost forward
/// @param index_map Vertex index map from mesh topology
/// @param dist Distance array (size = owned + ghosts). Modified in-place.
template <typename Real>
void min_exchange_vertices(
    const dolfinx::common::IndexMap& index_map,
    std::span<Real> dist)
{
    MPI_Comm comm = index_map.comm();
    const int mpi_size = dolfinx::MPI::size(comm);
    const int mpi_rank = dolfinx::MPI::rank(comm);
    
    if (mpi_size == 1)
        return;  // No communication needed for serial
    
    const std::int32_t size_local = index_map.size_local();
    const std::int32_t num_ghosts = index_map.num_ghosts();
    
    if (num_ghosts == 0)
        return;  // No ghosts to exchange
    
    // Get ghost owners and their global indices
    std::span<const int> ghost_owners = index_map.owners();
    std::span<const std::int64_t> ghosts = index_map.ghosts();
    
    MPI_Datatype mpi_real = get_mpi_type<Real>();
    const Real inf_val = Real(1e30);
    
    // ============================================================
    // Step 1: Ghost → Owner (reverse scatter with min reduction)
    // ============================================================
    
    // Count messages per destination
    std::vector<int> send_counts(mpi_size, 0);
    for (int owner : ghost_owners)
        send_counts[owner]++;
    
    std::vector<int> send_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        send_displs[i + 1] = send_displs[i] + send_counts[i];
    
    // Pack ghost data
    std::vector<Real> send_vals(num_ghosts, inf_val);
    std::vector<std::int64_t> send_global_ids(num_ghosts);
    std::vector<int> cursors(mpi_size, 0);
    
    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        send_vals[idx] = dist[size_local + g];
        send_global_ids[idx] = ghosts[g];
    }
    
    // Exchange counts
    std::vector<int> recv_counts(mpi_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, 
                 recv_counts.data(), 1, MPI_INT, comm);
    
    // DEBUG: Log exchange volume
    int total_send = 0;
    // for(int c : send_counts) total_send += c;
    // if (total_send > 0 || mpi_rank == 0) { // Log if active or rank 0
    //    std::cout << "[Exch Rank " << mpi_rank << "] Sending " << total_send << " ghost values. ";
    //    for(int r=0; r<mpi_size; ++r) if(send_counts[r] > 0) std::cout << "->R" << r << ":" << send_counts[r] << " ";
    //    std::cout << std::endl;
    // }

    std::vector<int> recv_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    
    int total_recv = recv_displs[mpi_size];
    
    std::vector<Real> recv_vals(total_recv, inf_val);
    std::vector<std::int64_t> recv_global_ids(total_recv);
    
    // Exchange values
    MPI_Alltoallv(send_vals.data(), send_counts.data(), send_displs.data(), 
                  mpi_real, recv_vals.data(), recv_counts.data(), recv_displs.data(), 
                  mpi_real, comm);
    
    MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(), 
                  MPI_INT64_T, recv_global_ids.data(), recv_counts.data(), recv_displs.data(), 
                  MPI_INT64_T, comm);
    
    // DEBUG: Log received finite values
    int finite_recv = 0;
    for(Real v : recv_vals) if(v < inf_val) finite_recv++;
    // if (finite_recv > 0) std::cout << "[Exch Rank " << mpi_rank << "] Received " << finite_recv << " finite values from ghosts." << std::endl;

    // Apply min to owned vertices
    auto local_range = index_map.local_range();
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1])
        {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            
            // Debug: track updates
            // if (recv_vals[i] < dist[local_id]) 
            //    std::cout << "[Exch Rank " << mpi_rank << "] Updating Owned " << local_id << " " << dist[local_id] << " -> " << recv_vals[i] << std::endl;
                
            dist[local_id] = std::min(dist[local_id], recv_vals[i]);
        }
    }
    
    // ============================================================
    // Step 2: Owner → Ghost (forward scatter)
    // ============================================================
    
    // Pack owned values for replies - INITIALIZE TO INF not 0!
    std::vector<Real> owned_reply(total_recv, inf_val);
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1])
        {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            owned_reply[i] = dist[local_id];
        }
    }
    
    // Send owned values back - INITIALIZE TO INF!
    std::vector<Real> ghost_update(num_ghosts, inf_val);
    MPI_Alltoallv(owned_reply.data(), recv_counts.data(), recv_displs.data(), 
                  mpi_real, ghost_update.data(), send_counts.data(), send_displs.data(), 
                  mpi_real, comm);
    
    // DEBUG: Log received updates
    // int update_recv = 0;
    // for(Real v : ghost_update) if(v < inf_val) update_recv++;
    // if (update_recv > 0) std::cout << "[Exch Rank " << mpi_rank << "] Received " << update_recv << " finite updates for ghosts." << std::endl;

    // Update ghost values - MIRROR OWNER STRICTLY
    // This prevents "ghost ahead of owner" leakage. 
    // Ghosts should be exact copies of owner state after sync.
    
    std::fill(cursors.begin(), cursors.end(), 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        
        // Owner sent back their current value for this vertex.
        // If finite, we MUST overwrite, even if our local ghost value is smaller.
        // Why? Because our local ghost value comes from incomplete info (ghost cells).
        // The owner has the truth (or the best current estimate of it).
        
        if (ghost_update[idx] < inf_val)
        {
            dist[size_local + g] = ghost_update[idx];
        }
        else 
        {
            // Owner sent INF.
            // If we have a finite value locally (seeded by NearField on this rank),
            // should we keep it? 
            // YES, because this might be the *source* of the value that will propagate 
            // to the owner in the next iteration!
            // But if it's not a source, it might be junk.
            // Conservative: Keep it. The next "Ghost->Owner" step will send it to owner.
        }
    }
}

} // namespace cutfemx::level_set
