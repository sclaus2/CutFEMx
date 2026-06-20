// Copyright (c) 2026 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT

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
#include <cmath>

namespace cutfemx::distance {

/// Get MPI datatype for a given C++ type
template <typename T>
MPI_Datatype get_mpi_type() {
    if constexpr (std::is_same_v<T, double>)
        return MPI_DOUBLE;
    else if constexpr (std::is_same_v<T, float>)
        return MPI_FLOAT;
    else if constexpr (std::is_same_v<T, std::int32_t>)
        return MPI_INT32_T;
    else if constexpr (std::is_same_v<T, std::int64_t>)
        return MPI_INT64_T;
    else
        static_assert(sizeof(T) == 0, "Unsupported type for MPI");
}

/// Synchronize distance array: ghost→owner min reduce, then owner→ghost MIN
/// 
/// FIXED: Now uses consistent infinity value and applies MIN rule for ghost update
/// instead of strict owner mirror. This prevents overwriting valid local updates
/// that haven't reached the owner yet.
///
/// @param index_map Vertex index map from mesh topology
/// @param dist Distance array (size = owned + ghosts). Modified in-place.
/// @param inf_value Value considered as "infinity" (uninitialized). Default: max<Real>.
template <typename Real>
void min_exchange_vertices(
    const dolfinx::common::IndexMap& index_map,
    std::span<Real> dist,
    Real inf_value = std::numeric_limits<Real>::max())
{
    MPI_Comm comm = index_map.comm();
    const int mpi_size = dolfinx::MPI::size(comm);
    
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
    
    // Pack ghost data - send inf_value for uninitialized
    std::vector<Real> send_vals(num_ghosts, inf_value);
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

    std::vector<int> recv_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    
    int total_recv = recv_displs[mpi_size];
    
    std::vector<Real> recv_vals(total_recv, inf_value);
    std::vector<std::int64_t> recv_global_ids(total_recv);
    
    // Exchange values
    MPI_Alltoallv(send_vals.data(), send_counts.data(), send_displs.data(), 
                  mpi_real, recv_vals.data(), recv_counts.data(), recv_displs.data(), 
                  mpi_real, comm);
    
    MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(), 
                  MPI_INT64_T, recv_global_ids.data(), recv_counts.data(), recv_displs.data(), 
                  MPI_INT64_T, comm);

    // Apply min to owned vertices (only if received value is finite)
    auto local_range = index_map.local_range();
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1])
        {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            
            // Only update if received value is finite (less than inf_value)
            if (recv_vals[i] < inf_value) {
                dist[local_id] = std::min(dist[local_id], recv_vals[i]);
            }
        }
    }
    
    // ============================================================
    // Step 2: Owner → Ghost (forward scatter with MIN rule)
    // ============================================================
    
    // Pack owned values for replies
    std::vector<Real> owned_reply(total_recv, inf_value);
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1])
        {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            owned_reply[i] = dist[local_id];
        }
    }
    
    // Send owned values back
    std::vector<Real> ghost_update(num_ghosts, inf_value);
    MPI_Alltoallv(owned_reply.data(), recv_counts.data(), recv_displs.data(), 
                  mpi_real, ghost_update.data(), send_counts.data(), send_displs.data(), 
                  mpi_real, comm);

    // Update ghost values - USE MIN RULE (not strict mirror!)
    // This is critical for FIM stability: a ghost might have a better value
    // from local computation that hasn't propagated to owner yet.
    // We take the minimum to ensure monotonic convergence.
    
    std::fill(cursors.begin(), cursors.end(), 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        
        // MIN rule: take the smaller of ghost's current value and owner's value
        if (ghost_update[idx] < inf_value) {
            dist[size_local + g] = std::min(dist[size_local + g], ghost_update[idx]);
        }
        // If owner sent inf, keep our local value (might be a source)
    }
}

/// Synchronize distance plus transported payload. The payload associated with
/// the minimum distance is copied together with that distance.
template <typename Real>
void min_exchange_vertices_payload(
    const dolfinx::common::IndexMap& index_map,
    std::span<Real> dist,
    std::span<Real> speed,
    std::span<Real> normals,
    int normal_dim,
    Real inf_value = std::numeric_limits<Real>::max())
{
    MPI_Comm comm = index_map.comm();
    const int mpi_size = dolfinx::MPI::size(comm);
    if (mpi_size == 1)
        return;

    const std::int32_t size_local = index_map.size_local();
    const std::int32_t num_ghosts = index_map.num_ghosts();
    if (num_ghosts == 0)
        return;

    std::span<const int> ghost_owners = index_map.owners();
    std::span<const std::int64_t> ghosts = index_map.ghosts();
    MPI_Datatype mpi_real = get_mpi_type<Real>();

    std::vector<int> send_counts(mpi_size, 0);
    for (int owner : ghost_owners)
        send_counts[owner]++;

    std::vector<int> send_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        send_displs[i + 1] = send_displs[i] + send_counts[i];

    const int payload_width = 1 + normal_dim;
    std::vector<Real> send_vals(num_ghosts, inf_value);
    std::vector<Real> send_payload(static_cast<std::size_t>(num_ghosts)
                                   * payload_width, Real(0));
    std::vector<std::int64_t> send_global_ids(num_ghosts);
    std::vector<int> cursors(mpi_size, 0);

    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        const std::int32_t local = size_local + g;
        send_vals[idx] = dist[local];
        send_payload[static_cast<std::size_t>(idx) * payload_width]
            = speed[local];
        for (int d = 0; d < normal_dim; ++d)
        {
            send_payload[static_cast<std::size_t>(idx) * payload_width + 1
                         + static_cast<std::size_t>(d)]
                = normals[static_cast<std::size_t>(local) * normal_dim
                          + static_cast<std::size_t>(d)];
        }
        send_global_ids[idx] = ghosts[g];
    }

    std::vector<int> recv_counts(mpi_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> recv_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    int total_recv = recv_displs[mpi_size];

    std::vector<Real> recv_vals(total_recv, inf_value);
    std::vector<Real> recv_payload(static_cast<std::size_t>(total_recv)
                                   * payload_width, Real(0));
    std::vector<std::int64_t> recv_global_ids(total_recv);

    MPI_Alltoallv(send_vals.data(), send_counts.data(), send_displs.data(),
                  mpi_real, recv_vals.data(), recv_counts.data(),
                  recv_displs.data(), mpi_real, comm);
    MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(),
                  MPI_INT64_T, recv_global_ids.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT64_T, comm);

    std::vector<int> send_payload_counts(mpi_size, 0);
    std::vector<int> recv_payload_counts(mpi_size, 0);
    std::vector<int> send_payload_displs(mpi_size + 1, 0);
    std::vector<int> recv_payload_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
    {
        send_payload_counts[i] = send_counts[i] * payload_width;
        recv_payload_counts[i] = recv_counts[i] * payload_width;
        send_payload_displs[i + 1]
            = send_payload_displs[i] + send_payload_counts[i];
        recv_payload_displs[i + 1]
            = recv_payload_displs[i] + recv_payload_counts[i];
    }
    MPI_Alltoallv(send_payload.data(), send_payload_counts.data(),
                  send_payload_displs.data(), mpi_real, recv_payload.data(),
                  recv_payload_counts.data(), recv_payload_displs.data(),
                  mpi_real, comm);

    auto local_range = index_map.local_range();
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1]
            && recv_vals[i] < inf_value)
        {
            std::int32_t local_id =
                static_cast<std::int32_t>(global_id - local_range[0]);
            if (recv_vals[i] < dist[local_id])
            {
                dist[local_id] = recv_vals[i];
                speed[local_id]
                    = recv_payload[static_cast<std::size_t>(i) * payload_width];
                for (int d = 0; d < normal_dim; ++d)
                {
                    normals[static_cast<std::size_t>(local_id) * normal_dim
                            + static_cast<std::size_t>(d)]
                        = recv_payload[static_cast<std::size_t>(i) * payload_width
                                       + 1 + static_cast<std::size_t>(d)];
                }
            }
        }
    }

    std::vector<Real> owned_reply(total_recv, inf_value);
    std::vector<Real> payload_reply(static_cast<std::size_t>(total_recv)
                                    * payload_width, Real(0));
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1])
        {
            std::int32_t local_id =
                static_cast<std::int32_t>(global_id - local_range[0]);
            owned_reply[i] = dist[local_id];
            payload_reply[static_cast<std::size_t>(i) * payload_width]
                = speed[local_id];
            for (int d = 0; d < normal_dim; ++d)
            {
                payload_reply[static_cast<std::size_t>(i) * payload_width + 1
                              + static_cast<std::size_t>(d)]
                    = normals[static_cast<std::size_t>(local_id) * normal_dim
                              + static_cast<std::size_t>(d)];
            }
        }
    }

    std::vector<Real> ghost_update(num_ghosts, inf_value);
    std::vector<Real> ghost_payload_update(static_cast<std::size_t>(num_ghosts)
                                           * payload_width, Real(0));
    MPI_Alltoallv(owned_reply.data(), recv_counts.data(), recv_displs.data(),
                  mpi_real, ghost_update.data(), send_counts.data(),
                  send_displs.data(), mpi_real, comm);
    MPI_Alltoallv(payload_reply.data(), recv_payload_counts.data(),
                  recv_payload_displs.data(), mpi_real,
                  ghost_payload_update.data(), send_payload_counts.data(),
                  send_payload_displs.data(), mpi_real, comm);

    std::fill(cursors.begin(), cursors.end(), 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        const std::int32_t local = size_local + g;
        if (ghost_update[idx] < dist[local])
        {
            dist[local] = ghost_update[idx];
            speed[local]
                = ghost_payload_update[static_cast<std::size_t>(idx)
                                       * payload_width];
            for (int d = 0; d < normal_dim; ++d)
            {
                normals[static_cast<std::size_t>(local) * normal_dim
                        + static_cast<std::size_t>(d)]
                    = ghost_payload_update[static_cast<std::size_t>(idx)
                                           * payload_width + 1
                                           + static_cast<std::size_t>(d)];
            }
        }
    }
}

/// Strict owner→ghost copy (no MIN rule)
/// Use this after FIM convergence to ensure ghost values exactly match owners
/// for consistent VTK I/O.
/// @param index_map Vertex index map from mesh topology
/// @param dist Distance array (size = owned + ghosts). Modified in-place.
template <typename Real>
void copy_owner_to_ghost(
    const dolfinx::common::IndexMap& index_map,
    std::span<Real> dist)
{
    MPI_Comm comm = index_map.comm();
    const int mpi_size = dolfinx::MPI::size(comm);
    
    if (mpi_size <= 1) return; // Nothing to do in serial
    
    const std::int32_t size_local = index_map.size_local();
    const std::int32_t num_ghosts = index_map.num_ghosts();
    
    std::span<const int> ghost_owners = index_map.owners();
    std::span<const std::int64_t> ghosts_global = index_map.ghosts();
    
    MPI_Datatype mpi_real = get_mpi_type<Real>();
    
    // Count messages per destination
    std::vector<int> send_counts(mpi_size, 0);
    for (int owner : ghost_owners) send_counts[owner]++;
    
    std::vector<int> send_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i) send_displs[i + 1] = send_displs[i] + send_counts[i];
    
    // Pack ghost global IDs to request from owners
    std::vector<std::int64_t> send_global_ids(num_ghosts);
    std::vector<int> cursors(mpi_size, 0);
    
    for (std::int32_t g = 0; g < num_ghosts; ++g) {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        send_global_ids[idx] = ghosts_global[g];
    }
    
    // Exchange counts
    std::vector<int> recv_counts(mpi_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);
    
    std::vector<int> recv_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i) recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    
    int total_recv = recv_displs[mpi_size];
    std::vector<std::int64_t> recv_global_ids(total_recv);
    
    // Exchange global IDs
    MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(), MPI_INT64_T,
                  recv_global_ids.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T, comm);
    
    // Prepare owner values to send back
    auto local_range = index_map.local_range();
    std::vector<Real> owned_reply(total_recv);
    
    for (int i = 0; i < total_recv; ++i) {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1]) {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            owned_reply[i] = dist[local_id];
        }
    }
    
    // Send owned values back to ghosts
    std::vector<Real> ghost_update(num_ghosts);
    MPI_Alltoallv(owned_reply.data(), recv_counts.data(), recv_displs.data(), mpi_real,
                  ghost_update.data(), send_counts.data(), send_displs.data(), mpi_real, comm);
    
    // STRICT COPY: Overwrite ghost values with owner values (no MIN)
    std::fill(cursors.begin(), cursors.end(), 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g) {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        dist[size_local + g] = ghost_update[idx];
    }
}

template <typename Real>
void copy_owner_to_ghost_payload(
    const dolfinx::common::IndexMap& index_map,
    std::span<Real> dist,
    std::span<Real> speed,
    std::span<Real> normals,
    int normal_dim)
{
    MPI_Comm comm = index_map.comm();
    const int mpi_size = dolfinx::MPI::size(comm);
    if (mpi_size <= 1)
        return;

    const std::int32_t size_local = index_map.size_local();
    const std::int32_t num_ghosts = index_map.num_ghosts();
    if (num_ghosts == 0)
        return;

    std::span<const int> ghost_owners = index_map.owners();
    std::span<const std::int64_t> ghosts_global = index_map.ghosts();
    MPI_Datatype mpi_real = get_mpi_type<Real>();

    std::vector<int> send_counts(mpi_size, 0);
    for (int owner : ghost_owners)
        send_counts[owner]++;

    std::vector<int> send_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        send_displs[i + 1] = send_displs[i] + send_counts[i];

    std::vector<std::int64_t> send_global_ids(num_ghosts);
    std::vector<int> cursors(mpi_size, 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        send_global_ids[idx] = ghosts_global[g];
    }

    std::vector<int> recv_counts(mpi_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 comm);

    std::vector<int> recv_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
        recv_displs[i + 1] = recv_displs[i] + recv_counts[i];
    int total_recv = recv_displs[mpi_size];
    std::vector<std::int64_t> recv_global_ids(total_recv);

    MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(),
                  MPI_INT64_T, recv_global_ids.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT64_T, comm);

    const int payload_width = 2 + normal_dim;
    std::vector<Real> owned_reply(static_cast<std::size_t>(total_recv)
                                  * payload_width, Real(0));
    auto local_range = index_map.local_range();
    for (int i = 0; i < total_recv; ++i)
    {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1])
        {
            std::int32_t local_id =
                static_cast<std::int32_t>(global_id - local_range[0]);
            const std::size_t base = static_cast<std::size_t>(i) * payload_width;
            owned_reply[base] = dist[local_id];
            owned_reply[base + 1] = speed[local_id];
            for (int d = 0; d < normal_dim; ++d)
            {
                owned_reply[base + 2 + static_cast<std::size_t>(d)]
                    = normals[static_cast<std::size_t>(local_id) * normal_dim
                              + static_cast<std::size_t>(d)];
            }
        }
    }

    std::vector<int> send_payload_counts(mpi_size, 0);
    std::vector<int> recv_payload_counts(mpi_size, 0);
    std::vector<int> send_payload_displs(mpi_size + 1, 0);
    std::vector<int> recv_payload_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i)
    {
        send_payload_counts[i] = send_counts[i] * payload_width;
        recv_payload_counts[i] = recv_counts[i] * payload_width;
        send_payload_displs[i + 1]
            = send_payload_displs[i] + send_payload_counts[i];
        recv_payload_displs[i + 1]
            = recv_payload_displs[i] + recv_payload_counts[i];
    }

    std::vector<Real> ghost_update(static_cast<std::size_t>(num_ghosts)
                                   * payload_width, Real(0));
    MPI_Alltoallv(owned_reply.data(), recv_payload_counts.data(),
                  recv_payload_displs.data(), mpi_real, ghost_update.data(),
                  send_payload_counts.data(), send_payload_displs.data(),
                  mpi_real, comm);

    std::fill(cursors.begin(), cursors.end(), 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g)
    {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        const std::int32_t local = size_local + g;
        const std::size_t base = static_cast<std::size_t>(idx) * payload_width;
        dist[local] = ghost_update[base];
        speed[local] = ghost_update[base + 1];
        for (int d = 0; d < normal_dim; ++d)
        {
            normals[static_cast<std::size_t>(local) * normal_dim
                    + static_cast<std::size_t>(d)]
                = ghost_update[base + 2 + static_cast<std::size_t>(d)];
        }
    }
}

/// Synchronize boolean array using Logical OR
/// @param index_map Facet index map
/// @param vals Boolean array (size = owned + ghosts). Modified in-place.
inline void or_exchange_bool(
    const dolfinx::common::IndexMap& index_map,
    std::vector<bool>& vals)
{
    MPI_Comm comm = index_map.comm();
    const int mpi_size = dolfinx::MPI::size(comm);

    if (mpi_size == 1) return;

    const std::int32_t size_local = index_map.size_local();
    const std::int32_t num_ghosts = index_map.num_ghosts();
    if (num_ghosts == 0) return;

    // Convert bool to int8 for MPI
    std::vector<std::int8_t> dist(vals.size());
    for(size_t i=0; i<vals.size(); ++i) dist[i] = vals[i] ? 1 : 0;

    // Get ghost owners and their global indices
    std::span<const int> ghost_owners = index_map.owners();
    std::span<const std::int64_t> ghosts = index_map.ghosts();

    // Step 1: Ghost -> Owner (OR reduction)
    std::vector<int> send_counts(mpi_size, 0);
    for (int owner : ghost_owners) send_counts[owner]++;

    std::vector<int> send_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i) send_displs[i + 1] = send_displs[i] + send_counts[i];

    std::vector<std::int8_t> send_vals(num_ghosts);
    std::vector<std::int64_t> send_global_ids(num_ghosts);
    std::vector<int> cursors(mpi_size, 0);

    for (std::int32_t g = 0; g < num_ghosts; ++g) {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        send_vals[idx] = dist[size_local + g];
        send_global_ids[idx] = ghosts[g];
    }

    std::vector<int> recv_counts(mpi_size);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> recv_displs(mpi_size + 1, 0);
    for (int i = 0; i < mpi_size; ++i) recv_displs[i + 1] = recv_displs[i] + recv_counts[i];

    int total_recv = recv_displs[mpi_size];
    std::vector<std::int8_t> recv_vals(total_recv);
    std::vector<std::int64_t> recv_global_ids(total_recv);

    MPI_Alltoallv(send_vals.data(), send_counts.data(), send_displs.data(), MPI_INT8_T, 
                  recv_vals.data(), recv_counts.data(), recv_displs.data(), MPI_INT8_T, comm);
    MPI_Alltoallv(send_global_ids.data(), send_counts.data(), send_displs.data(), MPI_INT64_T, 
                  recv_global_ids.data(), recv_counts.data(), recv_displs.data(), MPI_INT64_T, comm);

    // Apply OR to owned
    auto local_range = index_map.local_range();
    for (int i = 0; i < total_recv; ++i) {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1]) {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            if (recv_vals[i]) dist[local_id] = 1;
        }
    }

    // Step 2: Owner -> Ghost (Mirror)
    std::vector<std::int8_t> owned_reply(total_recv);
    for (int i = 0; i < total_recv; ++i) {
        std::int64_t global_id = recv_global_ids[i];
        if (global_id >= local_range[0] && global_id < local_range[1]) {
            std::int32_t local_id = static_cast<std::int32_t>(global_id - local_range[0]);
            owned_reply[i] = dist[local_id];
        }
    }

    std::vector<std::int8_t> ghost_update(num_ghosts);
    MPI_Alltoallv(owned_reply.data(), recv_counts.data(), recv_displs.data(), MPI_INT8_T, 
                  ghost_update.data(), send_counts.data(), send_displs.data(), MPI_INT8_T, comm);

    std::fill(cursors.begin(), cursors.end(), 0);
    for (std::int32_t g = 0; g < num_ghosts; ++g) {
        int owner = ghost_owners[g];
        int idx = send_displs[owner] + cursors[owner]++;
        // For booleans, OR is appropriate (if owner says true, we become true)
        dist[size_local + g] = std::max(dist[size_local + g], ghost_update[idx]);
    }

    // Convert back to bool
    for(size_t i=0; i<vals.size(); ++i) vals[i] = (dist[i] != 0);
}

} // namespace cutfemx::distance
