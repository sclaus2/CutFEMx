// Copyright (c) 2022-2024 ONERA
// Authors: Susanne Claus
// This file is part of CutFEMx
//
// SPDX-License-Identifier:    MIT
#include "create_mesh.h"

#include <cutcells/utils.h>
#include <cutcells/cell_topology.h>

#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

namespace cutfemx::mesh
{
  namespace
  {
    struct VertexKey
    {
      // 0: original vertex (global vertex id)
      // 1: edge intersection (global vertex id pair)
      // 2: special point (parent cell id + sid)
      uint8_t kind = 0;
      int32_t a = 0;
      int32_t b = 0;

      bool operator==(const VertexKey& other) const noexcept
      {
        return kind == other.kind && a == other.a && b == other.b;
      }
    };

    struct VertexKeyHash
    {
      std::size_t operator()(const VertexKey& k) const noexcept
      {
        // Basic mix; good enough for small integer keys.
        const std::size_t h0 = std::hash<int32_t>{}(k.a);
        const std::size_t h1 = std::hash<int32_t>{}(k.b);
        const std::size_t hk = std::hash<uint8_t>{}(k.kind);
        return h0 ^ (h1 + 0x9e3779b97f4a7c15ULL + (h0 << 6) + (h0 >> 2)) ^ (hk << 1);
      }
    };

    template <std::floating_point T>
    inline void append_vertex_coords(std::vector<T>& out, const std::vector<T>& in,
                    int local_vertex_id, int gdim)
    {
      const int base = local_vertex_id * gdim;
      for (int j = 0; j < gdim; ++j)
        out.push_back(in[base + j]);
    }
  }

  // Copy of create_mesh from utils.h in dolfinx with added return of reordered cell numbering
  // this cell reordering information is necessary to update the parent map for cut cells
  template <typename U>
  std::tuple<dolfinx::mesh::Mesh<U>, std::vector<std::int64_t>>
  create_mesh(
      MPI_Comm comm, MPI_Comm commt, std::span<const std::int64_t> cells,
      const dolfinx::fem::CoordinateElement<U>& element,
      MPI_Comm commg, const std::span<U> x, std::array<std::size_t, 2> xshape,
      const dolfinx::mesh::CellPartitionFunction& partitioner)
  {
    dolfinx::mesh::CellType celltype = element.cell_shape();
    const dolfinx::fem::ElementDofLayout doflayout = element.create_dof_layout();

    const int num_cell_vertices = dolfinx::mesh::num_cell_vertices(element.cell_shape());
    std::size_t num_cell_nodes = doflayout.num_dofs();

    // Note: `extract_topology` extracts topology data, i.e. just the
    // vertices. For P1 geometry this should just be the identity
    // operator. For other elements the filtered lists may have 'gaps',
    // i.e. the indices might not be contiguous.
    //
    // `extract_topology` could be skipped for 'P1 geometry' elements

    // -- Partition topology across ranks of comm
    std::vector<std::int64_t> cells1;
    std::vector<std::int64_t> original_idx1;
    std::vector<int> ghost_owners;
    std::int64_t offset = 0;

    if (partitioner)
    {
      spdlog::info("Using partitioner with {} cell data", cells.size());
      dolfinx::graph::AdjacencyList<std::int32_t> dest(0);
      if (commt != MPI_COMM_NULL)
      {
        int size = dolfinx::MPI::size(comm);
        auto t = extract_topology(element.cell_shape(), doflayout, cells);
        dest = partitioner(commt, size, {celltype}, {t});
      }

      // Distribute cells (topology, includes higher-order 'nodes') to
      // destination rank
      assert(cells.size() % num_cell_nodes == 0);
      std::size_t num_cells = cells.size() / num_cell_nodes;
      std::vector<int> src_ranks;
      std::tie(cells1, src_ranks, original_idx1, ghost_owners)
          = dolfinx::graph::build::distribute(comm, cells, {num_cells, num_cell_nodes},
                                    dest);
      spdlog::debug("Got {} cells from distribution", cells1.size());
    }
    else
    {
      cells1 = std::vector<std::int64_t>(cells.begin(), cells.end());
      assert(cells1.size() % num_cell_nodes == 0);
      std::int64_t num_owned = cells1.size() / num_cell_nodes;
      MPI_Exscan(&num_owned, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
      original_idx1.resize(num_owned);
      std::iota(original_idx1.begin(), original_idx1.end(), offset);
    }

    // Extract cell 'topology', i.e. extract the vertices for each cell
    // and discard any 'higher-order' nodes
    std::vector<std::int64_t> cells1_v
        = extract_topology(celltype, doflayout, cells1);
    spdlog::info("Extract basic topology: {}->{}", cells1.size(),
                cells1_v.size());

    // Build local dual graph for owned cells to (i) get list of vertices
    // on the process boundary and (ii) apply re-ordering to cells for
    // locality
    std::vector<std::int64_t> boundary_v;
    {
      std::int32_t num_owned_cells
          = cells1_v.size() / num_cell_vertices - ghost_owners.size();
      std::vector<std::int32_t> cell_offsets(num_owned_cells + 1, 0);
      for (std::size_t i = 1; i < cell_offsets.size(); ++i)
        cell_offsets[i] = cell_offsets[i - 1] + num_cell_vertices;
      spdlog::info("Build local dual graph");
      auto [graph, unmatched_facets, max_v, facet_attached_cells]
          = build_local_dual_graph(
              std::vector{celltype},
              {std::span(cells1_v.data(), num_owned_cells * num_cell_vertices)});
      const std::vector<int> remap = dolfinx::graph::reorder_gps(graph);

      // Create re-ordered cell lists (leaves ghosts unchanged)
      std::vector<std::int64_t> _original_idx(original_idx1.size());
      for (std::size_t i = 0; i < remap.size(); ++i)
        _original_idx[remap[i]] = original_idx1[i];
      std::copy_n(std::next(original_idx1.cbegin(), num_owned_cells),
                  ghost_owners.size(),
                  std::next(_original_idx.begin(), num_owned_cells));
      dolfinx::mesh::impl::reorder_list(
          std::span(cells1_v.data(), remap.size() * num_cell_vertices), remap);
      dolfinx::mesh::impl::reorder_list(std::span(cells1.data(), remap.size() * num_cell_nodes),
                        remap);
      original_idx1 = _original_idx;

      // Boundary vertices are marked as 'unknown'
      boundary_v = unmatched_facets;
      std::ranges::sort(boundary_v);
      auto [unique_end, range_end] = std::ranges::unique(boundary_v);
      boundary_v.erase(unique_end, range_end);

      // Remove -1 if it occurs in boundary vertices (may occur in mixed
      // topology)
      if (!boundary_v.empty() > 0 and boundary_v[0] == -1)
        boundary_v.erase(boundary_v.begin());
    }

    // Create Topology
    dolfinx::mesh::Topology topology = create_topology(comm, cells1_v, original_idx1,
                                        ghost_owners, celltype, boundary_v);

    // Create connectivities required higher-order geometries for creating
    // a Geometry object
    for (int e = 1; e < topology.dim(); ++e)
      if (doflayout.num_entity_dofs(e) > 0)
        topology.create_entities(e);
    if (element.needs_dof_permutations())
      topology.create_entity_permutations();

    // Build list of unique (global) node indices from cells1 and
    // distribute coordinate data
    std::vector<std::int64_t> nodes1 = cells1;
    dolfinx::radix_sort(nodes1);
    auto [unique_end, range_end] = std::ranges::unique(nodes1);
    nodes1.erase(unique_end, range_end);

    std::vector coords
        = dolfinx::MPI::distribute_data(comm, nodes1, commg, x, xshape[1]);

    // Create geometry object
    dolfinx::mesh::Geometry geometry
        = create_geometry(topology, element, nodes1, cells1, coords, xshape[1]);

    //keep track of reordering but keep local numbering of cells
    std::vector<std::int64_t> cell_idx_reordered_wo_offset(original_idx1.size());
    for(std::size_t i = 0 ; i < original_idx1.size() ; ++i)
      cell_idx_reordered_wo_offset[i] = original_idx1[i]-offset;


    return {dolfinx::mesh::Mesh(comm, std::make_shared<dolfinx::mesh::Topology>(std::move(topology)),
                std::move(geometry)),cell_idx_reordered_wo_offset};
  }

  //Merge vertex coordinates
  // input: cut_cells and vertex coordinates
  // returns: updated vertex coordinates and vertex_map
  template <std::floating_point T> void merge_vertex_coords(cutcells::mesh::CutCells<T>& cut_cells,
                std::vector<T>& vertex_coords,
                std::vector<std::vector<int>>& vertex_map)
  {
    // Guard against empty cut cells
    if(cut_cells._cut_cells.empty())
      return;
      
    // Find first non-empty cut cell to get gdim
    std::size_t gdim = 0;
    for(const auto& cut_cell : cut_cells._cut_cells)
    {
      if(cut_cell._vertex_coords.size() > 0)
      {
        gdim = cut_cell._gdim;
        break;
      }
    }
    
    if(gdim == 0) return;
    
    std::unordered_map<VertexKey, int, VertexKeyHash> global_vertex_ids;
    // Pre-reserve to avoid reallocations; rough estimate of unique vertices.
    // In 3D, unique vertices are approx 1/6 of total vertex accesses in simplices.
    global_vertex_ids.reserve(cut_cells._cut_cells.size() * 4); 

    int cut_cell_id = 0;
    for(auto & cut_cell : cut_cells._cut_cells)
    {
      if(cut_cell._vertex_coords.size()==0)
      {
        cut_cell_id++;
        continue;
      }

      int num_cut_cell_vertices = cut_cell._vertex_coords.size()/gdim;
      vertex_map[cut_cell_id].resize(num_cut_cell_vertices);

      // Determine whether we can use fast token-based dedup for this cut cell.
      const bool has_tokens = (static_cast<int>(cut_cell._vertex_parent_entity.size()) == num_cut_cell_vertices);
      const int parent_vertices = cutcells::cell::get_num_vertices(cut_cell._parent_cell_type);
      const bool has_parent_ids = (static_cast<int>(cut_cell._parent_vertex_ids.size()) == parent_vertices);
      const int parent_cell_id = cut_cells._parent_map.empty() ? -1 : cut_cells._parent_map[cut_cell_id];
      const bool can_fast_dedup = has_tokens && has_parent_ids && (parent_cell_id >= 0);

      if (can_fast_dedup)
      {
        const auto parent_edges = cutcells::cell::edges(cut_cell._parent_cell_type);
        for (int local_id = 0; local_id < num_cut_cell_vertices; ++local_id)
        {
          const int32_t token = cut_cell._vertex_parent_entity[local_id];
          VertexKey key;

          if (token >= 100 && token < 200)
          {
            const int ref_vid = static_cast<int>(token - 100);
            if (ref_vid < 0 || ref_vid >= parent_vertices)
            {
              key.kind = 2; key.a = parent_cell_id; key.b = token;
            }
            else
            {
              key.kind = 0;
              key.a = static_cast<int32_t>(cut_cell._parent_vertex_ids[ref_vid]);
              key.b = 0;
            }
          }
          else if (token >= 200)
          {
            key.kind = 2; key.a = parent_cell_id; key.b = static_cast<int32_t>(token - 200);
          }
          else
          {
            const int edge_id = static_cast<int>(token);
            if (edge_id < 0 || edge_id >= static_cast<int>(parent_edges.size()))
            {
              key.kind = 2; key.a = parent_cell_id; key.b = token;
            }
            else
            {
              const int v0 = parent_edges[edge_id][0];
              const int v1 = parent_edges[edge_id][1];
              const int32_t gv0 = static_cast<int32_t>(cut_cell._parent_vertex_ids[v0]);
              const int32_t gv1 = static_cast<int32_t>(cut_cell._parent_vertex_ids[v1]);
              key.kind = 1;
              key.a = std::min(gv0, gv1);
              key.b = std::max(gv0, gv1);
            }
          }

          auto it = global_vertex_ids.find(key);
          if (it == global_vertex_ids.end())
          {
            const int merged_vertex_id = static_cast<int>(vertex_coords.size() / gdim);
            append_vertex_coords<T>(vertex_coords, cut_cell._vertex_coords, local_id, static_cast<int>(gdim));
            global_vertex_ids.emplace(key, merged_vertex_id);
            vertex_map[cut_cell_id][local_id] = merged_vertex_id;
          }
          else
          {
            vertex_map[cut_cell_id][local_id] = it->second;
          }
        }
      }
      else
      {
        // Fallback: coordinate-based dedup
        for (int local_id = 0; local_id < num_cut_cell_vertices; ++local_id)
        {
          int id = cutcells::utils::vertex_exists<T>(vertex_coords, cut_cell._vertex_coords, local_id, gdim);
          if (id == -1)
          {
            int merged_vertex_id = static_cast<int>(vertex_coords.size() / gdim);
            for (int j = 0; j < gdim; j++) vertex_coords.push_back(cut_cell._vertex_coords[local_id * gdim + j]);
            vertex_map[cut_cell_id][local_id] = merged_vertex_id;
          }
          else
          {
            vertex_map[cut_cell_id][local_id] = id;
          }
        }
      }
      cut_cell_id++;
    }
  }

  //return Mesh and parent cell map as this can be used for interpolation
  template <std::floating_point T>
  CutMesh<T> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<T>& cut_cells)
  {
    CutMesh<T> cut_mesh;

    // Guard against completely empty cut cells mesh
    if(cut_cells._cut_cells.empty())
    {
      return cut_mesh;
    }
    
    // Find first non-empty cut cell to get gdim/tdim/cell_type
    std::size_t first_valid = 0;
    bool found_valid = false;
    for(std::size_t i = 0; i < cut_cells._cut_cells.size(); ++i)
    {
      if(cut_cells._cut_cells[i]._vertex_coords.size() > 0)
      {
        first_valid = i;
        found_valid = true;
        break;
      }
    }
    
    // If all cut cells are empty, return empty mesh
    if(!found_valid)
    {
      return cut_mesh;
    }

    //create local meshes with local vertex numbering
    std::size_t gdim = cut_cells._cut_cells[first_valid]._gdim;
    std::size_t tdim = cut_cells._cut_cells[first_valid]._tdim;
    auto cell_type = cutcells_to_dolfinx_cell_type(cut_cells._cut_cells[first_valid]._types[0]);
    const int num_cell_vertices = dolfinx::mesh::num_cell_vertices(cell_type);

    //Count the total number of cells in vector
    int num_cells =0;

    for(auto & cut_cell :  cut_cells._cut_cells)
    {
      // purpose here is to exclude ghosts
      if(cut_cell._vertex_coords.size()>0)
        num_cells += cut_cell._connectivity.size();
    }
    
    // If no valid cells, return empty mesh
    if(num_cells == 0)
    {
      return cut_mesh;
    }

    //Merged vertex coordinates
    // connectivity regarding merged vertex coordinates
    // parent index for cells
    std::vector<T> vertex_coords;
    std::vector<std::vector<int>> vertex_map(cut_cells._cut_cells.size());

    merge_vertex_coords(cut_cells, vertex_coords, vertex_map);

    // Determine offset between processors
    // this is because dolfinx expects global vertex numbering
    // Note that the current global numbering scheme doubles vertices at
    // processor boundaries
    int num_vertices_local = vertex_coords.size()/gdim;
    std::int64_t v_offset(0), num_v_owned(num_vertices_local);
    MPI_Exscan(&num_v_owned, &v_offset, 1, MPI_INT64_T, MPI_SUM, comm);

    /// @todo: this only works for mesh with one cell type
    std::vector<int64_t> cells(num_cells*num_cell_vertices);
    std::vector<int> original_parent_index(num_cells);
    int v_cnt = 0; //vertex counter
    int c_cnt = 0; //local cell counter
    int cut_cell_id = 0; // cut cell counter

    //create flattened connectivity vector with global vertex numbering
    //Starting from new local vertex numbering and adding the offset
    for(auto & cut_cell : cut_cells._cut_cells)
    {
      if((cut_cell._vertex_coords.size()==0))
      {
        continue;
      }

      for(std::size_t i=0;i<cut_cell._connectivity.size();i++)
      {
        for(std::size_t j=0;j<cut_cell._connectivity[i].size();j++)
        {
          cells[v_cnt] = vertex_map[cut_cell_id][cut_cell._connectivity[i][j]]+v_offset;
          v_cnt++;
        }
        original_parent_index[c_cnt] = cut_cells._parent_map[cut_cell_id];
        c_cnt++;
      }
      cut_cell_id++;
    }

    //create mesh
    dolfinx::fem::CoordinateElement<T> element(cell_type, 1);
    std::array<std::size_t, 2> xshape = {vertex_coords.size() / gdim, gdim};

    //create global vertex numbering and call create_mesh function based on dolfinx implementation
    const auto [dolfinx_cut_mesh, original_idx] = create_mesh<T>(comm, comm, cells, element, comm, vertex_coords, xshape, nullptr);

    //determine parent map for reordered cells
    cut_mesh._cut_mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(dolfinx_cut_mesh);
    cut_mesh._parent_index.resize(num_cells);
    cut_mesh._is_cut_cell.assign(num_cells, true);

    for(std::size_t i=0;i<original_idx.size();i++)
    {
      cut_mesh._parent_index[i] = original_parent_index[original_idx[i]];
    }

    // std::cout << "original_parent_index=[";
    // for (const auto& i: original_parent_index)
    //     std::cout << i << ',';
    // std::cout << "]" << std::endl;

    // std::cout << "parent_index=[";
    // for (const auto& i: parent_index)
    //     std::cout << i << ',';
    // std::cout << "]" << std::endl;

    return cut_mesh;
  }

  template <std::floating_point T>
  CutMesh<T> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<T>& cut_cells,
                                  const dolfinx::mesh::Mesh<T>& mesh, std::span<const std::int32_t> entities)
  {
    CutMesh<T> cut_mesh;
    cut_mesh._bg_mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(mesh);

    //get vertices and cells from background mesh entities
    auto tdim = mesh.topology()->dim();
    std::size_t gdim = mesh.geometry().dim();
    auto entity_map = mesh.topology()->index_map(tdim);

    // Find first non-empty cut cell to validate dimensions (if any exist)
    std::size_t first_valid = 0;
    bool found_valid = false;
    for(std::size_t i = 0; i < cut_cells._cut_cells.size(); ++i)
    {
      if(cut_cells._cut_cells[i]._vertex_coords.size() > 0 && !cut_cells._cut_cells[i]._types.empty())
      {
        first_valid = i;
        found_valid = true;
        break;
      }
    }
    
    // Only check assertions if we have valid cut cells
    if(found_valid)
    {
      assert(tdim == cut_cells._cut_cells[first_valid]._tdim);
      assert(gdim == cut_cells._cut_cells[first_valid]._gdim);
    }

    //get list of all vertices connected to entities
    mesh.topology_mutable()->create_connectivity(0, tdim);
    const std::vector<std::int32_t>& vertices = dolfinx::mesh::compute_incident_entities(*mesh.topology(), entities, tdim, 0);
    const std::vector<std::int32_t>& vertex_xdofs = dolfinx::mesh::entities_to_geometry(mesh, 0, vertices, false);

    //get physical location of vertices
    auto x_dofmap = mesh.geometry().dofmap();
    std::span<const T> x_nodes = mesh.geometry().x();

    //intialise vertex coordinates with number of vertices
    std::vector<T> vertex_coords(vertices.size()*gdim);
    std::map<std::int32_t,std::int32_t> entity_vertex_map;
    int v_cnt=0;

    // add all vertex coordinates and keep track of mapping
    for(std::size_t i=0;i<vertices.size();i++)
    {
      const auto vertex_coord = x_nodes.subspan(vertex_xdofs[i]*3, 3);

      for(std::size_t j=0;j<gdim;j++)
      {
        vertex_coords[v_cnt*gdim+j] = vertex_coord[j];
      }

      entity_vertex_map.insert(std::make_pair(vertices[i], v_cnt));
      v_cnt++;
    }

    // add vertices of cut_mesh and keep track of vertex map
    std::vector<std::vector<int>> vertex_map(cut_cells._cut_cells.size());
    merge_vertex_coords(cut_cells, vertex_coords, vertex_map);

    // at this point we have a list of vertices and two maps from the original vertex id
    // the new vertex id now we generate an offset between processors
    /// @todo: prevent doubling of vertices at processor boundaries
    int num_vertices_local = vertex_coords.size()/gdim;
    std::int64_t v_offset(0), num_v_owned(num_vertices_local);
    MPI_Exscan(&num_v_owned, &v_offset, 1, MPI_INT64_T, MPI_SUM, comm);

    // Now determine cells with new vertex numbering which corresponds location in vertex coords vector
    int num_cells = entities.size();

    for(auto & cut_cell :  cut_cells._cut_cells)
    {
      // purpose here is to exclude ghosts
      if((cut_cell._vertex_coords.size()>0))
        num_cells += cut_cell._connectivity.size();
    }

    // Get cell type - use first valid cut cell, or fall back to background mesh cell type
    dolfinx::mesh::CellType cell_type;
    if(found_valid)
    {
      cell_type = cutcells_to_dolfinx_cell_type(cut_cells._cut_cells[first_valid]._types[0]);
    }
    else
    {
      // No valid cut cells - use background mesh cell type
      cell_type = mesh.topology()->cell_type();
    }
    const int num_cell_vertices = dolfinx::mesh::num_cell_vertices(cell_type);

    std::vector<int64_t> cells(num_cells*num_cell_vertices);
    std::vector<int> original_parent_index(num_cells);
    std::vector<bool> original_is_cut_cell(num_cells);

    v_cnt = 0; //vertex counter
    int c_cnt = 0; //local cell counter
    int cut_cell_id = 0; // cut cell counter

    // first copy entity connectivity into cells with new vertex numbering
    auto c_to_v = mesh.topology()->connectivity(tdim, 0);
    for(auto& entity : entities)
    {
      //get vertices attached to entity
      std::span<const int> original_vertex_ids = c_to_v->links(entity);
      for(auto& vertex : original_vertex_ids)
      {
        cells[v_cnt] = entity_vertex_map[vertex]+v_offset;
        v_cnt++;
      }

      original_parent_index[c_cnt] = entity;
      original_is_cut_cell[c_cnt] = false;
      c_cnt++;
    }

    //create flattened connectivity vector with global vertex numbering
    //Starting from new local vertex numbering and adding the offset
    for(auto & cut_cell : cut_cells._cut_cells)
    {
      if((cut_cell._vertex_coords.size()==0))
      {
        continue;
      }

      for(std::size_t i=0;i<cut_cell._connectivity.size();i++)
      {
        for(std::size_t j=0;j<cut_cell._connectivity[i].size();j++)
        {
          cells[v_cnt] = vertex_map[cut_cell_id][cut_cell._connectivity[i][j]]+v_offset;
          v_cnt++;
        }
        original_parent_index[c_cnt] = cut_cells._parent_map[cut_cell_id];
        original_is_cut_cell[c_cnt] = true;
        c_cnt++;
      }
      cut_cell_id++;
    }

    //create mesh
    dolfinx::fem::CoordinateElement<T> element(cell_type, 1);
    std::array<std::size_t, 2> xshape = {vertex_coords.size() / gdim, gdim};

    //create global vertex numbering and call create_mesh function based on dolfinx implementation
    const auto [dolfinx_cut_mesh, original_idx] = create_mesh<T>(comm, comm, cells, element, comm, vertex_coords, xshape, nullptr);

    //determine parent map and if it is a cut cell for reordered cells
    cut_mesh._cut_mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(dolfinx_cut_mesh);
    cut_mesh._parent_index.resize(num_cells);
    cut_mesh._is_cut_cell.resize(num_cells);

    for(std::size_t i=0;i<original_idx.size();i++)
    {
      cut_mesh._parent_index[i] = original_parent_index[original_idx[i]];
      cut_mesh._is_cut_cell[i] = original_is_cut_cell[original_idx[i]];
    }

    return cut_mesh;
  }

  //----------------------------------------------------------------------
  template
  CutMesh<double> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<double>& cut_cells);

  template
  CutMesh<float> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<float>& cut_cells);

  template
  CutMesh<double> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<double>& cut_cells,
                                  const dolfinx::mesh::Mesh<double>& mesh, std::span<const std::int32_t> entities);

  template
  CutMesh<float> create_cut_mesh(MPI_Comm comm, cutcells::mesh::CutCells<float>& cut_cells,
                                  const dolfinx::mesh::Mesh<float>& mesh, std::span<const std::int32_t> entities);

  //----------------------------------------------------------------------
}