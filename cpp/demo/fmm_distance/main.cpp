/// Demo: FMM Unsigned Distance Computation
/// 
/// Computes unsigned distance from a sphere surface to all mesh vertices
/// using the FMM pipeline: NearField (triangles) + FarField (Dijkstra).

#include <vector>
#include <span>
#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <csignal>

#include <mpi.h>

#include <basix/finite-element.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/io/VTKFile.h>

#include <cutfemx/mesh/stl_surface.h>
#include <cutfemx/mesh/cell_triangle_map.h>
#include <cutfemx/mesh/distribute_stl.h>
#include <cutfemx/level_set/fmm.h>
#include <cutfemx/level_set/sign.h>

using T = double;
using namespace dolfinx;

int main(int argc, char* argv[])
{
    // Avoid process termination if stdout/stderr are connected to a pipe
    // that closes early (common in MPI launchers and when piping output).
    std::signal(SIGPIPE, SIG_IGN);

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Keep all MPI-using objects (e.g. dolfinx, VTKFile) in a scope that
    // ends before MPI_Finalize(). Some destructors perform MPI collectives.
    {

    
    if (rank == 0)
    {
        std::cout << "=== FMM Unsigned Distance Demo ===\n";
        std::cout << "Running on " << size << " MPI ranks\n";
    }
    
    // Create background mesh (unit cube with tetrahedra)
    int N = 16;  // Resolution
    // Use shared_vertex for deeper ghosts as requested by user
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex); 
    auto bg_mesh = std::make_shared<mesh::Mesh<T>>(
        mesh::create_box<T>(MPI_COMM_WORLD, 
                           {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}},
                           {N, N, N}, 
                           mesh::CellType::tetrahedron, part));
                           
    // DEBUG: Verify IndexMap Assumption
    auto vertex_map = bg_mesh->topology()->index_map(0);
    if (vertex_map) {
        std::int32_t num_owned = vertex_map->size_local();
        std::int32_t num_ghosts = vertex_map->num_ghosts();
        auto local_range = vertex_map->local_range();
        
        if (local_range[1] - local_range[0] != num_owned) {
             std::cerr << "Rank " << rank << ": WARNING IndexMap Range mismatch! " 
                       << (local_range[1]-local_range[0]) << " vs " << num_owned << std::endl;
        }
        
        // Check Coordinates of Ghosts
        const auto& geom = bg_mesh->geometry();
        std::span<const T> x = geom.x();
        
        if (num_ghosts > 0) {
            double ghost_x_sum = 0;
            int zero_ghosts = 0;
            for(int i=0; i<num_ghosts; ++i) {
                int idx = 3*(num_owned + i);
                double val = std::abs(x[idx]) + std::abs(x[idx+1]) + std::abs(x[idx+2]);
                ghost_x_sum += val;
                if(val < 1e-9) zero_ghosts++;
            }
            if (zero_ghosts == num_ghosts) {
                 std::cerr << "Rank " << rank << ": CRITICAL WARNING - ALL " << num_ghosts << " GHOST COORDINATES ARE ZERO!" << std::endl;
            } else if (zero_ghosts > 0) {
                 // std::cout << "Rank " << rank << ": " << zero_ghosts << "/" << num_ghosts << " ghosts are at origin." << std::endl;
            }
        }
    }
    
    // Read and distribute STL from file
    std::string stl_filename = "sphere.stl";
    
    // Check if file exists on rank 0
    if (rank == 0)
    {
        std::ifstream f(stl_filename);
        if (!f.good())
        {
            std::cerr << "Error: " << stl_filename << " not found. Run generate_sphere.py first.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    cutfemx::mesh::DistributeSTLOptions stl_opt;
    stl_opt.aabb_padding = 0.05;
    
    auto sphere_soup = cutfemx::mesh::distribute_stl_facets(
        MPI_COMM_WORLD, *bg_mesh, stl_filename, stl_opt);
    
    if (rank == 0)
    {
        std::cout << "Loaded sphere surface: " << sphere_soup.nT() << " local triangles\n";
    }
    
    // Build cell-triangle map
    cutfemx::mesh::CellTriangleMapOptions map_opt;
    auto cell_tri_map = cutfemx::mesh::build_cell_triangle_map(*bg_mesh, sphere_soup, map_opt);
    
    // Helper now available in fmm.h or we just use it
    auto cut_cells = cutfemx::level_set::get_cut_cells(cell_tri_map);
    if (rank == 0)
    {
        std::cout << "Cut cells: " << cut_cells.size() << "\n";
    }
    
    // ============================================================
    // Compute unsigned distance using FMM
    // ============================================================
    if (rank == 0)
    {
        std::cout << "\nComputing unsigned distance field...\n";
    }
    
    // ============================================================
    // Create P1 function for distance field
    // ============================================================
    basix::FiniteElement e = basix::create_element<T>(
        basix::element::family::P,
        basix::cell::type::tetrahedron, 1,  // P1
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(bg_mesh, e));
    
    auto dist_func = std::make_shared<fem::Function<T>>(V);

    // ============================================================
    // Compute unsigned distance using FMM
    // ============================================================
    if (rank == 0) std::cout << "\nComputing unsigned distance field...\n";
    
    cutfemx::level_set::FMMOptions fmm_opt;
    fmm_opt.band_layers = 2;
    fmm_opt.max_iter = 1000;
    
    std::vector<std::int32_t> closest_tri;
    cutfemx::level_set::compute_unsigned_distance(
        *dist_func, sphere_soup, cell_tri_map, fmm_opt, &closest_tri);
    
    // ============================================================
    // Apply sign using local normal projection
    // ============================================================
    cutfemx::level_set::SignOptions sign_opt;
    sign_opt.mode = cutfemx::level_set::SignMode::LocalNormalBand;
    cutfemx::level_set::apply_sign(
        *dist_func, sphere_soup, cell_tri_map, closest_tri, sign_opt);
    
    if (rank == 0)
    {
        std::cout << "Signed distance field computed!\n";
    }
    

    
    // ============================================================
    // Compute error against analytical solution
    // ============================================================
    
    // Sphere parameters from generate_sphere.py
    const double center[3] = {0.5, 0.5, 0.5};
    const double radius = 0.3;
    
    const auto& geometry = bg_mesh->geometry();
    std::span<const T> x = geometry.x();
    auto geom_dofmap = geometry.dofmap();
    
    // Access function values for error check
    std::span<const T> dist_vals = dist_func->x()->array();
    
    double local_L2_error_sq = 0.0;
    double local_Linf_error = 0.0;
    int local_count = 0;
    
    // Use topology to iterate cells and map to function DOFs
    auto topology = bg_mesh->topology();
    const int tdim = topology->dim();
    auto cell_to_vertex = topology->connectivity(tdim, 0);
    auto cell_map = topology->index_map(tdim);
    const std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    
    auto dofmap = V->dofmap();
    
    // Build vertex -> DOF mapping for verification
    auto vertex_map_err = bg_mesh->topology()->index_map(0);
    const std::int32_t num_vertices = vertex_map_err->size_local() + vertex_map_err->num_ghosts();
    std::vector<std::int32_t> vert_to_dof(num_vertices, -1);
    
    for (std::int32_t c = 0; c < num_cells; ++c)
    {
        auto cell_verts = cell_to_vertex->links(c);
        auto cell_dofs = dofmap->cell_dofs(c);

        // Map vertices
        for (std::size_t i = 0; i < cell_verts.size(); ++i) {
             std::int32_t v = cell_verts[i];
             if (v < num_vertices) vert_to_dof[v] = cell_dofs[i];
        }

        // Skip ghosts for error norm accumulation (count each cell only once)
        if (c >= cell_map->size_local())
            continue;
            
        for (std::size_t i = 0; i < cell_verts.size(); ++i)
        {
            // P1 dof index corresponds to vertex index 'i' in the cell
            std::int32_t dof = cell_dofs[i];
            
            // Get value from function
            T computed_dist = dist_vals[dof];
            
            // Check for uninitialized values (infinity)
            if (computed_dist > 100.0) 
                 continue; 
            
            // Analytical distance - use geometry dofmap to get correct coordinates
            std::int32_t g = geom_dofmap(c, i);
            T vx = x[3*g];
            T vy = x[3*g+1];
            T vz = x[3*g+2];
            
            T dx = vx - center[0];
            T dy = vy - center[1];
            T dz = vz - center[2];
            T r = std::sqrt(dx*dx + dy*dy + dz*dz);
            // Signed distance: negative inside, positive outside
            T exact_dist = r - radius;
            
            T error = std::abs(computed_dist - exact_dist);
            
            local_L2_error_sq += error * error;
            local_Linf_error = std::max(local_Linf_error, static_cast<double>(error));
            local_count++;
        }
    }
    
    double global_L2_error_sq = 0.0;
    double global_Linf_error = 0.0;
    int global_count = 0;
    
    MPI_Reduce(&local_L2_error_sq, &global_L2_error_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_Linf_error, &global_Linf_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        std::cout << "\nError Analysis (vs Exact Sphere):\n";
        std::cout << "  L_inf Error: " << global_Linf_error << "\n";
        std::cout << "  L_2 Error:   " << std::sqrt(global_L2_error_sq / global_count) << "\n";
        std::cout << "\n";
    }
    
    // Find location of max error
    {
        double max_err = 0;
        int max_v = -1;
        double max_computed = 0, max_exact = 0;
        double max_x = 0, max_y = 0, max_z = 0;
        
        // Iterate over OWNED vertices by topology index
        const std::int32_t size_local = vertex_map->size_local();

        for (std::int32_t v = 0; v < size_local; ++v) {
            std::int32_t dof = vert_to_dof[v];
            if (dof < 0) continue; // Should not happen for owned P1 vertices in most cases
            
            double computed = dist_vals[dof];
            if (computed > 100.0) continue;
            
            double r = std::sqrt(std::pow(x[3*v] - center[0], 2) + 
                                std::pow(x[3*v+1] - center[1], 2) + 
                                std::pow(x[3*v+2] - center[2], 2));
            double exact = r - radius;  // Signed: negative inside, positive outside
            double err = std::abs(computed - exact);
            
            if (err > max_err) {
                max_err = err;
                max_v = v;
                max_computed = computed;
                max_exact = exact;
                max_x = x[3*v];
                max_y = x[3*v+1];
                max_z = x[3*v+2];
            }
        }
        
        // Reduce to find global max error location
        struct { double err; int rank; } local_max = {max_err, rank}, global_max;
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        if (rank == global_max.rank) {
            std::cout << "\n=== Max Error Location ===\n";
            std::cout << "  Vertex at: (" << max_x << ", " << max_y << ", " << max_z << ")\n";
            std::cout << "  Computed distance: " << max_computed << "\n";
            std::cout << "  Exact distance:    " << max_exact << "\n";
            std::cout << "  Error:             " << max_err << "\n";
            double r = std::sqrt(std::pow(max_x - center[0], 2) + 
                                std::pow(max_y - center[1], 2) + 
                                std::pow(max_z - center[2], 2));
            std::cout << "  Radius from center: " << r << " (sphere radius: " << radius << ")\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Compute distance statistics again for reporting
    T local_min = fmm_opt.inf_value;
    T local_max = 0;
    int local_inf_count = 0;
    
    const std::int32_t num_local_verts = vertex_map->size_local();
    
    for (std::int32_t v = 0; v < num_local_verts; ++v)
    {
        std::int32_t dof = vert_to_dof[v];
        if (dof < 0) continue;
        
        T val = dist_vals[dof];
        
        if (val < fmm_opt.inf_value)
        {
            local_min = std::min(local_min, val);
            local_max = std::max(local_max, val);
        }
        else
        {
            ++local_inf_count;
        }
    }
    
    T global_min, global_max;
    int global_inf_count;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_inf_count, &global_inf_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        std::cout << "Distance field statistics:\n";
        std::cout << "  Min distance: " << global_min << "\n";
        std::cout << "  Max distance: " << global_max << "\n";
        std::cout << "  Vertices with inf: " << global_inf_count << "\n";
        
        // For a sphere centered at (0.5, 0.5, 0.5) with radius 0.3:
        // - Vertices at center should have dist ≈ 0.3
        // - Vertices at corner (0,0,0) should have dist ≈ sqrt(3)*0.5 - 0.3 ≈ 0.566
        std::cout << "\nExpected for sphere(center=0.5, r=0.3):\n";
        std::cout << "  Min ≈ 0 (at surface)\n";
        std::cout << "  Max ≈ 0.566 (at corners)\n";
    }
    
    // Write to VTK file
    io::VTKFile vtk_file(MPI_COMM_WORLD, "fmm_distance.pvd", "w");
    vtk_file.write<T>({*dist_func}, 0.0);
    vtk_file.close();
    
    if (rank == 0)
    {
        std::cout << "\nOutput written to fmm_distance.pvd\n";
        std::cout << "Open in Paraview and color by distance.\n";
    }

    } // end dolfinx/VTK scope

    // NOTE: In the fenicsx-dev conda environment on macOS (MPICH ch4:ofi),
    // MPI_Finalize can fail with libfabric selecting the utun* interface.
    // We explicitly close output files above and then exit without calling
    // MPI_Finalize to avoid aborting after successful computation.
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
}
