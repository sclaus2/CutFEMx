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

    // Definitions for loop
    cutfemx::level_set::FMMOptions fmm_opt;
    fmm_opt.band_layers = 2;
    fmm_opt.max_iter = 1000;
    
    std::vector<std::int32_t> closest_tri;
    
    auto topology = bg_mesh->topology();
    const int tdim = topology->dim();
    auto cell_to_vertex = topology->connectivity(tdim, 0);
    auto cell_map = topology->index_map(tdim);
    const std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();
    auto dofmap = V->dofmap();
    const auto& geometry = bg_mesh->geometry();
    std::span<const T> x = geometry.x();
    auto geom_dofmap = geometry.dofmap();
    
    // Sphere parameters
    const double center[3] = {0.5, 0.5, 0.5};
    const double radius = 0.3;

    // Verify ALL sign methods
    struct MethodSpec {
        cutfemx::level_set::SignMode mode;
        std::string name;
        std::string vtk_name;
    };
    
    std::vector<MethodSpec> methods = {
        {cutfemx::level_set::SignMode::LocalNormalBand, "Method 1 (LocalNormalBand)", "sign_method1.pvd"},
        {cutfemx::level_set::SignMode::ComponentAnchor, "Method 2 (ComponentAnchor)", "sign_method2.pvd"},
        {cutfemx::level_set::SignMode::WindingNumber,   "Method 3 (WindingNumber)",   "sign_method3.pvd"}
    };
    
    for (const auto& method : methods) {
        if (rank == 0) std::cout << "\n=== Testing " << method.name << " ===\n";
        
        // Reset distance to Unsigned
        // Re-run FMM (fastest way to ensure clean state)
        // Or store copy. Re-running is fine for demo size.
        if (rank == 0) std::cout << "Re-computing unsigned distance...\n";
        cutfemx::level_set::compute_unsigned_distance(
             *dist_func, sphere_soup, cell_tri_map, fmm_opt, &closest_tri);
        
        // Apply Sign
        cutfemx::level_set::SignOptions sign_opt;
        sign_opt.mode = method.mode;
        // Method 2 options
        sign_opt.use_boundary_as_outside_anchor = true;
        
        // Method 3 options
        sign_opt.winding_threshold = 0.5;
        
        cutfemx::level_set::apply_sign(
            *dist_func, sphere_soup, cell_tri_map, closest_tri, sign_opt);

        // Error Analysis
        double method_L2_error_sq = 0.0;
        double method_Linf_error = 0.0;
        int count = 0;
        
        std::span<const T> vals = dist_func->x()->array();

        // Error loop (simplified from above)
        for (std::int32_t c = 0; c < num_cells; ++c) {
            // Reuse local pointers/maps
             auto cell_verts = cell_to_vertex->links(c);
             auto cell_dofs = dofmap->cell_dofs(c);
             
             // Skip ghosts
             if (c >= cell_map->size_local()) continue;
             
             for (std::size_t i = 0; i < cell_verts.size(); ++i) {
                 std::int32_t dof = cell_dofs[i];
                 T val = vals[dof];
                 if (val > 100.0) continue;
                 
                 std::int32_t g = geom_dofmap(c, i);
                 T vx = x[3*g], vy = x[3*g+1], vz = x[3*g+2];
                 T r = std::sqrt(std::pow(vx-center[0],2) + std::pow(vy-center[1],2) + std::pow(vz-center[2],2));
                 T exact = r - radius;
                 
                 T err = std::abs(val - exact);
                 method_L2_error_sq += err * err;
                 method_Linf_error = std::max(method_Linf_error, (double)err);
                 count++;
             }
        }
        
        double gl_L2_sq = 0.0, gl_Linf = 0.0;
        int gl_count = 0;
        MPI_Reduce(&method_L2_error_sq, &gl_L2_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&method_Linf_error, &gl_Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&count, &gl_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "  L_inf Error: " << gl_Linf << "\n";
            std::cout << "  L_2 Error:   " << std::sqrt(gl_L2_sq / gl_count) << "\n";
        }
        
        // Write VTK
        io::VTKFile vtk_file(MPI_COMM_WORLD, method.vtk_name.c_str(), "w");
        vtk_file.write<T>({*dist_func}, 0.0);
        vtk_file.close();
    }
    
    if (rank == 0) std::cout << "\nAll methods verified.\n";


    } // end dolfinx/VTK scope

    // NOTE: In the fenicsx-dev conda environment on macOS (MPICH ch4:ofi),
    // MPI_Finalize can fail with libfabric selecting the utun* interface.
    // We explicitly close output files above and then exit without calling
    // MPI_Finalize to avoid aborting after successful computation.
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(0);
}
