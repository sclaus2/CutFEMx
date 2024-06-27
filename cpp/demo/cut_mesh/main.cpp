#include <vector>
#include <span>
#include <iostream>
#include <string>
#include <math.h>

#include <basix/finite-element.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/fem/utils.h>

#include <cutcells/cut_mesh.h>

#include <cutfemx/level_set/locate_entities.h>
#include <cutfemx/level_set/cut_entities.h>

using T = double;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  auto celltype = dolfinx::mesh::CellType::triangle;
  int degree = 1;

  long unsigned int N = 11;

  auto part = dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(
        dolfinx::mesh::create_rectangle(MPI_COMM_WORLD, {{{-1.0, -1.0}, {1.0, 1.0}}},
                               {N, N}, celltype, part));

  // Create a Basix continuous Lagrange element of degree 1
  basix::FiniteElement e = basix::create_element<T>(
      basix::element::family::P,
      dolfinx::mesh::cell_type_to_basix_type(celltype), degree,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  // Create a scalar function space
  auto V = std::make_shared<fem::FunctionSpace<T>>(
      dolfinx::fem::create_functionspace(mesh, e));

  // Create level set function
  auto level_set = std::make_shared<fem::Function<T>>(V);

  // Interpolate sqrt(x^2+y^2)-r in the scalar Lagrange finite element
  // space
  level_set->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f(x.extent(1));
        T r = 0.5;
        for (std::size_t p = 0; p < x.extent(1); ++p)
          f[p] = std::sqrt(x(0,p)*x(0,p)+x(1,p)*x(1,p))-r;
        return {f, {f.size()}};
      });

  //cutting of intersected cells
  int tdim = mesh->topology()->dim();
  auto intersected_cells = cutfemx::level_set::locate_entities<T>( level_set,tdim,"phi=0");
  cutcells::mesh::CutCells cut_cells = cutfemx::level_set::cut_entities<T>(level_set,intersected_cells, tdim,"phi<0");

  cutcells::mesh::str(cut_cells);
}