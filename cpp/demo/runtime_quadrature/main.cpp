#include <vector>
#include <span>
#include <iostream>
#include <string>
#include <math.h>

#include <basix/finite-element.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/fem/utils.h>

#include <dolfinx/io/XDMFFile.h>

#include <cutfemx/level_set/locate_entities.h>
#include <cutfemx/level_set/cut_entities.h>

#include <cutfemx/quadrature/quadrature.h>
#include <cutfemx/quadrature/generation.h>
#include <cutfemx/quadrature/physical_points.h>



using T = double;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  auto celltype = dolfinx::mesh::CellType::triangle;
  int degree = 1;

  int N = 21;

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

  cutfemx::quadrature::QuadratureRules<T> runtime_rules;
  int order = 2;
  cutfemx::quadrature::runtime_quadrature<T>(level_set, "phi<0", order, runtime_rules);
  const auto& points_phys = cutfemx::quadrature::physical_points(runtime_rules, *mesh);

}