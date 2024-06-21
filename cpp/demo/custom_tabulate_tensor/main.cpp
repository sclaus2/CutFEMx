#include <vector>
#include <span>
#include <iostream>
#include <string>
#include <math.h>

#include "scalar.h"

#include <basix/finite-element.h>
#include <basix/quadrature.h>

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/mesh/generation.h>


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

  const auto [pts, wts] = basix::quadrature::make_quadrature<T>(basix::quadrature::type::Default, dolfinx::mesh::cell_type_to_basix_type(celltype), basix::polyset::type::standard, degree * 2);
  const int num_points = pts.size();

  //assemble over all cells using standarad quadrature with the runtime quadrature kernel
  const std::size_t tdim = mesh->geometry().dim();
  auto cell_imap = mesh->topology()->index_map(tdim);
  assert(cell_imap);
  std::int32_t num_cells = cell_imap->size_local() + cell_imap->num_ghosts();

  const ufcx_form& ufcx_L = *form_scalar_L;
  // const int* integral_offsets = ufcx_L.form_integral_offsets;
  //just one integral in form for test purposes
  ufcx_integral* integral = ufcx_L.form_integrals[0];
  auto kernel = integral->custom_tabulate_tensor_float64;
  double alpha = 1.0;

  T value(0);
  T sum(0);
  // Iterate over all cells
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    kernel(&value, {}, &alpha, {}, nullptr,
        nullptr, &num_points,  pts.data(), wts.data());
    sum += value;
  }

  std::cout << "sum=" << sum << std::endl;

}