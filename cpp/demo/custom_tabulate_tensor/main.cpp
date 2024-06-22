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

T compute_detJ(std::span<const T> coordinate_dofs)
{
  auto p0 = coordinate_dofs.subspan(0, 3);
  auto p1 = coordinate_dofs.subspan(3, 3);
  auto p2 = coordinate_dofs.subspan(6, 3);

  T _detJ = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]);

  return _detJ;
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  auto celltype = dolfinx::mesh::CellType::triangle;
  int degree = 1;
  double alpha = 1.0;

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
  const int num_points = wts.size();

  //assemble over all cells using standarad quadrature with the runtime quadrature kernel
  const std::size_t tdim = mesh->geometry().dim();
  auto cell_imap = mesh->topology()->index_map(tdim);
  assert(cell_imap);
  std::int32_t num_cells = cell_imap->size_local() + cell_imap->num_ghosts();

  //get coordinates of all nodes in mesh
  auto x = mesh->geometry().x();
  //get mapping of cell index to coordinates
  auto x_dofmap = mesh->geometry().dofmap();

  // Container for coordinate of dofs
  std::vector<T> coordinate_dofs(3 * x_dofmap.extent(1));
  // container to pass weights = wts*detJ
  std::vector<T> weights(wts.size());

  const ufcx_form& ufcx_L = *form_scalar_L;
  const int* integral_offsets = ufcx_L.form_integral_offsets;

  // ----------------------------------------------------------------------------------
  // Custom Integral
  // ----------------------------------------------------------------------------------
  ufcx_integral* custom_integral = ufcx_L.form_integrals[integral_offsets[custom]];
  auto custom_kernel = custom_integral->custom_tabulate_tensor_float64;

  T value(0);
  T custom_sum(0);
  // Iterate over all cells
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    //get cell node coordinates
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    //obtain the determinante of the Jacobian for the cell as this is
    // multiplied by hand for custom integrals
    T _detJ = compute_detJ(coordinate_dofs);

    for(std::int32_t i = 0; i < num_points; ++i)
        weights[i] = wts[i]*std::fabs(_detJ);

    custom_kernel(&value, {}, &alpha, coordinate_dofs.data(), nullptr,
        nullptr, &num_points,  pts.data(), weights.data());

    custom_sum += value;
  }

  std::cout << "custom_sum=" << custom_sum << std::endl;

  // ----------------------------------------------------------------------------------
  // Standard Integral
  // ----------------------------------------------------------------------------------
  value = 0.0;
  T sum(0);

  //obtain standard integral
  ufcx_integral* integral = ufcx_L.form_integrals[integral_offsets[cell]];
  auto kernel = integral->tabulate_tensor_float64;

    // Iterate over all cells
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    //get cell node coordinates
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    kernel(&value, {}, &alpha, coordinate_dofs.data(), nullptr, nullptr);

    sum += value;
  }

  std::cout << "sum=" << sum << std::endl;

}