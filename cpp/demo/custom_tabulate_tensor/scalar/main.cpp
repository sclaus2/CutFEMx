#include <iostream>
#include <vector>

#include "scalar.h"

#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/mesh/generation.h>


using T = double;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  {
    auto celltype = dolfinx::mesh::CellType::triangle;
    T alpha = 1.0;

    int N = 11;

    auto part = dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<dolfinx::mesh::Mesh<T>>(
        dolfinx::mesh::create_rectangle(MPI_COMM_WORLD, {{{-1.0, -1.0}, {1.0, 1.0}}},
                               {N, N}, celltype, part));

    // Assemble over all cells by calling the generated UFCx cell kernel directly.
    const std::size_t tdim = mesh->geometry().dim();
    auto cell_imap = mesh->topology()->index_map(tdim);
    assert(cell_imap);
    std::int32_t num_cells = cell_imap->size_local() + cell_imap->num_ghosts();

    auto x = mesh->geometry().x();
    auto x_dofmap = mesh->geometry().dofmap();

    std::vector<T> coordinate_dofs(3 * x_dofmap.extent(1));

    const ufcx_form& ufcx_L = *form_scalar_L;
    const int* integral_offsets = ufcx_L.form_integral_offsets;

    T sum(0);

    ufcx_integral* integral = ufcx_L.form_integrals[integral_offsets[cell]];
    auto kernel = integral->tabulate_tensor_float64;

    for (std::int32_t c = 0; c < num_cells; ++c)
    {
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      T cell_value = 0;
      kernel(&cell_value, nullptr, &alpha, coordinate_dofs.data(), nullptr, nullptr, nullptr);
      sum += cell_value;
    }

    std::cout << "sum=" << sum << std::endl;
  }
  MPI_Finalize();
}
