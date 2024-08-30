
namespace cutfemx::fem
{
template <dolfinx::scalar T, std::floating_point U>
void interpolate( dolfinx::fem::Function<T, U>& u1, const dolfinx::fem::Function<T, U>& u0,
    std::span<const std::int32_t> cell_map,
    std::int32_t num_cut_cells)
    {
      //fictitious domain mesh
      assert(u1.function_space());
      assert(u0.function_space());
      auto mesh = u1.function_space()->mesh();
      auto cut_mesh = u0.function_space->mesh();
      assert(cut_mesh);
      assert(mesh);

      
    }
}