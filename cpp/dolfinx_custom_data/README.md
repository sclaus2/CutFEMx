# DOLFINx-Derived Custom-Data Form Copy

This directory is the provenance area for DOLFINx-derived form and assembler
code used by the CutFEMx runtime quadrature migration.

Only files that carry a real CutFEMx `custom_data`/loop-index delta, or that
must be specialized for `dolfinx_custom_data::fem::Form`, should live here. Unmodified
DOLFINx dependencies should be reached through upstream DOLFINx includes and
local forward declarations, not copied into this subtree.

The copied files in this directory should be taken from the local patched
DOLFINx branch that carries the `custom_data` and loop-index changes, then
adapted only as needed to build under the `dolfinx_custom_data::fem`
namespace. Keeping the copied sources here has two goals:

- make the LGPL-3.0-or-later provenance explicit;
- make future updates from newer DOLFINx versions mechanical and reviewable.

Do not mix these copied files into unrelated CutFEMx implementation folders.
CutFEMx code should depend on this layer through a small wrapper boundary.

## Source Provenance

Initial source reference:

- repository: `../dolfinx`
- branch: `sclaus2/pass_local_index_to_integration_kernel`
- upstream PR reference: `FEniCS/dolfinx#4210`
- source paths:
  - `cpp/dolfinx_custom_data/fem/Form.h`
  - `cpp/dolfinx_custom_data/fem/assemble_scalar_impl.h`
  - `cpp/dolfinx_custom_data/fem/assemble_vector_impl.h`
  - `cpp/dolfinx_custom_data/fem/assemble_matrix_impl.h`
  - `cpp/dolfinx_custom_data/fem/assembler.h`
  - `cpp/dolfinx_custom_data/fem/pack_form.h`, a small CutFEMx form adapter that
    delegates coefficient-packing primitives to upstream DOLFINx
  - `cpp/dolfinx_custom_data/fem/forward.h` as a CutFEMx-only bridge header

## Update Rule

When updating from a newer DOLFINx version:

1. Replace the copied files from the patched DOLFINx branch.
2. Re-apply only the CutFEMx namespace/build adaptations.
3. Do not copy unchanged dependency headers; update `forward.h` and upstream
   includes instead.
4. Keep the `custom_data` and loop-index delta aligned with the DOLFINx patch.
5. Add future helpers such as form factories or sparsity-pattern builders as
   narrow adapters, not by copying the full DOLFINx `utils.h`.
6. Re-run the CutFEMx runtime form and assembler tests.
