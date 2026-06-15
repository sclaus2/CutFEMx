# Third Party Notices

## DOLFINx Form and Assembler Copies

CutFEMx may include modified copies of selected DOLFINx form and assembler
sources in `cpp/dolfinx_custom_data`.

The copied code is derived from DOLFINx, which is licensed under
LGPL-3.0-or-later:

- Project: https://github.com/FEniCS/dolfinx
- License: LGPL-3.0-or-later
- License texts included in this repository: `COPYING` and `COPYING.LESSER`
- Local source reference for the initial custom-data copy:
  `../dolfinx`, branch `sclaus2/pass_local_index_to_integration_kernel`
- Upstream PR reference: https://github.com/FEniCS/dolfinx/pull/4210
- DOLFINx-derived/adapted paths currently kept under `cpp/dolfinx_custom_data`:
  - `cpp/dolfinx_custom_data/fem/Form.h`
  - `cpp/dolfinx_custom_data/fem/assemble_scalar_impl.h`
  - `cpp/dolfinx_custom_data/fem/assemble_vector_impl.h`
  - `cpp/dolfinx_custom_data/fem/assemble_matrix_impl.h`
  - `cpp/dolfinx_custom_data/fem/assembler.h`
  - `cpp/dolfinx_custom_data/fem/pack_form.h`

`cpp/dolfinx_custom_data/fem/forward.h` is a CutFEMx-owned MIT bridge header
that provides forward declarations and namespace aliases for the
`dolfinx_custom_data::fem` adaptation layer. It is not treated as a copied
DOLFINx source file.

## geogram.psm.MultiPrecision

This software includes code from
[geogram.psm.MultiPrecision](https://github.com/BrunoLevy/geogram.psm.MultiPrecision),
which is licensed under the BSD-3-Clause license. The packaged license file is
included at `third_party/geogram.psm.MultiPrecision/LICENSE`. The bundled PSM
source files themselves carry the following notice:

```
Copyright (c) 2000-2022 Inria
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the ALICE Project-Team nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## CutFEMx Demo Geometry Assets

`python/demo/demo_surface.stl` is a small generated STL surface created for the
CutFEMx STL-distance demo. `python/demo/Dino.stl`, `img/dino.png`, and
`img/dino2.png` are the dino STL-distance demo assets used by the README
example.
