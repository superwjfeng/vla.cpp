// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MUSA (Moore Threads) builds ride the CUDA rung of src/backend.h: ggml-musa
// compiles ggml-cuda's sources with mcc under -DGGML_USE_CUDA, so a MUSA device
// answers ggml_backend_cuda_init() and every op the float archs build runs on it.
//
// The in-tree CUDA kernel families behind the ggml extension hook are not built
// there: src/kernels/ and src/cuda/ are nvcc translation units, and the hook
// itself is added to ggml-cuda by scripts/patch_ggml_cuda_ext_hook.py, which only
// the GGML_CUDA configure runs. Only pi0 and Evo-1 (BF16 activation ops), BitVLA
// (ternary kernels) and a FoldQuant GGUF (INT8/INT4 linears) call these two
// registrations. Define them as no-ops so such a checkpoint reports what is
// missing instead of failing to link.

#include "cuda/vla_cuda_ops.h"

#ifdef GGML_USE_CUDA

#include <cstdio>

namespace vla {

void cuda_register_bf16_ops() {
    std::fprintf(stderr, "vla: the BF16 CUDA kernels are not part of this MUSA build; "
                         "run pi0/Evo-1 with --act-dtype f32\n");
}

void cuda_register_foldquant_ops() {
    std::fprintf(stderr, "vla: the FoldQuant CUDA kernels are not part of this MUSA build; "
                         "VLA_FQ_DEQUANT=1 reads the sites back as float weights\n");
}

}  // namespace vla

#endif  // GGML_USE_CUDA
