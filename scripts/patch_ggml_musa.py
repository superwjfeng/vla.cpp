#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply the MUSA compatibility and copy-throughput patches to ggml-cuda.

What it changes
---------------
1. Clamp the MUSA shared-memory opt-in to what the driver actually accepts in
   `ggml/src/ggml-cuda/ggml-cuda.cu`.
2. Add F32 vectorized and tiled-transpose copy paths in
   `ggml/src/ggml-cuda/cpy.cu`. The stock F32 copy launches 64 threads, moves
   one 4-byte element per thread, and performs several 64-bit divisions per
   element. On M1000 this measured about 2 GB/s and made `cpy_scalar` the
   largest single kernel family in pi0.5 and GR00T N1.7 traces.

One line in the MUSA branch of the device probe: the figure stored in
`info.devices[id].smpbo` is the device's `sharedMemPerBlockOptin`, capped at
64 KiB.

Why
---
An M1000 (MUSA 5.1.0, mp_22) reports sharedMemPerBlockOptin = 73728 and then
answers musaErrorInvalidValue when that exact number is handed back to
musaFuncSetAttribute(kernel, musaFuncAttributeMaxDynamicSharedMemorySize, ...):
the driver's real ceiling is below what it advertises. ggml passes smpbo
verbatim through CUDA_SET_SHARED_MEMORY_LIMIT, whose CUDA_CHECK aborts the
process, and the first caller is soft_max_f32 -- so the first attention op of
any model kills the run with

    MUSA error: invalid argument
      in function operator() at ggml/src/ggml-cuda/softmax.cu:290

The value is only ever used as an upper bound: soft_max_f32_cuda compares its
per-shape dynamic shared-memory request against it and softmax over a 4k context
asks for (4096 + 32) * 4 B = 16.5 KiB. No ggml kernel requests more than 64 KiB
of dynamic shared memory on a device whose entire budget is 72 KiB, so the clamp
costs nothing and the abort goes away.

With this applied a stock ggml-cuda behaves identically on every non-MUSA
device: the branch it edits is not compiled there.

The copy fast paths are exact byte copies, not conversions. They preserve the
logical ggml copy order and require aligned, non-overlapping layouts; anything
outside their preconditions falls through to the original kernels.

Usage: scripts/patch_ggml_musa.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: MUSA shared memory opt-in"
CPY_MARKER = "vla.cpp: F32 vectorized copy fast paths"

OLD = """#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
"""

NEW = """#elif defined(GGML_USE_MUSA)
        // vla.cpp: MUSA shared memory opt-in. The driver advertises 72 KiB on an
        // M1000 and rejects exactly that value, so cap it at the 64 KiB it takes.
        // See scripts/patch_ggml_musa.py for the measurement and the failure it
        // otherwise causes in soft_max_f32.
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin > 64*1024 ? 64*1024 : prop.sharedMemPerBlockOptin;
"""


CPY_CONST_OLD = """const int CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows
"""

CPY_CONST_NEW = CPY_CONST_OLD + """const int CUDA_CPY_VEC_BLOCK_SIZE = 256;
"""


CPY_KERNELS = r"""// vla.cpp: F32 vectorized copy fast paths. Copy each 16-byte vector when both
// tensors have a contiguous F32 dim-0 and 16-byte-aligned row strides.
static __global__ void cpy_f32_vec4(const char * cx, char * cdst, const int64_t ne,
                                    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                    const int64_t nb12, const int64_t nb13) {
    ggml_cuda_pdl_lc();
    const int64_t i = ((int64_t) blockDim.x*blockIdx.x + threadIdx.x)*4;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02)/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02 - i02*ne01*ne00)/ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;

    int64_t i13 = i03;
    int64_t i12 = i02;
    int64_t i11 = i01;
    int64_t i10 = i00;
    if (ne00 != ne10 || ne01 != ne11 || ne02 != ne12) {
        i13 = i/(ne10 * ne11 * ne12);
        i12 = (i - i13*ne10*ne11*ne12)/(ne10*ne11);
        i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11)/ne10;
        i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    }

    ggml_cuda_pdl_sync();
    const int64_t x_offset   = i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;
    *reinterpret_cast<uint4 *>(cdst + dst_offset) = *reinterpret_cast<const uint4 *>(cx + x_offset);
}

// A true layout transpose: source dim 1 and destination dim 0 are contiguous.
// Stage a 32x32 float tile through shared memory so both sides are coalesced.
static __global__ void cpy_f32_transpose(const char * cx, char * cdst,
                                         const int64_t ne00, const int64_t ne01, const int64_t ne02,
                                         const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
                                         const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13) {
    __shared__ float tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int64_t i0 = (int64_t) blockIdx.x*CUDA_CPY_TILE_DIM_2D;
    const int64_t i1 = (int64_t) blockIdx.y*CUDA_CPY_TILE_DIM_2D;
    const int64_t iz = blockIdx.z;
    const int64_t i2 = iz % ne02;
    const int64_t i3 = iz / ne02;
    const int64_t src_outer = i2*nb02 + i3*nb03;
    const int64_t dst_outer = i2*nb12 + i3*nb13;
    const float * src = reinterpret_cast<const float *>(cx + src_outer);
    float * dst = reinterpret_cast<float *>(cdst + dst_outer);

#pragma unroll
    for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
        const int l0 = threadIdx.y + j;
        const int l1 = threadIdx.x;
        if (i0 + l0 < ne00 && i1 + l1 < ne01) {
            tile[l0][l1] = *(const float *) ((const char *) src + (i0 + l0)*nb00 + (i1 + l1)*nb01);
        }
    }

    __syncthreads();

#pragma unroll
    for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
        const int l0 = threadIdx.x;
        const int l1 = threadIdx.y + j;
        if (i0 + l0 < ne00 && i1 + l1 < ne01) {
            *(float *) ((char *) dst + (i0 + l0)*nb10 + (i1 + l1)*nb11) = tile[l0][l1];
        }
    }
}
"""


CPY_SUPPORT = r"""static bool ggml_cpy_f32_vec4_supported(const ggml_tensor * src0, const ggml_tensor * src1) {
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] % 4 != 0 || src1->ne[0] % 4 != 0) {
        return false;
    }
    if (src0->nb[0] != (int64_t) sizeof(float) || src1->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }
    if (((uintptr_t) src0->data & 15) != 0 || ((uintptr_t) src1->data & 15) != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0->nb[i] % 16 != 0 || src1->nb[i] % 16 != 0) {
            return false;
        }
    }
    return true;
}

static bool ggml_cpy_f32_transpose_supported(const ggml_tensor * src0, const ggml_tensor * src1) {
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || !ggml_are_same_shape(src0, src1)) {
        return false;
    }
    if (src0->nb[1] != (int64_t) sizeof(float) || src1->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }
    if (src0->nb[0] == (int64_t) sizeof(float) || src0->ne[0] < 2 || src0->ne[1] < 2) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src0->nb[i] % sizeof(float) != 0 || src1->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    const int64_t grid_x = (src0->ne[0] + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    const int64_t grid_y = (src0->ne[1] + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    const int64_t grid_z = src0->ne[2] * src0->ne[3];
    return grid_x <= UINT_MAX && grid_y <= UINT_MAX && grid_z <= USHRT_MAX;
}

static void ggml_cpy_f32_vec4_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
    const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % 4 == 0);
    const int64_t num_blocks = (ne/4 + CUDA_CPY_VEC_BLOCK_SIZE - 1) / CUDA_CPY_VEC_BLOCK_SIZE;
    GGML_ASSERT(num_blocks <= INT_MAX);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
        dim3(num_blocks), CUDA_CPY_VEC_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(cpy_f32_vec4, launch_params,
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_transpose_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
    const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_UNUSED_VARS(ne10, ne11, ne12);
    const int64_t grid_x = (ne00 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    const int64_t grid_y = (ne01 + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    const int64_t grid_z = ne / (ne00*ne01);
    const dim3 grid((unsigned) grid_x, (unsigned) grid_y, (unsigned) grid_z);
    const dim3 block(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    ggml_cuda_kernel_launch(cpy_f32_transpose, launch_params,
        cx, cdst, ne00, ne01, ne02, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13);
}
"""


CPY_DISPATCH_OLD = """    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (can_be_transposed) {
"""

CPY_DISPATCH_NEW = """    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        // vla.cpp: F32 vectorized copy fast paths
        if (ggml_cpy_f32_vec4_supported(src0, src1)) {
            ggml_cpy_f32_vec4_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else if (ggml_cpy_f32_transpose_supported(src0, src1)) {
            ggml_cpy_f32_transpose_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else if (can_be_transposed) {
"""


def replace_once(src: str, old: str, new: str, path: pathlib.Path, label: str) -> str:
    if src.count(old) != 1:
        raise RuntimeError("%s anchor found %d times in %s (%s)" % (label, src.count(old), path, label))
    return src.replace(old, new, 1)


def patch_cpy(root: pathlib.Path) -> int:
    path = root / "ggml" / "src" / "ggml-cuda" / "cpy.cu"
    if not path.is_file():
        print("patch_ggml_musa: %s not found" % path, file=sys.stderr)
        return 1
    src = path.read_text(encoding="utf-8")
    if CPY_MARKER in src:
        print("patch_ggml_musa: copy fast paths already applied")
        return 0
    try:
        src = replace_once(src, CPY_CONST_OLD, CPY_CONST_NEW, path, "copy constant")
        src = replace_once(src, "template <cpy_kernel_t cpy_1>\n",
                           CPY_KERNELS + "\ntemplate <cpy_kernel_t cpy_1>\n", path, "copy kernels")
        src = replace_once(src, "static void ggml_cpy_f32_q8_0_cuda(\n",
                           CPY_SUPPORT + "\nstatic void ggml_cpy_f32_q8_0_cuda(\n", path, "copy dispatch helpers")
        src = replace_once(src, CPY_DISPATCH_OLD, CPY_DISPATCH_NEW, path, "copy dispatch")
    except RuntimeError as exc:
        print("patch_ggml_musa: %s" % exc, file=sys.stderr)
        return 1
    path.write_text(src, encoding="utf-8")
    print("patch_ggml_musa: patched %s" % path)
    return 0


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    path = root / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    if not path.is_file():
        print("patch_ggml_musa: %s not found" % path, file=sys.stderr)
        return 1
    if patch_cpy(root) != 0:
        return 1
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print("patch_ggml_musa: already applied")
        return 0
    if src.count(OLD) != 1:
        print("patch_ggml_musa: anchor found %d times in %s" % (src.count(OLD), path), file=sys.stderr)
        return 1
    path.write_text(src.replace(OLD, NEW), encoding="utf-8")
    print("patch_ggml_musa: patched %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
