# GR00T N1.7 copy kernel A/B trace (MUSA M1000)

This report compares the stock ggml F32 copy path with the copy fast paths in
`scripts/patch_ggml_musa.py` on the same GR00T N1.7 checkpoint and the same
profiler configuration. It is a kernel-level investigation, not the standard
`vla-bench` device table.

## Result

Both traces contain 3 forwards (`1` warmup + `2` timed steps) and 19,116
kernel launches. The copy launch count is also unchanged at 2,640, but the
implementation changed from 2,640 scalar copies to 2,196 vectorized copies plus
444 tiled transposes.

| Variant | Forward count | Kernel launches | Total kernel ms | Copy launches | Copy ms | Copy ms/forward | Server mean ms | Client mean ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline (`cpy_scalar`) | 3 | 19,116 | 4,451.059 | 2,640 | 1,495.911 | 498.637 | 1,737.189 | 1,754.645 |
| Optimized (vec4 + transpose) | 3 | 19,116 | 3,083.247 | 2,640 | 183.011 | 61.004 | 1,252.695 | 1,269.412 |

- Copy kernel time: `1,495.911 -> 183.011 ms`, `-87.77%`, `8.17x` faster.
- Total kernel time: `4,451.059 -> 3,083.247 ms`, `-30.73%`.
- Server-observed step mean: `1,737.189 -> 1,252.695 ms`, `-27.89%`.
- Client-observed step mean: `1,754.645 -> 1,269.412 ms`, `-27.65%`.

The earlier unprofiled 6-step benchmark measured `1,568.681 -> 1,092.567 ms`
server time (`-30.35%`). The trace run has profiler overhead and only two timed
samples, so use the ratio and kernel attribution here, not the absolute latency.

## Test setup

| | |
|---|---|
| Device | MUSA M1000, `mp_22` |
| Driver | 5.3.0-M1000 |
| Profiler | Moore Perf System 1.8.2 |
| OS | Linux 7.0.14, `aarch64` |
| Commit | `5b4e370` (F32 vec4 and tiled-transpose copy paths) |
| Model | `/home/mtuser/data/models/GR00T/N1.7/gguf/libero_object/gr00tn1d7-libero-object.gguf` |
| Model SHA-256 | `616c19a5103584f46cc404cf7e6b9722911f28c6071f2efe6f24c29cadc512b6` |
| Task | `libero_object/0` |
| Client config | `--n-steps 2 --warmup-steps 1 --n-action-steps 1` |
| Server timing | `--timing-detail none` |
| Profiler config | `msys profile -t musa --musa-graph-trace=node --duration 55` |
| Profiler env | `MUSA_INFLIGHT_SUBMISSION_LIMIT=9999`, `VLA_FIXED_NOISE_SEED=20261028` |

`--n-action-steps 1` makes every `get_action()` perform a real forward pass. The
client used the local Qwen3-VL tokenizer snapshot because the default Cosmos
tokenizer is gated. This changes text preprocessing semantics, not the tensor
shapes exercised by the copy kernels.

Only `ggml/src/ggml-cuda/cpy.cu` is reverted for the baseline. The MUSA
shared-memory clamp in `ggml-cuda.cu` remains in both builds, so the comparison
isolates the copy kernels.

## Baseline kernel ranking

`pct` is the share of total kernel time in the 3-forward trace.

| Kernel | Instances | Total ms | Avg us | pct |
|---|--:|--:|--:|--:|
| `gemmex_nt_512_256x256_bf16f32_epilogue_qy2` | 2634 | 1614.704 | 613.0 | 36.28% |
| `cpy_scalar` | 2640 | 1495.911 | 566.6 | 33.61% |
| `k_bin_bcast` | 6342 | 403.182 | 63.6 | 9.06% |
| `soft_max_f32` | 588 | 203.499 | 346.1 | 4.57% |
| `sgemm_nt_256_128x128_epilogue_qy2` | 1140 | 174.170 | 152.8 | 3.91% |
| `rope_multi` | 96 | 121.700 | 1267.7 | 2.73% |
| `concat_cont` | 312 | 92.613 | 296.8 | 2.08% |
| `convert_unary_cont_vec4` | 2634 | 91.984 | 34.9 | 2.07% |
| `unary_op_kernel` | 1063 | 78.241 | 73.6 | 1.76% |
| `norm_f32` | 1119 | 53.061 | 47.4 | 1.19% |
| `gemm_small_kernel` | 96 | 49.164 | 512.1 | 1.10% |
| `mul_mat_vec_f` | 146 | 48.950 | 335.3 | 1.10% |
| `rms_norm_f32` | 192 | 20.618 | 107.4 | 0.46% |
| `k_compute_batched_ptrs` | 96 | 2.698 | 28.1 | 0.06% |
| `scale_f32` | 12 | 0.298 | 24.8 | 0.01% |

## Optimized kernel ranking

| Kernel | Instances | Total ms | Avg us | pct |
|---|--:|--:|--:|--:|
| `gemmex_nt_512_256x256_bf16f32_epilogue_qy2` | 2634 | 1595.584 | 605.8 | 51.75% |
| `k_bin_bcast` | 6342 | 387.759 | 61.1 | 12.58% |
| `soft_max_f32` | 588 | 203.172 | 345.5 | 6.59% |
| `sgemm_nt_256_128x128_epilogue_qy2` | 1140 | 169.928 | 149.1 | 5.51% |
| `cpy_f32_vec4` | 2196 | 159.662 | 72.7 | 5.18% |
| `rope_multi` | 96 | 121.515 | 1265.8 | 3.94% |
| `concat_cont` | 312 | 92.142 | 295.3 | 2.99% |
| `convert_unary_cont_vec4` | 2634 | 84.981 | 32.3 | 2.76% |
| `unary_op_kernel` | 1063 | 73.854 | 69.5 | 2.40% |
| `norm_f32` | 1119 | 51.522 | 46.0 | 1.67% |
| `gemm_small_kernel` | 96 | 49.072 | 511.2 | 1.59% |
| `mul_mat_vec_f` | 146 | 47.971 | 328.6 | 1.56% |
| `cpy_f32_transpose` | 444 | 23.349 | 52.6 | 0.76% |
| `rms_norm_f32` | 192 | 19.680 | 102.5 | 0.64% |
| `k_compute_batched_ptrs` | 96 | 2.585 | 26.9 | 0.08% |

## Copy delta

| Kernel | Baseline instances | Baseline ms | Optimized instances | Optimized ms | Delta ms |
|---|--:|--:|--:|--:|--:|
| `cpy_scalar` | 2640 | 1495.911 | 0 | 0.000 | -1495.911 |
| `cpy_f32_vec4` | 0 | 0.000 | 2196 | 159.662 | +159.662 |
| `cpy_f32_transpose` | 0 | 0.000 | 444 | 23.349 | +23.349 |
| **Total copy** | **2640** | **1495.911** | **2640** | **183.011** | **-1312.900** |

`gemmex` is now the largest remaining kernel family at 1,595.584 ms across the
three forwards. `musa_memcpy_async` still reports pageable host-to-device copies;
that is a separate issue and was not changed by this work.

## Artifacts

All traces, exports and CSV files are under:

`/home/mtuser/vla-perfetto/copy-ab-20261029/`

| Artifact | Path |
|---|---|
| Baseline raw trace | `gr00t-n17-copy-baseline-20261029.msys-rep` |
| Optimized raw trace | `gr00t-n17-copy-optimized-20261029.msys-rep` |
| Baseline Perfetto JSON | `gr00t-n17-copy-baseline.perfetto.json` |
| Optimized Perfetto JSON | `gr00t-n17-copy-optimized.perfetto.json` |
| Baseline SQLite | `gr00t-n17-copy-baseline.sqlite` |
| Optimized SQLite | `gr00t-n17-copy-optimized.sqlite` |
| Baseline `msys stats` CSV | `gr00t-n17-copy-baseline-kernels_musa_gpu_kern_sum.csv` |
| Optimized `msys stats` CSV | `gr00t-n17-copy-optimized-kernels_musa_gpu_kern_sum.csv` |
| Baseline ranking CSV | `kernel-ranking-baseline.csv` |
| Optimized ranking CSV | `kernel-ranking-optimized.csv` |
| Joined ranking CSV | `kernel-ranking-diff.csv` |
| Copy summary CSV | `copy-summary.csv` |
| Baseline benchmark JSON | `gr00t-n17-copy-baseline-benchmark.json` |
| Optimized benchmark JSON | `gr00t-n17-copy-optimized-benchmark.json` |
| Build and server logs | `build-baseline.log`, `build-optimized.log`, `gr00t-n17-copy-*-server.log` |
| Trace and benchmark checksums | `artifact-sha256.txt`, `baseline-sha256.txt`, `optimized-sha256.txt` |

The baseline trace contains `cpy_scalar`; the optimized trace contains
`cpy_f32_vec4` and `cpy_f32_transpose`. Both traces have the same total kernel
count, so the comparison is not confounded by a graph change.

## Reproduction outline

```bash
ART=/home/mtuser/vla-perfetto/copy-ab-20261029
CKPT=/home/mtuser/data/models/GR00T/N1.7/gguf/libero_object/gr00tn1d7-libero-object.gguf
TOK=/home/mtuser/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
STATS=/home/mtuser/data/models/GR00T/N1.7/gguf/libero_object/dataset_statistics.json
export LD_LIBRARY_PATH=/home/mtuser/miniconda3/envs/vlasrv/lib:${LD_LIBRARY_PATH:-}
export PATH=/home/mtuser/opt/protoc-3.20/bin:$PATH
export MUJOCO_GL=egl

# Baseline: use the stock cpy.cu from llama.cpp-b11223, then rebuild.
cp /home/mtuser/workspace/fork/vla-deps/llama.cpp-b11223/ggml/src/ggml-cuda/cpy.cu \
   build-musa/_deps/llama-src/ggml/src/ggml-cuda/cpy.cu
cmake --build build-musa --target vla-server -j"$(nproc)"

msys profile -t musa --musa-graph-trace=node --duration 55 \
  -e MUSA_INFLIGHT_SUBMISSION_LIMIT=9999,VLA_FIXED_NOISE_SEED=20261028 \
  -o "$ART/gr00t-n17-copy-baseline-20261029.msys-rep" \
  ./build-musa/vla-server --bind 'tcp://*:5555' "$CKPT"

# Run this while the baseline server is ready.
eval/sim/libero/libero_uv/.venv/bin/python eval/client/benchmark.py \
  --backend vla-cpp --addr tcp://localhost:5555 --arch gr00t_n1_7 \
  --task libero_object --task-id 0 --n-steps 2 --warmup-steps 1 \
  --n-action-steps 1 --tokenizer "$TOK" --stats-json "$STATS" \
  --output "$ART/gr00t-n17-copy-baseline-benchmark.json"

# Optimized: reapply the patch and rebuild, then repeat the same profiler/client run.
python3 scripts/patch_ggml_musa.py build-musa/_deps/llama-src
cmake --build build-musa --target vla-server -j"$(nproc)"
```

Export and ranking commands:

```bash
msys stats -r musa_gpu_kern_sum -f csv \
  -o "$ART/gr00t-n17-copy-optimized-kernels" \
  "$ART/gr00t-n17-copy-optimized-20261029.msys-rep"
msys export -t json -o "$ART/gr00t-n17-copy-optimized.perfetto.json" \
  "$ART/gr00t-n17-copy-optimized-20261029.msys-rep"
msys export -t sqlite -o "$ART/gr00t-n17-copy-optimized.sqlite" \
  "$ART/gr00t-n17-copy-optimized-20261029.msys-rep"
```

## Caveats

- The trace run uses two timed steps; the existing 6-step benchmark remains the
  better source for a latency claim.
- `msys profile --duration 55` covers model load plus the three forwards. The
  `.msys-rep` is finalized by the profiler timeout, not by killing the server.
- The local Qwen3-VL tokenizer override is for performance measurement only.
- Pageable `musaMemcpyAsync` operations remain visible in `msys analyze`; they
  are not part of the device-side `cpy_scalar` bottleneck addressed here.
