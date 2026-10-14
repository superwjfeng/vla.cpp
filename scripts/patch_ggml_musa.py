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

"""Clamp the MUSA shared-memory opt-in to what the driver actually accepts.

What it changes (ggml/src/ggml-cuda/ggml-cuda.cu only)
------------------------------------------------------
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

Usage: scripts/patch_ggml_musa.py [<llama-src-dir>]
"""

import pathlib
import sys

MARKER = "vla.cpp: MUSA shared memory opt-in"

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


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    path = root / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    if not path.is_file():
        print("patch_ggml_musa: %s not found" % path, file=sys.stderr)
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
