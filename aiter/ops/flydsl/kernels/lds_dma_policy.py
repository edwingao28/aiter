# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Architecture policy for raw buffer-load-to-LDS copy widths."""

from __future__ import annotations


def select_raw_ptr_buffer_load_lds_bytes(gpu_arch: str, bytes_per_thread: int) -> int:
    """Choose a legal load-to-LDS width that evenly covers one thread's copy.

    gfx94 supports only 1-, 2-, and 4-byte
    ``llvm.amdgcn.raw.ptr.buffer.load.lds`` operations. Existing A16W4 kernels
    use dwordx4 on architectures that support it, so retain 16 bytes elsewhere
    when the per-thread copy can be divided exactly.
    """

    if bytes_per_thread <= 0 or bytes_per_thread % 4 != 0:
        raise ValueError(
            "bytes_per_thread must be positive and divisible by 4, "
            f"got {bytes_per_thread}"
        )

    arch = str(gpu_arch or "").strip().lower().split(":", 1)[0]
    if arch.startswith("gfx94"):
        return 4
    if bytes_per_thread % 16 == 0:
        return 16
    return 4
