# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Architecture policy for A16W4 BF16 MFMA lowering."""


def select_a16w4_bf16_mfma_k(gpu_arch: str) -> int:
    """Return the legal BF16 MFMA K width for an A16W4 kernel."""

    if str(gpu_arch).lower().startswith("gfx94"):
        return 16
    return 32
