# Kimi K3 gfx942 A16W4 LDS DMA Design

## Goal

Make the current AITER Kimi K3 SiTUv2 A16W4 FlyDSL kernels compile and run
numerically on MI300X (`gfx942`) without changing the validated gfx950/gfx1250
paths.

## Verified root cause

The current mixed-MoE A16W4 kernels already dequantize MXFP4 weights to BF16,
use legacy BF16 MFMA, and implement SiTUv2. The failing K3 stage-1 compile is
therefore narrower than the older AITER #3926 port.

Both A16W4 stages hard-code a 16-byte
`llvm.amdgcn.raw.ptr.buffer.load.lds`. ROCm LLVM permits 16-byte load-to-LDS on
gfx950, but gfx94 only supports 1-, 2-, and 4-byte widths. The exact K3 probe
therefore aborts during LLVM instruction selection before executing a kernel.

## Considered approaches

1. **Port all of AITER #3926.** This contains useful gfx942 prior art, but it
   targets the older `moe_gemm_2stage.py`, predates the current K3 SiTUv2
   kernel, and would duplicate functionality already present on main.
2. **Route to CK-Tile.** Rejected because the current CK activation path does
   not implement K3 SiTUv2 semantics.
3. **Use a gfx-aware load-to-LDS width in the current mixed kernel.**
   Recommended: split each gfx94 activation copy into 4-byte operations while
   preserving the current 16-byte operation elsewhere.

## Design

Add one dependency-free policy helper that selects a legal DMA width from the
GPU architecture and per-thread copy size:

- `gfx94*`: 4 bytes;
- other architectures: retain 16 bytes when divisible, otherwise use 4 bytes;
- reject copy sizes that cannot be represented by 4-byte operations.

Use the selected width consistently for:

- activation-tile coordinate generation;
- number of loads per thread;
- global byte offsets;
- LDS destination increments;
- the intrinsic byte-size operand.

Apply the same policy to both A16W4 stage 1 and stage 2. Do not change weight
dequantization, MFMA selection, SiTUv2 math, layouts, tuning, A8W4, or generic
mixed-MoE paths.

## Validation

1. A dependency-free CPU unit test locks the architecture/width policy.
2. Static checks confirm both A16W4 stages consume the selected width rather
   than a hard-coded 16.
3. The existing exact K3 probe must compile and pass numerically on gfx942:
   `M=1, D=3584, I=384, E=896, topk=16`, A16W4 `per_1x32`, separated SiTUv2
   with `beta=4` and `linear_beta=25`.
4. Only after the exact probe passes may InferenceX stage the full model and
   run the two-node vLLM and AgentX canaries.

## Completion criteria

The fix is complete only when the exact gfx942 numerical probe passes. A local
unit-test-only result is insufficient.
