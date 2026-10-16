---
id: kernel-sm120-attention
title: "SM120 Attention Kernels (FlashAttention, FP8/NVFP4 Attention, Decode)"
type: kernel
architectures: [sm120]
tags: [attention, flash-attention, decode, prefill, fp8, nvfp4, block-scale, tma, mbarrier, ldmatrix, warp-specialization, ping-pong-scheduling, swizzling, pipeline-stages]
confidence: source-reported
reproducibility: snippet
kernel_types: [attention, flash-attention, decode, prefill, sparse-attention, mla]
languages: [cuda-cpp, cute-dsl, triton, ptx]
related:
  - hw-sm120-rtx-blackwell
  - technique-sm120-blockscaled-mma-sync
  - lang-ptx-sm120
  - pattern-sm120-porting-failures
  - migration-sm100-to-sm120
  - technique-swizzling
  - technique-ping-pong-scheduling
sources:
  - blog-gau-nernst-flash-attention-5090
  - blog-sm120-attention-kernels-2026h2
  - blog-sm120-sparse-mla-dsv4-2026h2
  - doc-sageattention3-micromix-fp4-rtx5090
  - doc-cudnn-sm120-sdpa-matmul-support
  - blog-gb10-dgx-spark-sm121-measurements
performance_claims:
  - gpu: "RTX 5090 (400 W)"
    dtype: bf16
    shape: "bs=1, heads=8, q=4096, kv=8192, d=128, non-causal"
    metric: TFLOPS
    value: 197.74
    utilization: "94.4% of 209.5"
    source_id: blog-gau-nernst-flash-attention-5090
    source_locator: "https://gau-nernst.github.io/fa-5090/ (final results table, v5 row)"
  - gpu: "RTX 5090 (400 W)"
    dtype: bf16
    shape: "bs=1, heads=8, q=4096, kv=8192, d=128, cuDNN SDPA"
    metric: TFLOPS
    value: 203.61
    utilization: "97.2% of 209.5"
    source_id: blog-gau-nernst-flash-attention-5090
    source_locator: "https://gau-nernst.github.io/fa-5090/ (final results table, F.sdpa() cuDNN backend row)"
  - gpu: "RTX PRO 6000 Blackwell"
    dtype: bf16
    shape: "flashinfer #3576 fa2 paged prefill, BF16 Q, page 16, 32/8 heads, s_qo=s_kv=8192 causal, D=128"
    metric: TFLOPS
    value: 280.8
    source_id: blog-sm120-attention-kernels-2026h2
    source_locator: "https://github.com/flashinfer-ai/flashinfer/pull/3576 (PR description benchmark, RTX PRO 6000 prefill D=128)"
  - gpu: "RTX 5090"
    dtype: fp4
    shape: "SageAttention3 NVFP4 forward"
    metric: TOPS
    value: 1038
    source_id: doc-sageattention3-micromix-fp4-rtx5090
    source_locator: "https://arxiv.org/abs/2505.11594 (abstract: 1038 TOPS on RTX5090)"
aliases:
  - sm120 flash attention
  - fa on rtx 5090
  - sm120 fmha
  - sm120 sdpa
---

# SM120 Attention Kernels

## What Does Not Port

FlashAttention-3 (wgmma + TMA warp specialization, SM90) and FlashAttention-4's fast path (`tcgen05` + TMEM, SM100) do not run on SM120: the hardware has neither `wgmma` nor TMEM. FA2 runs but is not tuned for SM120. The Dao-AILab CuTe-DSL tree has `FlashAttentionForwardSm120`/`BackwardSm120` classes built on SM80-style `mma.sync` with tiles sized for 99 KB shared memory; SplitKV is SM100-only there and SM120 falls back to `num_splits=1` (flash-attention #2656, #2758). cuDNN SDPA on consumer Blackwell supports fp16/bf16 only (no FP8 path; prefill d <= 256, decode/backward d <= 128).

## The SM120 Recipe

The design that works is "half of the Hopper recipe" (TensorRT-LLM #15163): TMA async loads and mbarriers survive, async MMA does not.

- One producer warp issues `cp.async.bulk.tensor` for Q/K/V tiles into a 2-stage shared-memory ring guarded by full/empty mbarriers.
- Consumer warps run S = QK^T, online softmax and O += PV on `mma.sync` (BF16 `m16n8k16`, FP8 `m16n8k32 kind::f8f6f4`, NVFP4 `m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X`).
- FA3-style ping-pong between two consumer warp groups on named barriers overlaps softmax with the other group's MMA (flashinfer #5539).
- Persistent CTAs (one per SM) with LPT ordering and runtime-SM-count-aware tile selection (sglang #32991).
- Keep CTA shared memory small: cuDNN's 97%-of-peak BF16 kernel on RTX 5090 uses BLOCK_Q = BLOCK_KV = 64, d = 128, 4 warps and ~41 KB smem, letting 2 CTAs share the 100 KB SM.

## Shared-Memory Swizzle for ldmatrix

The single largest step in the BF16 ladder below (+27%) is XOR-swizzling K/V tiles so that the 8 row addresses of each `ldmatrix` hit different banks. For 128-byte rows (64 bf16) this is the standard `Swizzle<3,4,3>` / TMA `SWIZZLE_128B` pattern; wider rows apply it per 128-byte segment:

```cuda
// Byte offset of (row, col_bytes) inside a tile with 128-byte rows: XOR the 16-byte chunk index with row % 8.
__device__ __forceinline__ uint32_t swizzle_128b(uint32_t row, uint32_t col_bytes) {
  return row * 128u + ((((col_bytes >> 4) ^ (row & 7u)) & 7u) << 4) + (col_bytes & 15u);
}
```

Use the same function for the `cp.async`/TMA write and the `ldmatrix` read addresses; with TMA, `CU_TENSOR_MAP_SWIZZLE_128B` produces this layout in hardware and needs 1024-byte-aligned destination buffers.

## Hand-Written BF16 Ladder on RTX 5090

gau-nernst's CUDA C++ kernel (cp.async + ldmatrix + `mma.m16n8k16`, BLOCK_Q=128, BLOCK_KV=64, 4 warps, Q kept in registers):

| Step | TFLOPS | % of 209.5 |
|---|---|---|
| v1 basic | 142.9 | 68% |
| + XOR swizzle of K/V smem (removes 8-way ldmatrix bank conflicts) | 181.1 | 86% |
| + 2-stage cp.async pipeline | 189.8 | 91% |
| + `ldmatrix.x4` for K and V | 194.3 | 93% |
| + double-buffer K only, BLOCK_KV 32 -> 64 | 197.7 | 94% |

Profiler progression: Short Scoreboard (bank conflicts) -> Long Scoreboard (global latency) -> Math Pipe Throttle (compute bound). On GeForce, BF16 with FP32 accumulate is capped at half rate (209.5 TFLOPS on 5090); beyond that, change precision (FP16 accumulate for PV, FP8, NVFP4), not scheduling. RTX PRO 6000 has the full 503.8 TFLOPS BF16 rate.

## Low-Precision Attention

- FP8 prefill (flashinfer #5539): persistent 256-thread CTAs, 8 warps x 16 rows, 128-key tiles via 2-stage TMA ring, ping-pong between 2 warp groups, `kind::f8f6f4` m16n8k32 for QK and PV, E4M3 probabilities with an exponent bias, K mean-subtracted per 128-key block. 1.82-1.87x (PRO 6000) / 1.93-1.96x (RTX 5090) over the fastest BF16 path.
- NVFP4 SageAttention3-style (flashinfer #5545): Q/K/V as E2M1 + UE4M3 per 16, K channel-mean removal, per-128-row Q block-mean correction computed in-kernel, two-level P quantization, QK and PV on `mxf4nvf4` 4X; ~2.3x over BF16. A warp-specialized variant uses 8 QK/softmax warps + 4 PV warps (384 threads, 1 CTA/SM) for long segments (#6060).
- TensorRT-LLM #19825 tiled FP8 context FMHA for head_dim 256 on RTX PRO 6000: geomean 3.01x over the non-tiled kernel.
- SageAttention3 on RTX 5090: 1038 TOPS NVFP4 forward.

## Decode and Paged KV

- FlashInfer XQA is the default SM12x decode backend (head_dim up to 512, FP8 and NVFP4 KV); fa2 handles GQA/short prefill; fmha_v2 HMMA kernels serve MHA prefill with seqlen >= 256.
- Low-batch decode on the 188-SM PRO 6000 under-fills the GPU: increase split-KV segments when `batch x kv_heads x splits < SMs` (vllm #58482 used 64 instead of 16 segments).
- head_dim 512 needs tiled O (several CTAs per O row) and a correspondingly scaled grid (flashinfer #3518).

## Correctness Pitfalls Seen on SM120

- TMA producers in batched/varlen prefill must add the per-request `cu_q_seqlens` offset to TMA coordinates, otherwise every request returns request 0's result (TRT-LLM #15163).
- Varlen schedulers launch padding tiles; guard with `is_valid_tile` or read `cu_seqlens` out of bounds (CUDA error 700, flash-attention #2763).
- Several paths silently ignored masks: sliding-window requests ran the maskless kernel (flashinfer #3518), FA2 prefill accepted but ignored attention sinks (vllm #52148). Test masks explicitly.
- Asymmetric named-barrier arrive/sync with a reused barrier id and idle lanes spinning on mbarrier parity caused hangs in SM120 sparse-MLA kernels.
- PDL combined with `__restrict__` (`ld.global.nc`) loads of data produced by the previous kernel can race.
