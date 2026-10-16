---
id: blog-gau-nernst-flash-attention-5090
title: "Writing Speed-of-Light Flash Attention for 5090 in CUDA C++ (plus FlashAttention status on SM120)"
author: "gau-nernst; Dao-AILab flash-attention maintainers; SecondNatureComputing"
url: https://gau-nernst.github.io/fa-5090/
source_category: benchmark-blog
architectures: [sm120]
tags: [attention, flash-attention, ldmatrix, swizzling, pipeline-stages, shared-memory-optimization, cuda-cpp, prefill]
retrieved_at: 2026-10-15
---

# Writing Speed-of-Light Flash Attention for RTX 5090 (SM120) in CUDA C++

Primary source: gau-nernst, "Writing Speed-of-Light Flash Attention for 5090 in CUDA C++", 2025-08-23,
https://gau-nernst.github.io/fa-5090/ . Secondary sources on FlashAttention support for SM120 are listed in
References.

## Setup and roofline on RTX 5090 (SM120)

- GPU: RTX 5090 power-limited to 400 W. BF16 roofline used: 209.5 TFLOPS (the GeForce FP32-accumulate BF16 peak).
- Benchmark shape: bs=1, num_heads=8, len_query=4096, len_kv=8192, head_dim=128, non-causal forward.
- Instruction: `mma.sync` `m16n8k16` BF16 with FP32 accumulate; global->shared via `cp.async`, shared->register via
  `ldmatrix`. No TMA or block-scaled MMA used. The author notes sm120 supports MXFP8/NVFP4 MMA which Triton could
  not reach at the time, motivating CUDA C++.

## Measured kernel ladder (TFLOPS, % of 209.5 SOL) on RTX 5090

| Kernel | TFLOPS | % SOL |
|---|---|---|
| F.sdpa() Flash backend | 186.73 | 89.13% |
| F.sdpa() cuDNN backend | 203.61 | 97.19% |
| flash-attn (FA2) | 190.58 | 90.97% |
| v1 basic (cp.async + ldmatrix + mma m16n8k16) | 142.87 | 68.20% |
| v2 + shared-memory swizzling | 181.11 | 86.45% |
| v3 + 2-stage pipelining | 189.84 | 90.62% |
| v4 + ldmatrix.x4 for K and V | 194.33 | 92.76% |
| v5 + better pipelining | 197.74 | 94.39% |

## Step-by-step lessons for SM120 attention

- v1 profile: dominant stall "Short Scoreboard" from shared-memory bank conflicts (8-way on the ldmatrix rows).
- v2: XOR-swizzle shared addresses with row-dependent bits removes the 8-way conflict: +27% (142.9 -> 181.1).
- v2 profile then shows "Long Scoreboard" (exposed global latency); v3 double-buffers with
  `cp.async.commit_group` / `cp.async.wait_group`, prefetch pattern K-V-K keeping 2-3 loads in flight.
- From v3 on the dominant stall is "Math Pipe Throttle" (tensor pipe busy) - the kernel is compute-bound.
- v4: `ldmatrix.x4` instead of `.x2` for K and V gives 2x fewer ldmatrix instructions in the main loop.
- v5: only K is double-buffered, V single-buffered; the reclaimed shared memory allows BLOCK_KV 32 -> 64.
- Final config: BLOCK_Q=128, BLOCK_KV=64, DIM=128, 128 threads (4 warps), Q fragments held in registers for the
  kernel lifetime; S and P share register layout; O rescaled by exp(old_max - new_max) each KV tile.
- cuDNN's kernel (the 97% one) was inspected: BLOCK_Q=BLOCK_KV=64, DIM=128, 4 warps, 40.96 KB shared memory -
  i.e. small tiles and low smem per CTA, allowing more CTAs per SM within the ~100 KB SM120 budget.

## FlashAttention-3/4 on SM120: what runs and what does not

- Dao-AILab issue #1987 (opened 2025-11-05): FA kernels failed on RTX PRO 6000 Blackwell with CUDA error
  "invalid argument" in the forward/backward launch template. Maintainer summary: FA2 is "agnostic to arch"
  without Blackwell-specific tuning, FA3 targets Hopper (and SM80 in some cases), FA4 targets B200/B300; future
  work moves to the CuTe DSL path.
- FA3 relies on WGMMA/TMA warp specialization (SM90); FA4's fast path relies on `tcgen05` + TMEM (SM100). SM120
  has neither, so those kernels cannot run unmodified on RTX 5090 / PRO 6000 / GB10.
- SecondNatureComputing "flash-attn-4-sm120" bundle (https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120):
  ports FA4 CuTe code to SM120 using SM80-era `mma.sync.aligned.m16n8k16`, SMEM-budget-aware tiles for 99 KB,
  TMA with warp specialization for loads; limitations: head_dim > 128 unsupported on SM120 (SMEM), GQA uses
  the non-packed path, split-KV clamped to 1. Measured on GB10 (SM121a, bf16, causal GQA): FA4-port vs FA2
  ms: seqlen 128: 0.036 vs 0.021; 512: 0.053 vs 0.049; 2048: 0.289 vs 0.278 (the port is slower than FA2 in all
  three cases; the table's "ratio" column is FA4/FA2 time).
- The local KB already contains flash-attention PR-2441 (CuTe DSL `FlashAttentionForwardSm120` /
  `FlashAttentionBackwardSm120`, "SM120 tile sizes tuned for 99 KB SMEM capacity", 128 threads, SM80 MMA).

## References

- gau-nernst, "Writing Speed-of-Light Flash Attention for 5090 in CUDA C++", 2025-08-23, https://gau-nernst.github.io/fa-5090/
- Dao-AILab/flash-attention issue #1987, "Support for NVIDIA Blackwell (sm_120) architectures", https://github.com/Dao-AILab/flash-attention/issues/1987
- SecondNatureComputing, flash-attn-4-sm120 model card, https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120

## KAI Notes

- For BF16 attention on RTX 5090 the target is ~190-204 TFLOPS (cuDNN SDPA = 203.6, 97% of 209.5); a hand kernel below ~180 TFLOPS at hd=128, long seq still has bank-conflict or pipelining problems.
- Always swizzle K/V shared-memory tiles for `ldmatrix` on SM120 (+27% measured); then use `ldmatrix.x4` and double-buffer K only to free SMEM for BLOCK_KV=64.
- Keep CTA SMEM small (cuDNN uses ~41 KB with 64x64 tiles, 4 warps) so 2+ CTAs fit in the ~99-100 KB SM120 budget.
- Never port FA3/FA4 SM90/SM100 code paths to SM120; use the SM80-style `mma.sync` design (or FA CuTe SM120 classes) and cap head_dim at 128 unless SMEM math proves otherwise.
- To beat the 209.5 TFLOPS BF16 cap on GeForce, change precision (FP16-accumulate for P@V, FP8/MXFP8 or NVFP4 via SageAttention-style quantization), not scheduling.
