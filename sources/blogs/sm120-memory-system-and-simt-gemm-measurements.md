---
id: blog-sm120-memory-system-and-simt-gemm-measurements
title: "SM120 memory-system and SIMT/HGEMM measurements: Chips and Cheese (RTX PRO 6000), CloudRift (RTX 5090 cuBLAS SGEMM bug), cuda-ops-5090d"
author: "Chester Lam (Chips and Cheese); Dmitry Trifonov (CloudRift); jackbrowntkd (GitHub)"
url: https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu
source_category: benchmark-blog
architectures: [sm120]
tags: [gemm, shared-memory-optimization, tma, mbarrier, warp-specialization, pipeline-stages, vectorized-loads, register-budgeting]
retrieved_at: 2026-10-15
---

# SM120 memory-system and SIMT/HGEMM measurements (RTX PRO 6000, RTX 5090)

Sources fetched: Chips and Cheese "Blackwell: Nvidia's Massive GPU" (Chester Lam, 2025-06-29); CloudRift
"A 60% cuBLAS Performance Bug in FP32 SGEMM on the RTX 5090" (2026-04-06) and "Modern GPU Matmul Optimization"
(2026-05-29), both by Dmitry Trifonov; GitHub jackbrowntkd/cuda-ops-5090d (RTX 5090 D, Chinese README).

## Chips and Cheese: RTX PRO 6000 Blackwell cache and memory (OpenCL/Vulkan microbenchmarks)

| Metric (RTX PRO 6000, GB202, 188 SMs) | Measured / stated |
|---|---|
| L1/shared per SM | 128 KB SM-wide block partitioned between L1 and shared; L1 latency unaffected by the carveout split |
| Total L1/shared | 24 MB across the card |
| L1/shared bandwidth | 128 B/cycle per SM; "more than 60 TB/s" aggregate over 188 SMs |
| L2 bandwidth | ~8.7 TB/s; "each bank likely delivers 64 bytes per cycle (of which the test was able to achieve 48B/cycle)" |
| L2 latency | "just over 130 ns" (Ada: 107 ns); RTX 5070 Vulkan: 122 ns |
| VRAM latency | 329 ns, ~200 ns over L2 hit latency |
| Occupancy | 12 waves (warps) per SM sub-partition max; register file 64 KB per partition |
| INT32 | 16 INT32 multiplies per cycle per partition; 16 INT32 atomic ALUs per SM for local-memory atomics |

The article attributes part of the L2 latency regression to GB202's size, noting 5070 results suggest more to it.
Ratio check: 8.7 TB/s L2 vs 1.79 TB/s DRAM = ~4.9x; L2 is ~2.5x lower latency than VRAM.

## Achieved DRAM bandwidth on RTX 5090 (two independent measurements)

| Source | GPU | memcpy / D2D bandwidth | % of 1792 GB/s |
|---|---|---|---|
| gau-nernst gn-kernels | RTX 5090 @400W / @600W | 1509.05 / 1509.43 GB/s | 84% |
| cuda-ops-5090d | RTX 5090 D | 1514.9 GB/s (stated within 0.9% of third-party tests) | 85% |

cuda-ops-5090d also reports reductions reaching 96.2% (MSE, N=5e7) and 97.7% (subarray sum, grid tuned to 4096
blocks) of its measured peak - i.e. ~1.45-1.48 TB/s is a realistic target for streaming kernels on RTX 5090.

## CloudRift: cuBLAS FP32 SGEMM dispatch bug on RTX 5090 (cuBLAS 13.3.0, CUDA 13.2.51)

- Batched FP32 SGEMM (256^2..8192^2, batch 4-16): cuBLAS dispatched `simt_sgemm_128x32_8x5` for the whole range
  on RTX 5090, at ~41% FMA-pipe utilization.

| Size, batch 8 | RTX 5090 cuBLAS FMA% | RTX PRO 6000 cuBLAS kernel (FMA%) | H200 kernel (FMA%) |
|---|---|---|---|
| 512 | 39% | magma_Ex (32%) | xmma_64x128x8 (69%) |
| 1024 | 41% | simt_128x64_8x5 (64%) | simt_256x128_8x4 (78%) |
| 4096 | 42% | simt_256x128_8x4 (73%) | xmma_128x128x8 (82%) |

- Their generated TMA-based FP32 kernel on RTX 5090: single 4096^3 2.213 ms vs cuBLAS 2.214 ms; 8192^3 17.558 vs
  16.581 ms (94%); batched B=8: 120% (512), 142% (1024), 162% (4096) of cuBLAS; ~68% FMA-pipe utilization.
- Tuned config on 5090: BK=32, BMxBN from 64x128 to 224x128, thread tile TM 8-28, 241 registers/thread, 89 KB
  shared memory per block, 2 stages (double buffer).

## CloudRift: matmul optimization ladder on RTX 5090

- FP32 2048^3: naive 1404 us -> register-tiled (FM=26, FN=4) 275 us (5.2x); final 96% of cuBLAS.
- FP16 tensor-core path (`m16n8k16`, FP32 accumulate) with TMA + one producer warp + mbarrier ring buffer and
  consumer warps reaching 105% of cuBLAS HGEMM; author notes cuBLAS used "an old Ampere-era cp.async-based
  kernel". Final kernel: 86 KB SMEM (two double-buffered slabs + mbarriers).
- Split-K 8-way on 128x128x16384: 7.1x.

## cuda-ops-5090d: vendor baselines and gaps on RTX 5090 D (sm_120)

| Operator | Shape | Vendor GFLOPS | Own kernel | Ratio progression |
|---|---|---|---|---|
| fp16 GEMM | 1024^3 | 124,437 | 82,000 | 0.59x -> 0.66x |
| fp32 GEMM | 8192x6144x4096 | 61,222 | 44,146 | 0.09x -> 0.72x |
| fp32 batched | 256^3 x 128 | 39,924 | 46,176 | 0.15x -> 1.16x |
| fp16 batched | 256^3 x 128 | 98,030 | 115,367 | 0.07x -> 1.18x |
| int8 matmul | 8192x4096x2048 | 210,704 | 86,678 | 0.03x -> 0.41x |

Reported step gains: adaptive tile binning (128/32) +43% for fp32; removing per-element bounds checks removed a
2.8x penalty; split-K for 512x4096x4096 +25%.

## References

- C. Lam, "Blackwell: Nvidia's Massive GPU", Chips and Cheese, 2025-06-29, https://chipsandcheese.com/p/blackwell-nvidias-massive-gpu
- D. Trifonov, "A 60% cuBLAS Performance Bug in FP32 SGEMM on the RTX 5090", https://www.cloudrift.ai/blog/beating-cublas-on-rtx-5090
- D. Trifonov, "Modern GPU Matmul Optimization", https://www.cloudrift.ai/blog/gpu-matmul-optimization
- jackbrowntkd, cuda-ops-5090d, https://github.com/jackbrowntkd/cuda-ops-5090d
- gau-nernst, gn-kernels README (memcpy BW), https://github.com/gau-nernst/gn-kernels

## KAI Notes

- Use ~1.50 TB/s (84% of 1792 GB/s) as the achievable DRAM roofline for RTX 5090 streaming kernels; >95% of that is reachable for reductions with tuned grids.
- L2 on GB202 is ~8.7 TB/s and ~130 ns; if a kernel's reuse set fits in 96 MB (5090) / 128 MB (PRO 6000), plan for L2-bandwidth-bound rather than DRAM-bound performance.
- Do not trust cuBLAS heuristics on SM120 for FP32 batched SGEMM or small/odd shapes - measure; a custom TMA + mbarrier kernel was 1.2-1.6x faster at batch 8.
- SM120 SIMT GEMMs fit at ~89 KB SMEM with 2 stages and up to 241 registers/thread; with the 99 KB per-block cap, more stages require smaller BK or tiles.
- Warp-specialized TMA producer + mbarrier ring is worth it even for `mma.sync` FP16 on SM120 (105% of cuBLAS); split-K is mandatory for skinny shapes (7.1x).
