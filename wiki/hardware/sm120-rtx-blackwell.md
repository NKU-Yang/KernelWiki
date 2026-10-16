---
id: hw-sm120-rtx-blackwell
title: "SM120 RTX Blackwell - GeForce and Workstation Blackwell"
type: hardware
architectures: [sm120]
tags: [fp4, fp6, fp8, nvfp4, block-scale, tma, cluster, ldmatrix]
confidence: verified
related:
  - lang-ptx-sm120
  - lang-cutlass-sm120-cute
  - migration-sm100-to-sm120
  - kernel-sm120-blockscaled-gemm-moe
  - kernel-sm120-attention
  - technique-sm120-blockscaled-mma-sync
  - pattern-sm120-porting-failures
  - hw-nvfp4
  - technique-vectorized-loads
sources:
  - doc-nvidia-cuda-gpus-sm120
  - doc-nvidia-tensorrt-rtx-sm12
  - doc-nvidia-blackwell-tuning-guide-sm120
  - doc-nvidia-rtx-pro-6000-blackwell-specs
  - paper-blackwell-sm120-microbenchmarking-rtx5080
  - doc-cuda-programming-guide-cc12
  - doc-nvidia-rtx-blackwell-whitepapers-gb202
  - doc-ptx-isa-sm120-targets-tma-async
  - doc-cuda-toolkit-release-notes-sm12x
  - doc-nsight-compute-sm120-tensor-metrics
  - doc-accurate-models-tensor-cores-rtx-pro-6000
  - blog-gau-nernst-sm120-mma-throughput
  - blog-sm120-memory-system-and-simt-gemm-measurements
  - blog-gb10-dgx-spark-sm121-measurements
  - doc-cutlass-sm120-collective-builder-internals
aliases:
  - sm120
  - SM120
  - sm_120
  - compute_120
  - "SM 12.x"
  - "RTX PRO 6000 Blackwell"
  - "GeForce RTX 50"
  - "RTX 5090"
  - GB202
  - sm121
  - GB10
evidence_basis:
  - source_id: doc-cuda-programming-guide-cc12
    evidence_type: official-doc
  - source_id: doc-nvidia-rtx-blackwell-whitepapers-gb202
    evidence_type: official-doc
  - source_id: blog-gau-nernst-sm120-mma-throughput
    evidence_type: benchmark
  - source_id: doc-cutlass-sm120-collective-builder-internals
    evidence_type: upstream-code
  - source_id: doc-cuda-programming-guide-cc12
    evidence_type: reproduction
---

# SM120 RTX Blackwell - GeForce and Workstation Blackwell

## Scope

SM120 is CUDA compute capability 12.0: RTX PRO 6000 Blackwell (Workstation, Max-Q, Server Edition) and GeForce RTX 50 (GB202/GB203/...). SM121 (CC 12.1) is GB10 / DGX Spark and shares the SM12x family (`sm_120f` cubins run on both). SM120 is a separate optimization target from SM100/SM100a datacenter Blackwell (B200/GB200): there is no `tcgen05`, no TMEM, no `wgmma`, no 2-SM cooperative MMA, and GeForce/RTX CUTLASS paths use cluster shape 1x1x1.

## Per-SM Resource Limits (measured)

Measured with `cudaDeviceGetAttribute` on an RTX PRO 6000 Blackwell Server Edition (CC 12.0, driver 595.58.03, CUDA 13.0) on 2026-10-15. These match the CUDA 13.4 Programming Guide; the older Blackwell Tuning Guide statement of "128 KB shared memory per SM" and "32 blocks per SM" is wrong for CC 12.0 (128 KB is the unified L1 + shared array).

| Limit | SM120 value | SM100 (B200) for contrast |
|---|---|---|
| Shared memory per SM | 100 KB (102400 B) | 228 KB |
| Max dynamic shared memory per block (opt-in) | 99 KB (101376 B) | 227 KB |
| Default per-block shared memory without opt-in | 48 KB | 48 KB |
| Reserved shared memory per block | 1 KB | 1 KB |
| Unified L1 + shared per SM | 128 KB | 256 KB |
| Carveout choices | 0, 8, 16, 32, 64, 100 KB | up to 228 KB |
| Resident blocks per SM | 24 | 32 |
| Resident threads / warps per SM | 1536 / 48 | 2048 / 64 |
| 32-bit registers per SM / per block | 64K / 64K | 64K / 64K |
| FP32 : FP64 (non-tensor) | 64 : 1 | 2 : 1 |

CUTLASS hard-codes the same budget: `cutlass::arch::sm120_smem_capacity_bytes = 101376`. Any SM90/SM100 tile config that assumes >99 KB of shared memory per CTA fails to launch on SM120; recompute stages.

## SKU Table

From the NVIDIA RTX Blackwell (GeForce) and RTX Blackwell PRO whitepapers v1.1; Server Edition values measured locally.

| SKU | SMs | Boost / max SM clock | L2 | Memory | DRAM BW |
|---|---|---|---|---|---|
| GeForce RTX 5090 (GB202) | 170 | 2407 MHz | 96 MB | 32 GB GDDR7, 512-bit | 1792 GB/s |
| GeForce RTX 5080 (GB203) | 84 | 2617 MHz | 64 MB | 16 GB GDDR7, 256-bit | 960 GB/s |
| RTX PRO 6000 Workstation (GB202) | 188 | 2617 MHz | 128 MB | 96 GB GDDR7, 512-bit | 1792 GB/s |
| RTX PRO 6000 Max-Q (GB202) | 188 | 2280 MHz | 128 MB | 96 GB GDDR7, 512-bit | 1792 GB/s |
| RTX PRO 6000 Server Edition (measured) | 188 | 2430 MHz max, 600 W | 128 MB (80 MB max persisting) | 96 GB GDDR7, 512-bit | ~1792 GB/s spec |

Full GB202 has 192 SMs. Each SM: 128 CUDA cores, 4 fifth-generation Tensor Cores, 256 KB register file, 128 KB L1/shared.

## Tensor-Core Peak Rates

Dense TFLOPS from the whitepapers (sparse = 2x). Per-SM-per-clock rates derived as TFLOPS / (SMs x clock).

| Format | RTX 5090 | PRO 6000 WS | FLOP/clk/SM (PRO) | PRO 6000 Server @2430 MHz (derived) |
|---|---|---|---|---|
| NVFP4 / MXFP4 (FP32 acc) | 1676 | 2015.2 | 4096 | ~1871 |
| FP8 / FP6, FP32 acc | 419 (half rate) | 1007.6 | 2048 | ~936 |
| FP8, FP16 acc | 838 | 1007.6 | 2048 | ~936 |
| FP16/BF16, FP32 acc | 209.5 (half rate) | 503.8 | 1024 | ~468 |
| FP16, FP16 acc | 419 | 503.8 | 1024 | ~468 |
| INT8 | 838 | 1007.6 | 2048 | ~936 |
| TF32 | 104.8 | 251.9 | 512 | ~234 |

GeForce SKUs halve FP16/BF16/FP8 MMA with FP32 accumulation (product segmentation); RTX PRO does not. Block-scaled MXFP8 is not capped on GeForce: gau-nernst measured MXFP8 660 TFLOPS vs plain FP8 480 TFLOPS on RTX 5090.

## Measured Speed-of-Light

| Device | BF16 | FP8 | MXFP8 | NVFP4 | INT8 | memcpy |
|---|---|---|---|---|---|---|
| RTX PRO 6000 @600 W (gau-nernst) | 427.6 | 789.3 | 741.6 | 1426.0 | 795.9 | - |
| RTX 5090 @600 W (gau-nernst) | 245.8 | 480.4 | 660.0 | 1306.5 | 726.3 | 1509 GB/s (84%) |

Chips and Cheese on RTX PRO 6000: L2 bandwidth ~8.7 TB/s, L2 latency just over 130 ns, DRAM latency ~329 ns, L1/shared 128 B/clk/SM. Practical targets: ~70-85% of whitepaper peak for large GEMMs, ~84% of spec DRAM bandwidth for streaming kernels.

## Instruction Families Available

- Tensor cores: warp-level `mma.sync` only. Narrow precision via `.kind::f8f6f4` (m16n8k32), `.kind::mxf8f6f4` (m16n8k32, ue8m0, 1X), `.kind::mxf4` (m16n8k64, ue8m0, 2X), `.kind::mxf4nvf4` (m16n8k64, 2X ue8m0 or 4X ue4m3/ue8m0). Sparse: `mma.sp::ordered_metadata` (m16n8k64 / m16n8k128). FP8 `mma` compiles to native `QMMA.16832` SASS on SM120.
- All FP4/FP6 and block-scaled forms, FP4/FP6 `cvt`, `ldmatrix .m8n16/.m16n16` with `.b4x16_p64/.b6x16_p32`, and `setmaxnreg` require `sm_120a` or `sm_120f` (verified with nvcc 13.0; rejected for plain `sm_120`).
- TMA (`cp.async.bulk.tensor`), `mbarrier` with tx-count, PDL (`griddepcontrol`), `elect.sync`, clusters and DSMEM (portable cluster size 8) are available. CUTLASS SM120 GEMMs still use cluster 1x1x1 and TMA into `.shared::cta`; TMA multicast is documented as optimized only for sm_90a/sm_10x.
- Not available: `tcgen05.*`, TMEM, `wgmma`, TMA `.cta_group`.

## Profiling Notes

Nsight Compute reports SM120 chips as GB20x. Use `sm__pipe_tensor_cycles_active` (and `sm__inst_executed_pipe_tensor`) for tensor utilization; `sm__pipe_tc_*` metrics belong to tcgen05 chips. Confirm metric availability with `ncu --query-metrics` on the target.

## Optimization Implications

1. Recompute occupancy and stage counts against 99 KB/CTA and 100 KB/SM; two 50 KB CTAs or one 99 KB CTA fill an SM.
2. Use `mma.sync` block-scaled paths (see `technique-sm120-blockscaled-mma-sync`) for FP4/FP8 GEMM; use NVFP4 (`mxf4nvf4` 4X ue4m3) for the 4096 FLOP/clk/SM rate.
3. On GeForce, prefer FP16 accumulation or block-scaled formats when FP32-accumulate halving limits throughput and accuracy allows.
4. For memory-bound kernels, the 96-128 MB L2 is large: reuse via L2 tiling and persisting-L2 windows; DRAM is GDDR7 (~1.5 TB/s achievable), not HBM.
5. Treat B200 throughput claims and kernel designs (FA3/FA4, DeepGEMM SM100) as non-transferable.
