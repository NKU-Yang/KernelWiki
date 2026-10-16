---
id: blog-gb10-dgx-spark-sm121-measurements
title: "GB10 / DGX Spark (SM121) community measurements: FP4 peak, padded vs packed NVFP4, custom FP4 GEMM, FP8-vs-NVFP4 serving trap"
author: "NVIDIA Developer Forums users (adg1 et al., VincentKaufmannUAE, Nanodeoclus); Jeongkyu Shin and Kyujin Cho (Backend.AI); ai-muninn.com; NVIDIA DGX Spark docs"
url: https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618
source_category: community-note
architectures: [sm120]
tags: [nvfp4, fp4, fp8, block-scale, gemm, quantization, decode, ptx]
retrieved_at: 2026-10-15
---

# GB10 / DGX Spark (SM121) community measurements

GB10 (DGX Spark and OEM boxes: ASUS GX10, HP ZGX Nano, Lenovo PGX) carries a compute-capability 12.1 Blackwell GPU.
It is SM120-family: `mma.sync`-based tensor cores, no `tcgen05`, no TMEM, no `wgmma`. Sources fetched are listed in
References.

## GB10 hardware facts (official + Backend.AI)

| Item | Value | Source |
|---|---|---|
| Compute capability | 12.1 (sm_121 / sm_121a) | Backend.AI blog, forum |
| SMs / CUDA cores | 48 SMs, 6,144 CUDA cores | Backend.AI; NVIDIA docs (6,144 cores) |
| Memory | 128 GB LPDDR5x unified, 256-bit, 4266 MHz, 273 GB/s | NVIDIA DGX Spark hardware docs |
| FP4 marketing peak | "up to 1 PFLOP at FP4 precision with sparsity" | NVIDIA docs |
| SoC TDP | 140 W | NVIDIA docs |
| Max warps per SM | 48 (vs 64 on SM100) | Backend.AI |
| Shared memory per SM | Backend.AI states 128 KB (this is the unified L1/shared array; CUDA guide max per block on 12.x is 99 KB) | Backend.AI |
| Absent features | tcgen05, TMEM (256 KB/SM on SM100), warpgroup MMA | Backend.AI |

## Measured FP4 tensor peak on GB10 (register-resident microbenchmark)

Forum thread "GB10 really does hit ~1 PFLOP NVFP4 (2:4 sparse) - measured" (adg1, 2026-06-17); kernels use
`mma.sync` / `mma.sp` with `kind::mxf4nvf4` (packed FP4) or `kind::mxf8f6f4` (FP4 padded to a byte):

| Variant | Measured | vs spec |
|---|---|---|
| NVFP4 dense, `kind::mxf8f6f4` (byte-padded) | ~256 TFLOPS | 51% of 500 |
| NVFP4 dense, `kind::mxf4nvf4` (packed) | ~511 TFLOPS | 102% of 500 |
| NVFP4 2:4 sparse, `mxf8f6f4` | ~511 TFLOPS | 51% of 1000 |
| NVFP4 2:4 sparse, `mxf4nvf4` | ~1014-1022 TFLOPS | 102% of 1000 |

Reproductions (dense / sparse TFLOPS): ASUS GX10 497.5 / 994.8; HP ZGX Nano 503.9 / 1007.8; Founders Edition Spark
486.3 / 973.2; Lenovo PGX 498.4 / 996.7. Caveats from the author: "register-resident, no global/shared memory
traffic", varies with clocks/thermals. Build note: must compile with `-gencode=arch=compute_121a,code=sm_121a` to get
block-scaled FP4 MMA.

## Custom NVFP4 GEMM on GB10: 129 TFLOPS with cached pre-quantized weights

Forum post by VincentKaufmannUAE (2026-02-24), CUTLASS-based (`cutlass::nv_float4_t<cutlass::float_e2m1_t>`,
`mma.sync.aligned.block_scale`), threadblock tile 128x128x128, ClusterShape 1x1x1 ("No multicast (SM121)"), one
quantization thread per 16-element scale block, UE4M3 scales, `SfKMajorAtom` interleaved scale layout:

| Shape (M x N x K) | Mode | Time | TFLOPS |
|---|---|---|---|
| 2048 x 2880 x 7680 | dynamic (quantize each call) | 1.993 ms | 46 |
| 2048 x 2880 x 7680 | cached pre-quantized weights | 0.702 ms | 129 |
| 256 x 2880 x 2880 | cached | 0.050 ms | 85 |
| FLUX.1 MMDiT attention, 2048 patches | - | 0.29 ms | 133 |

Reported 1.4-2.4x faster than BF16 at small-to-medium M; BF16 `F.linear` wins at M >= 4096 (compute-bound) and on
SigLIP-SO400M (FP4 0.64-0.92x of BF16). A reply (Nanodeoclus, 2026-02-25) corrects the author: both sm_120a and
sm_121a support MXFP4 block-scaled MMA (ue8m0 and ue4m3 scales) since nvcc 12.9.0.

## Serving-level trap: NVFP4 slower than FP8 on GB10 because of kernel dispatch

ai-muninn.com (2026-04-21, updated 2026-05-06 and 2026-08-23), Qwen 3.6-35B-A3B on one DGX Spark, vLLM 0.19.1:
FP8 53.8 tok/s (driver 580.142; 51.8 with 580.126) vs NVFP4 40.8 tok/s (-24%) because vLLM fell back to Marlin
(dequantize FP4 -> BF16). The original explanation (SM121 lacks `cvt.rn.satfinite.e2m1x2.f32`) was retracted by
the author on 2026-08-23: "What I measured back then was the software layer picking the wrong kernel path". The
driver upgrade 580.126 -> 580.142 gave +4% FP8 throughput ("FP8 kernels now run native SM121 code instead of
compatibility mode").

## Implications for SM121 vs SM120 kernels

- GB10 dense NVFP4 peak ~500 TFLOPS = 48 SMs x 4096 FLOP/clk/SM at ~2.5 GHz - the same per-SM FP4 rate as RTX 5090
  and RTX PRO 6000 (derived from NVIDIA whitepapers); the GPU is ~1/4 of a 5090 in SMs.
- 273 GB/s LPDDR5x is ~6.6x less than RTX 5090/PRO 6000 GDDR7 (1792 GB/s); decode-phase kernels on GB10 are
  overwhelmingly bandwidth bound, so 4-bit weights matter more there than on GB202 - provided the native
  block-scaled path is actually selected.

## References

- NVIDIA Developer Forums, "GB10 really does hit ~1 PFLOP NVFP4 (2:4 sparse) - measured, with an open-source tool to reproduce it", https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618
- NVIDIA Developer Forums, "Custom FP4 CUDA Kernel - 129 TFLOPS on DGX Spark with Pre-Quantized Weight Cache", https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600
- J. Shin, K. Cho, "Inside NVIDIA DGX Spark: Is DGX Spark Actually Blackwell?", Backend.AI, 2026-02-19, https://www.backend.ai/blog/2026-02-is-dgx-spark-actually-a-blackwell
- ai-muninn, "DGX Spark NVFP4 trap: GB10 FP8 wins", https://ai-muninn.com/en/blog/dgx-spark-nvfp4-trap-gb10-fp8-wins
- NVIDIA, DGX Spark Hardware Overview, https://docs.nvidia.com/dgx/dgx-spark/hardware.html

## KAI Notes

- On GB10 compile with `-gencode=arch=compute_121a,code=sm_121a` (arch-specific `a` target) or block-scaled FP4 MMA is unavailable; RTX 5090/PRO 6000 need `sm_120a` likewise.
- Use packed `kind::mxf4nvf4` for FP4 (~511 TFLOPS dense on GB10); byte-padded `kind::mxf8f6f4` FP4 halves throughput (~256).
- Quantize weights once and cache them: on-the-fly quantization cut a GB10 NVFP4 GEMM from 129 to 46 TFLOPS at 2048x2880x7680.
- Verify the dispatched kernel (nsys/ncu kernel names) before concluding NVFP4 is slow on SM121; Marlin/BF16 fallbacks masquerade as hardware limits.
- No TMA multicast / clusters > 1 in CUTLASS SM12x configs used by the community (ClusterShape 1x1x1); design for single-CTA tiles.
