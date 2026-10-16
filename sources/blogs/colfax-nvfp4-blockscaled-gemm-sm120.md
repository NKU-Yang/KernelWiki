---
id: blog-colfax-nvfp4-blockscaled-gemm-sm120
title: "Colfax Research: NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x) with CuTe DSL"
author: "Colfax Research"
url: https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
source_category: benchmark-blog
architectures: [sm120]
tags: [nvfp4, fp4, block-scale, gemm, cute-dsl, tma, ldmatrix, pipeline-stages, register-budgeting]
retrieved_at: 2026-10-15
---

# Colfax Research: NVFP4 blockscaled GEMM on RTX PRO Blackwell (SM12x) with CuTe DSL

Source: Colfax Research CUTLASS tutorial "NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)"
(page dated June 20, 2026 in the fetched copy), https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/ .
Targets SM120 (RTX PRO 6000 Blackwell Server Edition) and SM121 (DGX Spark).

## The only NVFP4 MMA on SM12x

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

| Fragment | Shape | Type |
|---|---|---|
| A | 16 x 64 | E2M1 |
| B | 64 x 8 | E2M1 |
| SFA | 16 x 4 | UE4M3 (one scale per 16 K-elements) |
| SFB | 4 x 8 | UE4M3 |
| C/D | 16 x 8 | F32 |

- m16n8k64 is described as the only supported shape for NVFP4 block scaling on SM12x; accumulation is F32.
- E2M1 values: {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}; UE4M3 = unsigned 8-bit float, 4 exp / 3 mantissa bits;
  micro-block size 16.

## Scale-factor thread-value layouts on SM120 (CuTe notation)

- SFA TV layout: `((2, 2, 8),(16, 4)):((32, 0, 4),(0, 1))`
- SFB TV layout: `((4, 8), (16, 4)):((0, 4), (0, 1))`
- Scale factors are replicated across lanes: SFA 512 bytes per warp (2x replication), SFB 2048 bytes per warp (4x).
- Per-thread register fragments: `tCrSFA_tile ((16, 4), 2, 2):((0, 1), 4, 8)` = 16 bytes/lane;
  `tCrSFB_tile ((16, 4), (2, 4), 2):((0, 1), (16, 4), 32)` = 64 bytes/lane.
- Staged SMEM layout template for SFA/SFB:
  `(((32,4), REST_M/N), ((16,4), 1, REST_K)) : (((16, 4), 512 * REST_K), ((0, 1), 4, 512))`
- GMEM uses the same "interleaved" scale-factor layout convention as SM10x (our inference: SM100-style quantizers
  producing that layout can be reused, even though the GEMM kernels themselves are incompatible).

## Tutorial kernel configuration on SM120

| Parameter | Value |
|---|---|
| CTA tile (bM, bN, bK) | (128, 128, 128) |
| TiledMMA atom layout | (4, 2, 1): 4 atoms along M, 2 along N |
| Threads | 8 warps = 256 |
| GMEM -> SMEM | TMA for A, B, SFA and SFB, one producer-consumer pipeline for all four streams |
| SMEM -> RMEM | `ldmatrix` for A/B (via CuTe); universal copy atom (`CopyUniversalOp`) for scale factors |
| Max bM / bN supported by helpers | 128 |

CUTLASS helpers named: `sm120_utils.get_permutation_mnk()`, `blockscaled_utils.tile_atom_to_shape_SF()`,
`blockscaled_utils.sm120_make_smem_layout_sfa/sfb()`, `sm120_utils.get_layoutSFA_TV()/get_layoutSFB_TV()`,
`sm120_utils.partition_fragment_SFA/SFB()`. Reference code lives under
`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm` (extending `dense_gemm.py`).

## Measured performance on RTX PRO 6000 Blackwell Server Edition (SM120)

- Peaks used: dense FP16 500 TFLOP/s, dense FP4 2000 TFLOP/s.
- The tutorial NVFP4 GEMM reaches "around 60% of peak theoretical throughput" (~1200 TFLOPS implied) and about 3x
  over PyTorch dense FP16 `torch.mm`; also compared against cuBLAS NVFP4 blockscaled GEMM (exact numbers in the
  post's charts were not extracted).
- 2k x 2k problems underperform due to wave quantization: 256 CTAs of 128x128 on 188 SMs = 1.36 waves.
- Quantization time is excluded from the benchmark.

## Pitfalls called out for SM12x

- "SM10x GEMM kernels are completely incompatible with SM12x" (no tcgen05/TMEM; different MMA atoms).
- 128 is the largest bM/bN the current CUTLASS helper functions support on SM12x.
- Scale-factor replication across lanes costs registers and SMEM bandwidth; the per-lane SF fragment for B (64 B)
  is 4x that of A (16 B) (our inference: this register cost grows with per-warp N extent).

## References

- Colfax Research, "NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs (SM12x)", https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/
- NVIDIA CUTLASS repository, CuTe DSL example `blackwell_geforce/kernel/blockscaled_gemm` (named in the post).

## KAI Notes

- For NVFP4 on SM120/SM121 emit `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...ue4m3`; one MMA consumes 16x64 A, 64x8 B and 16x4 / 4x8 UE4M3 scales.
- Start from CTA 128x128x128, 8 warps, TMA for A/B/SFA/SFB, `ldmatrix` for A/B; expect ~60% of the 2000 TFLOPS PRO 6000 peak from a straightforward pipeline and need persistence/stream-K tricks for more.
- Choose grid/tile so CTA count is near a multiple of SM count (188 on PRO 6000, 170 on 5090) - 2k x 2k with 128x128 tiles is 1.36 waves.
- Keep GMEM scale-factor layout identical to SM100 interleaved layout so existing quantizers are reusable; rewrite only the GEMM mainloop.
- Budget registers for replicated scale fragments (16 B/lane SFA, 64 B/lane SFB per k-tile) on top of A/B fragments and FP32 accumulators.
