---
id: blog-gau-nernst-sm120-mma-throughput
title: "gau-nernst: realistic speed-of-light and MMA-variant throughput on RTX 5090 / RTX PRO 6000 (gn-kernels, NVRTC matmul)"
author: "gau-nernst (Thien Tran)"
url: https://github.com/gau-nernst/gn-kernels
source_category: benchmark-blog
architectures: [sm120]
tags: [gemm, fp8, nvfp4, fp4, block-scale, ptx, cute-dsl, triton, register-budgeting]
retrieved_at: 2026-10-15
---

# gau-nernst: realistic SOL and MMA-variant throughput on RTX 5090 / RTX PRO 6000

Sources fetched:
- gn-kernels README (raw README.md, read in full): https://github.com/gau-nernst/gn-kernels
- Blog "Use NVRTC to explore MMA instruction variants" (2025-09-08): https://gau-nernst.github.io/nvrtc-matmul/

gn-kernels contains CuTe DSL SM120 kernels using "TMA + `mma.sync`": matmul (BF16, INT8, FP8, MXFP8, NVFP4),
non-causal BF16 attention, and an NVFP4 gated GEMM, plus Triton matmul/attention with FP16 accumulation options.

## Realistic SOL table (max TFLOPS over a sweep of large GEMM shapes) on SM120

Copied verbatim from the README "Realistic SOL" table:

| Device | memcpy BW (GB/s) | BF16 | INT8 | FP8 | MXFP8 | NVFP4 |
|---|---|---|---|---|---|---|
| RTX PRO 6000 @ 600W | (not given) | 427.62 | 795.87 | 789.28 | 741.56 | 1425.95 |
| RTX 5090 @ 600W | 1509.43 | 245.80 | 726.34 | 480.37 | 659.95 | 1306.48 |
| RTX 5090 @ 400W | 1509.05 | 229.55 | 602.79 | 447.50 | 552.20 | 1165.04 |
| B200 (Modal), for contrast | 5981.99 | 1722.53 | 3378.12 | 2668.98 | 3171.94 | 6241.55 |

Efficiency vs NVIDIA whitepaper dense peaks (computed here):

| Device | BF16 | FP8 | MXFP8 | NVFP4 | INT8 | DRAM |
|---|---|---|---|---|---|---|
| 5090 @600W (spec 209.5 / 419 / 838 / 1676 / 838 / 1792) | 117% | 115% | 79% | 78% | 87% | 84% |
| PRO 6000 @600W (spec 503.8 / 1007.6 / 1007.6 / 2015.2 / 1007.6) | 85% | 78% | 74% | 71% | 79% | n/a |

The >100% numbers on the 5090 for BF16 and FP8 mean the card sustains clocks above the 2407 MHz spec boost clock;
they also confirm those two paths are capped at the half-rate FP32-accumulate peak (BF16 ~246 vs 419 FP16-acc).

## Key finding: block-scaled MXFP8 escapes the GeForce FP32-accumulate cap

On RTX 5090 @600W, MXFP8 (block-scaled `kind::mxf8f6f4`, FP32 accumulate) reaches 659.95 TFLOPS while plain FP8
(FP32 accumulate) is 480.37 TFLOPS. The README's own SOL column assigns 838 to MXFP8 and 419 to plain FP8. So on GeForce SM120 the half-rate limit applies to legacy `mma.sync` FP16/BF16/FP8 with `.f32`
accumulators, but not to block-scaled MX/NV formats. On PRO 6000 (no cap) FP8 789 > MXFP8 742.

## RTX 5090 matmul table @400W (torch 2.12.0, triton 3.7.0, row-major x column-major)

| Kernel | M=N=K=1024 | 2048 | 4096 | SOL |
|---|---|---|---|---|
| PyTorch (cuBLAS) BF16 | 130.82 | 167.77 | 203.05 | 209.5 |
| Triton FP16 w/ FP16 accumulate | 149.46 | 279.47 | 267.37 | 419 |
| Triton FP8 | 173.86 | 320.33 | 379.08 | 419 |
| PyTorch (CUTLASS) row-scaled FP8 | 174.76 | 310.69 | 390.17 | 419 |
| Inductor (Triton) row-scaled FP8 | 178.96 | 309.08 | 392.09 | 419 |
| PyTorch (cuBLAS) INT8 | 209.72 | 466.03 | 583.56 | 838 |
| Triton INT8 | 175.22 | 470.53 | 530.47 | 838 |
| PyTorch (cuBLAS) MXFP8 | 171.2 | 462.82 | 531.52 | 838 |
| PyTorch (cuBLAS) NVFP4 | 215.09 | 762.6 | 1139.85 | 1676 |

Observations: at 1024^3 every dtype is launch/wave limited (130-215 TFLOPS); NVFP4 needs >= 4096^3 to approach
68% of peak at 400W. Triton FP8 on SM120 is competitive with CUTLASS row-scaled FP8.

## NVRTC blog: FP16 vs FP32 accumulate measured on RTX 5090 (SM120)

RTX 5090 @600W, M=N=K=8192, author's NVRTC-compiled `mma.sync` kernel:

| dtype | FP32 accumulate | FP16 accumulate |
|---|---|---|
| FP16 | 233 | 350 |
| FP8 | 465 | 692 |

Comparison with PyTorch eager on 5090: BF16 230 (PyTorch) vs 233 (mine); INT8 678 vs 736; FP8 473 vs 465.
On RTX 4090 @450W: INT8 390 / PyTorch 597, INT4 756, FP16 w/ FP32 acc 161 / PyTorch 156, FP16 w/ FP16 acc 272.

Author statements: FP16 accumulate uses "2x fewer registers for the accumulator, we can double either BLOCK_M or
BLOCK_N"; FP8 with FP16 accumulation is slightly faster than MXFP8 MMA on 5090 ("I suppose the MXFP8 MMA
instruction is spending some extra energy on the block scaling circuitry"). Gains are below 2x (FP8 465 -> 692,
+49%) because the card becomes power-limited.

PTX templates used:
```
mma.sync.aligned.<shape>.row.col.f32.<ab>.<ab>.f32    // FP32 accumulate
mma.sync.aligned.<shape>.row.col.f16.<ab>.<ab>.f16    // FP16 accumulate
mma.sync.aligned.m16n8k64.row.col.satfinite.s32.<ab>.<ab>.s32  // INT4
```
Shapes: FP16/BF16 `m16n8k16`, FP8/INT8 `m16n8k32` (both A tiles are 32 bytes wide). NVRTC compile was 0.0095 s vs
9.19 s for `torch.utils.cpp_extension.load_inline()`.

## Power sensitivity on RTX 5090 (SM120)

Going from 400 W to 600 W: NVFP4 1165 -> 1306 (+12%), INT8 603 -> 726 (+20%), MXFP8 552 -> 660 (+20%), BF16
229.6 -> 245.8 (+7%). memcpy is power-insensitive (1509 GB/s, 84% of 1792 spec). Tensor-heavy SM120 kernels
are power/clock bound; always record the power limit with a measurement.

## References

- gau-nernst, gn-kernels README "Realistic SOL" and "Matmul" tables, https://github.com/gau-nernst/gn-kernels
- gau-nernst, "Use NVRTC to explore MMA instruction variants", 2025-09-08, https://gau-nernst.github.io/nvrtc-matmul/
- NVIDIA RTX Blackwell whitepapers (spec peaks used for efficiency ratios); see doc-nvidia-rtx-blackwell-whitepapers-gb202.

## KAI Notes

- On RTX 5090 use realistic ceilings, not spec: BF16 ~230-246, FP8 (f32 acc) ~450-480, MXFP8 ~550-660, NVFP4 ~1165-1306 TFLOPS, memcpy ~1509 GB/s (400-600 W). On PRO 6000 @600W: BF16 ~428, FP8 ~789, NVFP4 ~1426.
- On GeForce, an FP8 GEMM that must accumulate in FP32 should prefer block-scaled `kind::mxf8f6f4` (MXFP8, scale = 1.0 if needed) over plain `e4m3` `mma.sync ... .f32` - measured 660 vs 480 TFLOPS on 5090.
- FP16-accumulate is the other GeForce escape hatch (FP16 233 -> 350, FP8 465 -> 692) and halves accumulator registers; validate numerics first.
- Don't expect speedups at M=N=K <= 1024 on SM120 - every dtype stalls near 130-215 TFLOPS; fuse or batch small GEMMs instead.
- Report power limit and clocks alongside TFLOPS; a 400 W vs 600 W cap changes tensor throughput by 7-20%.
