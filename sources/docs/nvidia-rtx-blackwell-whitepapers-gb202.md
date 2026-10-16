---
id: doc-nvidia-rtx-blackwell-whitepapers-gb202
title: "NVIDIA RTX Blackwell GPU Architecture whitepapers (GeForce v1.1 and RTX PRO v1.1): GB202/GB203 tensor-core and memory specs"
author: "NVIDIA Corporation"
url: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
source_category: official-doc
architectures: [sm120]
tags: [fp4, fp6, fp8, nvfp4, block-scale, gemm, register-budgeting, shared-memory-optimization]
retrieved_at: 2026-10-15
---

# NVIDIA RTX Blackwell GPU Architecture whitepapers: GB202/GB203 specs for SM120 kernel writers

Two official NVIDIA whitepapers were fetched and read in full text:

- GeForce: "NVIDIA RTX Blackwell GPU Architecture" v1.1 (RTX 5090 / 5080 / 5070 Ti / 5070),
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- Professional: "NVIDIA RTX Blackwell PRO GPU Architecture" v1.1 (RTX PRO 6000 Workstation and Max-Q),
  https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf

All numbers below are copied from the spec tables (Appendix A/B). "x/y" means dense/sparse. Per-SM-per-clock
rates are derived here (TFLOPS / (SMs x boost clock)) and are labelled "derived".

## GB202 full die and SM (SM120) building block

| Item | Value (whitepaper) |
|---|---|
| Full GB202 | 12 GPCs, 96 TPCs, 192 SMs, 512-bit bus (16 x 32-bit controllers), 92.2 B transistors, 750 mm2, TSMC 4N |
| Per SM | 128 CUDA cores, 4 fifth-gen Tensor Cores, 256 KB register file, 128 KB L1/shared memory |
| Full-die L2 | 128 MB (131,072 KB) |
| FP64 | 2 FP64 cores per SM; "FP64 TFLOP rate is 1/64th the TFLOP rate of FP32" |
| INT32 | INT32 and FP32 cores fully unified; a core does either FP32 or INT32 in a given clock; "not all INT operations can attain 2x speedups" vs Ada |
| GPC | 8 TPCs = 16 SMs per GPC (full die) |

Kernel-relevant consequences on SM120: the register file is 64K x 32-bit per SM (256 KB), and the 128 KB L1/shared
array is the same size as Ada; there is no larger shared memory as on SM100/SM90 datacenter parts.

## SKU table: SM counts, clocks, L2, bandwidth (RTX 5090 / 5080 / PRO 6000)

| SKU | Die | SMs | Boost MHz | L2 | Memory | BW | TGP |
|---|---|---|---|---|---|---|---|
| GeForce RTX 5090 | GB202 | 170 (11 GPCs, 85 TPCs) | 2407 | 96 MB (98,304 KB) | 32 GB GDDR7, 512-bit, 28 Gbps | 1792 GB/s | 575 W |
| GeForce RTX 5080 | GB203 (full: 84 SMs, 64 MB L2) | 84 | 2617 | 64 MB (65,536 KB) | 16 GB GDDR7, 256-bit, 30 Gbps | 960 GB/s | n/a |
| RTX PRO 6000 Blackwell Workstation | GB202 | 188 (12 GPCs, 94 TPCs) | 2617 | 128 MB (131,072 KB) | 96 GB GDDR7, 512-bit, 28 Gbps | 1792 GB/s | 600 W |
| RTX PRO 6000 Blackwell Max-Q | GB202 | 188 | 2280 | 128 MB | 96 GB GDDR7, 512-bit, 28 Gbps | 1792 GB/s | 300 W |

Verified: RTX 5090 = 170 SMs; RTX PRO 6000 = 188 SMs (of 192). The PRO 6000 keeps the full 128 MB L2 while the
5090 has 96 MB. GDDR7 on RTX Blackwell has always-on ECC (SEC) with "no performance hit" per the v1.1 update.

## Tensor-core peak table (dense/sparse TFLOPS) on SM120 SKUs

| Format | RTX 5090 | RTX 5080 | PRO 6000 WS (2617 MHz) | PRO 6000 Max-Q (2280 MHz) |
|---|---|---|---|---|
| FP4 w/ FP32 acc | 1676 / 3352 | 900.4 / 1801 | 2015.2 / 4030.4 | 1755.7 / 3511.4 |
| FP8 w/ FP16 acc | 838 / 1676 | 450.2 / 900.4 | 1007.6 / 2015.2 | 877.9 / 1755.7 |
| FP8 w/ FP32 acc | **419 / 838** | **225.1 / 450.2** | 1007.6 / 2015.2 | 877.9 / 1755.7 |
| FP16 w/ FP16 acc | 419 / 838 | 225.1 / 450.2 | 503.8 / 1007.6 | 438.9 / 877.9 |
| FP16 w/ FP32 acc | **209.5 / 419** | **112.6 / 225.1** | 503.8 / 1007.6 | 438.9 / 877.9 |
| BF16 w/ FP32 acc | **209.5 / 419** | **112.6 / 225.1** | 503.8 / 1007.6 | 438.9 / 877.9 |
| TF32 | 104.8 / 209.5 | 56.3 / 112.6 | 251.9 / 503.8 | 219.5 / 438.9 |
| INT8 | 838 / 1676 | 450.2 / 900.4 | 1007.6 / 2015.2 | 877.9 / 1755.7 |
| FP32 non-tensor | 104.8 | 56.3 | 126.0 | 109.7 |

FP6 is mentioned as supported ("RTX Blackwell adds new support for FP4 and FP6 Tensor Core operations") but no
FP6 row exists in either spec table.

## The GeForce FP32-accumulate half-rate (verified) and the PRO full rate

The GeForce whitepaper lists FP16/BF16 and FP8 with FP32 accumulate at exactly half the FP16-accumulate rate on
RTX 5090 and RTX 5080 (bold cells above). The RTX PRO whitepaper lists FP32-accumulate equal to FP16-accumulate on
RTX PRO 6000. Same GB202 silicon; this is a product-segmentation rate limit.

Derived per-SM per-clock dense rates (FLOP/clk/SM):

| Format | RTX 5090 (170 x 2.407 GHz) | RTX 5080 (84 x 2.617 GHz) | PRO 6000 WS (188 x 2.617 GHz) |
|---|---|---|---|
| FP16/BF16, FP32 acc | 512 | 512 | 1024 |
| FP16, FP16 acc | 1024 | 1024 | 1024 |
| FP8, FP32 acc | 1024 | 1024 | 2048 |
| FP8, FP16 acc | 2048 | 2048 | 2048 |
| INT8 | 2048 | 2048 | 2048 |
| FP4 (FP32 acc) | 4096 | 4096 | 4096 |
| TF32 | 256 | 256 | 512 |

Note that the GeForce FP4 row is listed only "with FP32 accumulate" and is NOT halved (4096/clk/SM on both
segments). Community measurements (see blog-gau-nernst-sm120-mma-throughput) show block-scaled MXFP8 on RTX 5090
also exceeds the 419 TFLOPS FP8/FP32-acc cap, i.e. the halving applies to the legacy (non-block-scaled)
FP16/BF16/FP8 mma paths.

## Whitepaper inconsistencies to be aware of (SM120 PRO table)

- The PRO whitepaper Table 1 lists RTX A6000 memory bandwidth as 936 GB/s while Appendix A lists 768 GB/s for the
  same card; treat comparison-column numbers with care. The PRO 6000 numbers themselves are consistent (1792 GB/s).
- Whitepaper TF32 values differ between segments (5090: 256 FLOP/clk/SM; PRO: 512), consistent with the
  FP32-accumulate cap on GeForce since TF32 always accumulates in FP32.

## References

- NVIDIA, "NVIDIA RTX Blackwell GPU Architecture" v1.1 (GeForce), Appendix A Table 3 (RTX 5090), Appendix B Table 4 (RTX 5080).
- NVIDIA, "NVIDIA RTX Blackwell PRO GPU Architecture" v1.1, Table 1 and Appendix A (RTX PRO 6000 Workstation/Max-Q), MIG Table 3 (PRO 6000: 1x96 GB, 2x48 GB, 4x24 GB instances).

## KAI Notes

- On GeForce SM120 (RTX 5090/5080), plain `mma.sync` FP16/BF16/FP8 with `.f32` accumulator runs at half rate: the BF16 roofline is 209.5 TFLOPS (5090), not 419. Use 209.5 as SOL for BF16 GEMM/attention on 5090; use 503.8 on PRO 6000.
- If accuracy allows on GeForce, FP16-accumulate (`.f16` C/D) doubles legacy-path tensor throughput and halves accumulator registers; on PRO 6000 it buys only register savings, not FLOPS.
- Size grids against the real SM count: 170 (5090), 188 (PRO 6000), 84 (5080). A 256-CTA grid on 188 SMs is 1.36 waves; prefer persistent/stream-K or tile sizes giving near-integer wave counts.
- Treat L2 as 96 MB (5090) vs 128 MB (PRO 6000) when deciding whether a weight/KV working set is L2-resident; 1792 GB/s is the DRAM spec on both GB202 SKUs.
- FP64 is 1/64 of FP32 on GB202; never route FP64-heavy work to SM120 tensor/CUDA cores expecting datacenter rates.
