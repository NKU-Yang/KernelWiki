---
id: doc-sageattention3-micromix-fp4-rtx5090
title: "FP4 microscaling kernels on RTX 5090: SageAttention3 (arXiv 2505.11594) and MicroMix (arXiv 2508.02343, ICLR 2026)"
author: "Jintao Zhang, Jia Wei, Pengle Zhang, et al. (Tsinghua); Wenyuan Liu, Haoqian Meng, et al. (Tianjin University)"
url: https://arxiv.org/abs/2505.11594
source_category: paper
architectures: [sm120]
tags: [attention, flash-attention, nvfp4, fp4, fp6, fp8, block-scale, quantization, gemm, warp-specialization, kernel-fusion, register-budgeting]
retrieved_at: 2026-10-15
---

# FP4 microscaling kernels on RTX 5090 (SM120): SageAttention3 and MicroMix

Both PDFs were downloaded from arXiv and read in text form:
- SageAttention3: "Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training", arXiv 2505.11594.
- MicroMix: "Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models", arXiv 2508.02343 (ICLR 2026).

## SageAttention3: NVFP4 attention on RTX 5090 - headline numbers

| Item | Value (paper) |
|---|---|
| Kernel throughput | 1038 TOPS on RTX 5090 (head_dim 128, non-causal); ~5x the fastest FlashAttention on 5090 |
| Speedup vs FlashAttention2 | 4-5x; 8-11x vs xformers (Fig. 4/5, RTX 5090) |
| End-to-end | ~3x HunyuanVideo (490 s -> 164 s), ~2.4x CogVideoX on RTX 5090 |
| GEMM motivation | FP16 matmul "about 200 TOPS on RTX5090"; FP4 microscaling matmul "about 1600 TOPS" (8x) |
| Theoretical peaks used (Table 17) | 5090: FA3-style 16-bit 209.5, FP8 419, FP4 1676 |

Table 18 (TOPS on 5090 / accuracy CosSim on CogVideoX-2B): FlashAttention2 214 / 100%; SageAttention1 479 / 99.996%;
SageAttention2 (8-bit) 643 / 99.995%; SageAttention3 (4-bit) 1038 / 99.551%. FA3 numbers are N/A on 5090 because
"FlashAttention3 can only run on Hopper GPUs, so FlashAttention2 is already the fastest on RTX5090".

## SageAttention3: format choice and quantization details

- Both QK^T and PV use FP4 microscaling (`FP4MM(A_hat, sA, B_hat, sB)`).
- NVFP4 chosen over MXFP4: NVFP4 = E2M1 with block size 1x16 and E4M3 scale; MXFP4 = E2M1, 1x32 blocks, E8M0
  scale. NVFP4 chosen for accuracy.
- Quantization: s = max(|X|)/6 per 1x16 block; Q and K smoothing borrowed from SageAttention2.
- Two-level quantization of P: P values are in [0,1], so a direct E4M3 scale only reaches max/6 ~ 0.167 and uses
  35 distinct E4M3 values; first rescaling rows to [0, 448*6] lets scales span up to 448 using 127 E4M3 values.

## SageAttention3: kernel-level tricks that matter on SM120

- Permutation for K: the FP32 accumulator register layout of the FP4 MMA differs from the FP4 operand-A register
  layout (paper Fig. 19-21). Instead of thread shuffles, permute P's columns (and correspondingly K's columns,
  fused into the K quantization kernel) so accumulator registers feed the next MMA directly.
- Reuse shuffle: the 16-element max for P's microscaling is spread over 4 threads; fusing it with the online
  softmax row max cuts redundant shuffles/max ops by 50%, ~10% whole-kernel speedup.
- Producer-warp epilogue: register pressure of the FP4 kernel makes consumer ping-pong infeasible; instead two
  producer warps ping-pong (one loads next inputs while the other stores outputs to global), consumers only move
  MMA results registers -> shared memory.
- V must be contiguous along the sequence dimension for PV; the transpose of V is fused into its quantization
  kernel to avoid an extra memory pass.

## MicroMix: MXFP4/MXFP6/MXFP8 mixed GEMM on RTX 5070 Ti laptop, RTX 5090, RTX PRO 6000

- Design: channels reordered and assigned MXFP4, MXFP6 (E3M2) or MXFP8 per channel group; each precision group is
  a separate CUTLASS-derived block-scaled GEMM accumulating FP32 into a BF16 output; reorder+quantize fused into
  one kernel (7.9%-17.0% of MicroMix runtime on RTX 5090 across tested configs, Table 6).
- Kernel speedups: 2.45-2.93x vs TensorRT-FP16 on RTX 5070 Ti laptop, up to 1.45x vs TRT-FP8; 2.29-3.38x vs
  TRT-FP16 and up to 1.74x vs TRT-FP8 on RTX 5090.
- Prefill on RTX 5090 (Llama2-7B/Llama3.1-8B, bs 8, seq 2048): ~2.0x lower latency and 2.29-2.84x less memory vs
  HF FP16; vs bitsandbytes INT8 1.80-1.84x lower latency.
- RTX PRO 6000: ~85% lower prefill latency vs Atom/QuaRot; decode throughput 1.82-3.02x vs Atom.

Table 7 (custom kernel vs CUTLASS, N=K=4096; the paper does not state which of its three GPUs):

| M | W4A4 CUTLASS / custom TFLOPS | W6A6 CUTLASS / custom | W8A8 CUTLASS / custom |
|---|---|---|---|
| 1 | 1.04 / 2.72 | 0.65 / 2.04 | 1.02 / 1.63 |
| 8 | 8.09 / 26.17 | 5.23 / 18.72 | 8.46 / 16.37 |
| 32 | 32.58 / 130.99 | 20.95 / 104.75 | 33.58 / 104.80 |
| 64 | 66.82 / 209.50 | 41.92 / 134.61 | 67.28 / 130.96 |
| 128 | 130.91 / 260.90 | 83.83 / 161.35 | 134.82 / 160.90 |

Takeaway: stock CUTLASS block-scaled GEMM tiles are poorly suited to small-M (decode-like) shapes on Blackwell
consumer parts; small-M specialized tiles gave 2-5x.

## References

- J. Zhang et al., "SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training", arXiv 2505.11594, https://arxiv.org/abs/2505.11594
- W. Liu et al., "MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models", ICLR 2026, arXiv 2508.02343, https://arxiv.org/pdf/2508.02343

## KAI Notes

- For FP4 attention on SM120 use NVFP4 (1x16 blocks, E4M3 scales) rather than MXFP4; budget ~1000 TOPS as an achieved target on RTX 5090 (SageAttention3: 1038).
- When chaining MMAs (S = QK^T -> P -> PV) with FP4 operands, fix the accumulator-vs-operand register layout mismatch by permuting K/P columns offline, not with warp shuffles.
- Fuse microscaling max computation into the online-softmax max; fuse V transpose into V quantization; fuse activation reorder into activation quantization.
- Under FP4 register pressure, move global stores to producer warps instead of consumer ping-pong.
- For small-M (M <= 128) block-scaled GEMM on SM120, don't trust default CUTLASS 128x128 tiles - specialized small-M tiles were 2-5x faster at N=K=4096.
