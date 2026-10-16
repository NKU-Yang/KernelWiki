---
id: blog-sm120-sparse-mla-dsv4-2026h2
title: "SM120/SM121 sparse MLA, DSA indexer and DeepGEMM for DeepSeek-V4 / V3.2 / GLM (Jun-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/flashinfer-ai/flashinfer/pull/5914
source_category: community-note
architectures: [sm120]
tags: [mla, sparse-attention, attention, decode, prefill, nvfp4, fp8, fp4, block-scale, gemm, grouped-gemm, moe, mbarrier, tma, ldmatrix, warp-specialization, persistent-kernel, pipeline-stages, ptx, pdl, cuda-cpp]
retrieved_at: 2026-10-15
---

# SM120/SM121 sparse MLA and DeepSeek-V4 kernels (2026 H2 digest)

The trtllm-gen sparse-MLA kernels are tcgen05-only and compile out on SM120 (TensorRT-LLM#16224), so
DeepSeek-V4 / V3.2 (DSA) / GLM-5.x on RTX PRO 6000, RTX 5090 and GB10 run on a separate family of
register-MMA kernels in FlashInfer (`flashinfer.mla.SparseMLASm120Wrapper`,
`trtllm_batch_decode_sparse_mla_dsv4`) and on DeepGEMM's SM120 port for the indexer and MoE.

## DeepGEMM#324 - first-class sm_120 support for DeepGEMM
- Merged 2026-06-24 (nv_dev branch, NVIDIA DevTech APAC). https://github.com/deepseek-ai/DeepGEMM/pull/324
- Dense `sm120_fp8_fp4_gemm_1d1d`: FP8 E4M3 UE8M0 block-scaled 1D1D (NT/NN/TN/TT), FP4 E2M1 (`mxf4nvf4`, gran_k 32), BF16, TF32, mixed FP8_A x FP4_B (`mxf8f6f4 .e4m3.e2m1`) both directions. **AB-swap small-M path (M=1..16) via TMA + per-element strided epilogue**; wave-packing heuristic (BLOCK_M 64/128); split-K for small dims.
- M-grouped contiguous/masked and K-grouped contiguous grouped GEMM (FP8/FP4/BF16); einsum `sm120_bmk_bnk_mn`; TF32 HC-prenorm GEMM + square-sum with split-K; FP8/FP4 MQA logits (ragged warp-specialized with L2-cached KV, split-KV, and paged).
- RTX PRO 6000 roofline used: FP8 ~814, FP4 ~1628, BF16 ~445 TFLOPS. Measured: dense FP8 778 (96%), FP4 1561 (96%), BF16 444 TFLOPS; grouped contiguous FP8 814 / FP4 1571; masked 670 / 1363; MQA logits ragged FP4 1049, paged FP4 685, ragged FP8 636, paged FP8 547 TFLOPS.
- DeepGEMM#447 (2026-09-23) migrates these SM120 kernels to main's API/DeepJIT runtime incl. DSv4.1 MXFP8/MXFP4 sparse and paged sparse MQA. DeepGEMM#369 adds `indexer_n_heads = 16`.

## flashinfer#4802 - Refactor Sparse MLA SM120
- Merged 2026-09-03. https://github.com/flashinfer-ai/flashinfer/pull/4802
- RTX PRO 6000: DSv4 dual cache H64 (topk 128+512) T=1 14.80 -> 11.40 us (1.30x), T=8 1.23x, T=16 1.24x; bit-identical.
- Analytical `chunks_per_block` (cpb) model + measured decode/prefill crossover replace the per-shape autotune sweep (the sweep profiled L2-resident synthetic indices, which distorts cpb when production caches are DRAM-resident). Decode any heads in [1,128], prefill any T, topk % 64 == 0. swapAB prefill 1.12-2.37x over the previous "MG" impl.
- New model types `GLM53_NOPE` and `DOTS3_SWA` (d_qk 1088, d_v 1024, 1160 B/token footer-scale); public `SparseMLASm120Wrapper` (CUDA-graph safe).

## flashinfer#4732 / #5048 - barrier hangs in SM120 sparse-MLA prefill (GB10)
- #4732 (2026-09-01): prefill used **named barrier 1 as an asymmetric handshake (count 384)**: 256 math threads `barrier.cta.arrive`, 128 IO threads `barrier.cta.sync`. Math could arrive for tile ti, run ti+1 and arrive again before IO synced ti - a surplus arrival completes the phase without IO (PTX ISA forbids arrive then another barrier on the same id before reset). Fix: **alternate barrier ids 1 and 5 by tile parity**; two ids suffice because math cannot start ti until IO returned from sync of ti-2. https://github.com/flashinfer-ai/flashinfer/pull/4732
- #5048 (2026-09-09): swapAB prefill had 4 IO warps but only `BI` IO threads gather (64 for DSv3.2, 32 for DOTS3); idle IO threads still spun on `mbarrier_wait_parity` and, starved past a full phase window, waited on a parity that never recurs (~2% of launches at H=128). Fix: `if (io_tid >= BI) return;` before the pipeline loop. https://github.com/flashinfer-ai/flashinfer/pull/5048

## flashinfer#4955 - NVFP4 sparse MLA for DeepSeek V4 Flash
- Merged 2026-09-07. https://github.com/flashinfer-ai/flashinfer/pull/4955
- 384 B/token paged NVFP4 cache: 448-dim NoPE as E2M1 with one E4M3 scale per 16, 64-dim RoPE BF16. `kv_cache_format="nvfp4"`.
- Single-launch streaming prefill gathers paged NVFP4 V directly, transposes candidate tiles in CTA-local smem, double-buffers the next tile (no global transposed-V workspace); grouped split-K decode with autotuned chunks-per-block.
- RTX PRO 5000: prefill T=8192 H=64 K=128 3910 -> 2338 us (1.67x vs FP8 cache); K=128+512 1.42x; decode T=8 H=64 43.0 -> 32.8 us (1.31x).

## flashinfer#5197 - unify SM120 sparse-MLA execution, calibration and DSv4.1
- Merged 2026-09-18. https://github.com/flashinfer-ai/flashinfer/pull/5197
- DSv4.1 dual cache (FP8 primary + optional FP4/FP8 extra), runtime page sizes incl. non-powers of two for six families (`dsv4`, `dsv3_2`, `glm_nsa`, `glm53_nope`, `dots3_swa`, `dsv4_1`), per-configuration **L2-cold** calibration. 947 tests on SM120 and SM121.

## flashinfer#5097 - halve cpb L2 guard rail on integrated GPUs
- Merged 2026-09-11. https://github.com/flashinfer-ai/flashinfer/pull/5097
- GB10 reports 24 MiB L2, but the measured knee is cpb 5-6 (~11 MB footprint) because the SoC L2 is shared with the CPU fabric; when `props.is_integrated`, the planner uses half the reported L2.

## flashinfer#5914 / #5949 / #6000 / #6027 - Cake-generated DSv4 NVFP4 sparse-MLA decode
- #5914 merged 2026-10-02. https://github.com/flashinfer-ai/flashinfer/pull/5914
- Layout `Dsv4Nvfp4Layout`: 224 B E2M1 nope + 128 B BF16 rope + 28 E4M3 scales per token.
- Kernel: **one 320-thread CTA per token x 16-head block = 8 math warps + 2 gather warps**; 16 B `cp.async` gather of 64-candidate chunks into a 2-stage ring; NVFP4 QK on `mma.sync m16n8k64 kind::mxf4nvf4` with native scales; fp32 base-2 online softmax; FP8 PV on `kind::f8f6f4 m16n8k32`; one or two 16-head tiles share a gather; split-K + merge.
- vs hand-written route: RTX PRO 6000 geomean **1.877x** (118/118 rows faster, max 3.59x), RTX 5090 **2.000x**.
- #5949: GB10 (48 SMs) planner: below 64 SMs keep split grid <= SMs/3 CTAs, gather streams <= SMs/6, >= 2 chunks per CTA, no wave-quantization split.
- #6000: batched merge kernel (loads all split LSE slots up front, 4 partial groups in flight) for num_splits >= 8; bit-identical.
- #6027 pitfall: **CUDA 12.9 ptxas mis-assembles `mov.b16 {b,_}` extraction of the `.b8` operand for `cvt.rn.f16x2.e2m1x2`** (wrong FP4 byte, deterministic wrong results); regenerated with 32-bit `mov.b32 {b,_,_,_}` (3712 sites). CUDA 13.x is fine.

## flashinfer#5959 / #6004 - Cake DSv4 NVFP4 sparse-MLA prefill
- #5959 (2026-10-02): register-MMA only streaming prefill (`mma.sync` mxf4nvf4 QK / f8f6f4 PV, `cp.async` gather, fp32 online softmax), one CTA per token tile and head-tile group (1/2/4 16-head tiles share a gather), direct epilogue, no merge. https://github.com/flashinfer-ai/flashinfer/pull/5959
- #6004 (2026-10-03), bitwise identical schedule changes: IO warps dequantize V (E2M1 -> E4M3) for 4-tile kernels while math warps run QK; 8 math warps split into two independent 4-warp groups so one group's softmax overlaps the other's MMAs; third cp.async ring stage for 1-tile CTAs with long candidate lists; L2 evict-first on BF16 Q loads from 512 tokens.

## flashinfer#5983 - DSv4.1 mixed-cache sparse-MLA decode (SM120/SM121)
- Merged 2026-10-03. https://github.com/flashinfer-ai/flashinfer/pull/5983
- Reads 528 B/token FP8 SWA cache (512 E4M3 + 16 UE8M0 group-32 footer scales) and 288 B/token FP4 compressed cache (256 B E2M1 + 32 B E4M3 group-16 scales) in place; BF16 Q; QK in BF16 `mma.sync m16n8k16` with exact on-chip dequant (E4M3 x 2^e and E2M1 x E4M3 are exact in BF16).

## flashinfer#6180 / #6199 / #6219 / #6530 - SM120 paged FP8 MQA lightning-indexer logits
- #6180 merged 2026-10-07. https://github.com/flashinfer-ai/flashinfer/pull/6180
- Exact reproduction of DeepGEMM's `sm120_fp8_paged_mqa_logits.cuh`: **384 threads = 8 math warps in 64-row groups + 4 TMA warps**; `mma.sync.m16n8k32` E4M3 -> FP32 with **KV as the A operand via 128B-swizzled `ldmatrix`**, Q as B; per-group 3-stage KV ring of 64-row tiles with FP32 scales, 2-stage Q/weights pipe; **`setmaxnreg` 232/40**; persistent per-SM schedule; weighted ReLU + quad-shuffle reduction; PDL between metadata and logits launches. Bitwise identical to DeepGEMM.
- Programs: 32 heads x page {128,64} x next_n {1,2,4} (DSv4.1-Flash) and 64 heads x page 64 (DSv3.2); later PRs add next_n 3/5/6 and batch-aware Q-atom programs.

## Serving-framework integration
- TensorRT-LLM#16224 (2026-08-24): `FlashInferSparseMlaFmha` in the FMHA registry for DSv4 and DSA on SM120 (`fp8_ds_mla`, DSv4 584-byte footer-scale tokens). https://github.com/NVIDIA/TensorRT-LLM/pull/16224 ; #20062 fixes `_latent_append` stride lookup.
- vllm#43477 (2026-06-22): DSv4 / GLM-5.1 on SM120 via FlashInfer sparse MLA + DeepGEMM MXFP4 MoE. https://github.com/vllm-project/vllm/pull/43477
- vllm#51395: SM120 sparse backend inherited `supports_dense_mha_prefill = True` but has no dense path -> long prompts crashed; now False. vllm#53574: width-narrowed slices of a persistent topk buffer are non-contiguous; FlashInfer paged path (`num_tokens > 64`) checks `eidx.IsContiguous()`.
- sglang#29927 (2026-09-02): DSv4-Flash on 4x RTX PRO 6000: indexer to DeepGEMM paged-MQA-logits (eager torch fallback dominated decode), batched sparse prefill (q_len > 64) via FlashInfer, indexer metadata chunked at M=4096, page-split only of pages referenced by this step's top-k (was ~105 MB/layer per step); decode latency up to 3.4x lower. https://github.com/sgl-project/sglang/pull/29927
- sglang#36655: stop padding TP4 decode queries from 16 to 64 heads (4x wasted work) when FlashInfer supports the native width.
- sglang#32779 (2026-10-06): fused Triton sparse-MLA prefill, one program per query token with online softmax in registers; FlashMLA requires heads % 64 (SM90) / % 128 (SM100+), so GLM-5.x TP8 (8 heads) wasted 7/8-15/16 of MMA on padding.

## KAI Notes
- Never reuse a named barrier id for an asymmetric arrive/sync handshake across consecutive tiles; alternate ids by tile parity or use mbarriers with explicit phases.
- Every thread that waits on an mbarrier parity must also participate in each phase; retire idle producer lanes (`return`) before the pipeline loop.
- Sparse-MLA on SM120: 8 math warps + 2-4 gather/TMA warps per CTA, cp.async/TMA ring of 64-candidate chunks, NVFP4 QK (`m16n8k64 mxf4nvf4`) + FP8 PV (`m16n8k32 f8f6f4`), split-K + batched LSE merge.
- Use `setmaxnreg` (e.g. 232 math / 40 producer) in warp-specialized SM120 kernels to move registers to MMA warps.
- Calibrate split/cpb with DRAM-resident (L2-cold) data and treat GB10 L2 as ~12 MiB effective; plan split counts by SM count (48 vs 170 vs 188).
- For E2M1 unpack via `cvt.rn.f16x2.e2m1x2`, extract the byte with `mov.b32 {b,_,_,_}`; CUDA 12.9 ptxas miscompiles the `mov.b16` form.
