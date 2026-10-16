---
id: blog-sm120-attention-kernels-2026h2
title: "SM120/SM121 dense attention kernels: FA4 CuTe, FlashInfer fmha_v2/XQA/NVFP4, TRT-LLM, vLLM, SGLang (May-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/NVIDIA/TensorRT-LLM/pull/15163
source_category: community-note
architectures: [sm120]
tags: [attention, flash-attention, decode, prefill, tma, mbarrier, warp-specialization, persistent-kernel, nvfp4, fp8, fp4, block-scale, cute-dsl, cuda-cpp, triton, sparse-attention, pipeline-stages, pdl]
retrieved_at: 2026-10-15
---

# SM120/SM121 attention kernels (2026 H2 digest)

SM120 has TMA (`cp.async.bulk.tensor`) and mbarriers but **no `wgmma`, no tcgen05/TMEM**: all MMA is
warp-synchronous `mma.sync` (BF16 HMMA m16n8k16, FP8 `m16n8k32 kind::f8f6f4`, FP4
`m16n8k64 kind::mxf4nvf4.block_scale`). The winning attention design therefore is "Hopper-style
producer/consumer with TMA + mbarrier, but consumer warps on mma.sync", plus FA3-style ping-pong
between consumer warp groups. Low-precision (FP8/NVFP4) attention gives ~1.8-2.4x over BF16 because
BF16 SDPA already reaches ~95% of BF16 mma.sync peak (sglang#40175 measured on RTX 5080).

## TensorRT-LLM#15163 - skip-softmax: TMA-load + sync-MMA warp-specialized context FMHA for sm_120/121
- Merged 2026-06-15. https://github.com/NVIDIA/TensorRT-LLM/pull/15163
- "Only half of the Hopper warp-specialization recipe ports": TMA async loads survive, async MMA does not. One **producer warp** issues `cp.async.bulk.tensor` for Q/K/V into granular smem buffers (driver-API `CUtensorMap`), consumer warps run BMM1 + softmax (+ per-warp skip-softmax) + BMM2 on `mma.sync`, with an **mbarrier producer/consumer handshake**.
- Files: `cpp/kernels/fmha_v2/src/fmha/warpspec_sm120/` (`kernel_traits.h`, `dma_sync_mma.h`, `compute_sync_mma.h`, README). BF16, `head_dim == head_dim_v` in {128, 256}, causal, packed QKV. Opt-in via `SkipSoftmaxAttentionConfig`.
- Pitfall fixed: producer must add per-request `cu_q_seqlens[bidb]` to the TMA sequence coordinates; without it every request in a batched prefill returned request 0's result.

## TensorRT-LLM#19825 - tiled FP8 context FMHA for head_dim 256 on SM120
- Merged 2026-10-08. https://github.com/NVIDIA/TensorRT-LLM/pull/19825
- Adds a tiled `64x128` QMMA fmha_v2 kernel for FP8 E4M3 head_dim 256 (beside the non-tiled `64x32`); forced non-tiled FP8 selection restricted to SM89.
- RTX PRO 6000: tiled faster on all 35 shapes, **2.03x-3.52x, geomean 3.01x** (KV 65 -> 32k: 2.15x -> 3.32x; chunked prefill 3.38-3.43x).

## flashinfer#3016 / #3518 / #4661 - fmha_v2 (TRT-LLM HMMA FMHA) for SM120
- #3016 (2026-07-17): enumerate HMMA (`sm_mma=80`) fmha_v2 kernels for SM120 bf16/fp16, head_dim 64/128/256; auto-used by `BatchPrefillWithRaggedKVCacheWrapper` for SM12x + MHA + seqlen >= 256 + no custom mask; GQA or short seq -> fa2. GB10: B=2 S=512 H=4 D=64 0.050 ms vs FA2 0.073 ms. https://github.com/flashinfer-ai/flashinfer/pull/3016
- #3518 (2026-06-09): head_dim 512 spec for SM120; `ldgsts_q = True` for head_size > 256 (tiled kernel static_asserts `USE_LDGSTS`); fix `grid.x *= ctas_per_o_row` in `_nl_tiled` launcher (previously only half the head dim was written); add `MASK_VERSION == 4` sliding/chunked-causal traits (previously sliding requests **silently ran the maskless kernel**). https://github.com/flashinfer-ai/flashinfer/pull/3518
- #4661 (2026-09-09): enables fmha_v2 prefill on SM121.

## flashinfer#3576 - head_dim 512 fa2 prefill/decode for Gemma 4 on SM120/121
- Merged 2026-06-15. https://github.com/flashinfer-ai/flashinfer/pull/3576
- BF16 Q, BF16 or FP8 E4M3 KV, page 16, 32/8 heads, s_kv 8192 causal. Prefill s_qo 8192: RTX PRO 6000 280.8 / 213.3 / 156.7 TFLOP/s for D=128/256/512; GB10 66.2 / 45.7 / 34.6 TFLOP/s. Decode q=1 D=128: 0.48 TB/s (PRO 6000), 117 GB/s (Spark).

## flashinfer#3684 / #4747 - asymmetric VO-split NVFP4 paged prefill (qk=512, vo=256)
- #3684 merged 2026-08-13. https://github.com/flashinfer-ai/flashinfer/pull/3684
- Gemma-4 global layers: head_dim_qk 512, head_dim_vo 256. Adds FP4 V-load into the VO-split FA2 path, an `[all-data | all-SF]` contiguous scale-factor layout per page (the trtllm-gen per-page swizzle does not commute with head-dim slicing), asymmetric K/V byte/scale offsets, `CTA_TILE_Q` keyed on `head_dim_qk` (otherwise infeasible tile, `max_mma_kv: 0`). RTX 5090 max_abs_err 0.0047.
- #4747 (2026-09-02): the NVFP4 "disable split-KV" corruption workaround is scoped to SM120/121 only; it was costing low-batch decode elsewhere.

## flashinfer#3897 - SM121 enable for the SM120 NVFP4 attention kernel
- Merged 2026-07-13. https://github.com/flashinfer-ai/flashinfer/pull/3897
- Kernel unchanged; only gates widened. GB10 vs FA2 BF16 ragged prefill (H8, non-causal): B4 S4096 D128 2.97 -> 1.28 ms (93 -> 215 TF/s, 2.32x); B1 S32768 D128 46.1 -> 19.5 ms (2.36x); D64 only 1.55x.

## flashinfer#4570 / #4137 - XQA decode on SM12x: head_dim 512 and ragged spec-decode
- #4570 (2026-08-18): XQA (default SM12x decode backend) now accepts non-MLA head_dim 512. Fixed latent V page-advance bug when `cacheVTileSeqStride % tokensPerPage != 0` (`exactDiv` gave zero step under NDEBUG) - needed for fp16 KV 512 with 16-token V tiles in 32-token pages. RTX PRO 6000, 4 KV heads, group 4, b64 seq 4096: fp8 KV 741 us; fp16 KV 1505 us vs FA2 1417 us. JIT-only (kept out of AOT). https://github.com/flashinfer-ai/flashinfer/pull/4570
- #4137 (2026-07-28): `q_cu_seq_lens` exposed for ragged draft counts; sliding window computed **per draft row** (previously from the last draft token, wrongly masking KV earlier drafts need). https://github.com/flashinfer-ai/flashinfer/pull/4137

## TensorRT-LLM#17976 - XQA multi-block FP8 KV on SM120 (see MoE digest for model context)
- https://github.com/NVIDIA/TensorRT-LLM/pull/17976 - V-cache page cursor advanced once per `xIter` instead of per V tile at head_dim 256; output race (only merge was gated on `isLastCta`). ITL BS1 5.53 -> 3.74 ms on RTX PRO 6000.

## FlashAttention (Dao-AILab) CuTe-DSL FA4 on SM120
- flash-attention#2656 (2026-07-07): **Pack-GQA** fully wired on SM120 (previously only the epilogue was packed -> crash). `mQ/mO` reshaped via `pack_gqa_layout` folding `qhead_per_kvhead` into the seqlen mode; scheduler uses packed total rows and logical `seqlen_q_static`; Q-load gathers rows via `PackGQA.load_Q`; num_head indexes the KV head. SplitKV is SM100-only: SM120 falls back to `num_splits=1`. Packed vs unpacked bit-identical on RTX PRO 6000. https://github.com/Dao-AILab/flash-attention/pull/2656
- flash-attention#2671 (2026-07-07): cast optional `window_size_left/right` to `cutlass.Int32` for `cute.compile`; init `dQ_single_wg` for SM120 backward; keep `softmax_scale` when `score_mod is None`.
- flash-attention#2763 (2026-09-08): guard SM80-derived forward body (inherited by SM120) with `work_tile.is_valid_tile`: `SingleTileVarlenScheduler` launches padding tiles with `batch_idx == num_batch` -> OOB `cu_seqlens` read, CUDA error 700 on RTX PRO 6000.
- flash-attention#2775 (2026-09-15): PackGQA predicates for head dims 72/88/104/120/136 (8-byte aligned, not multiple of 16): predicate fragment must match the sliced copy operand's vector layout. RTX 5090 D=72 PackGQA ~1.00-1.05x.
- flash-attention#2758 (2026-10-13): `num_splits` 0/-1 (auto) maps to 1 on SM120; explicit >1 still asserts.

## sglang#32991 - architecture-owned SM12x FA4 kernels
- Merged 2026-08-13. https://github.com/sgl-project/sglang/pull/32991
- Replaces the SM80-derived SM120 path with TMA-based forward and paged-decode kernels on warp MMA, persistent scheduling with **runtime-SM-count-aware tile selection**, paged KV, SplitKV, sinks, packed GQA, relative/shared bias, CUDA-graph-safe decode plans, PDL. 53 tests.

## Low-precision (FP8 / NVFP4) prefill attention on SM120
- flashinfer#5539 (2026-09-25) FP8 packed-varlen (MiniMax-H3, [T,56,128]): 4 launches - K/V stats, finalize (K channel mean, V amax/448), FP8 quant (Q per token, K per 128-key block after mean subtraction = exact softmax shift, V per (segment, channel) into transposed permuted V^T), attention: **persistent 256-thread CTA/SM, 8 warps x 16 rows, 128-key tiles via 2-stage TMA ring, FA3 ping-pong between 2 warp groups on named barriers, `mma.sync.m16n8k32 kind::f8f6f4` for QK^T and PV**, E4M3 probabilities with 2^8 exponent bias, LPT work placement. 1.82-1.87x (PRO 6000) / 1.93-1.96x (5090) vs fastest FlashInfer BF16; rel-L2 0.051. https://github.com/flashinfer-ai/flashinfer/pull/5539
- flashinfer#5545 (2026-09-27) NVFP4 SageAttention3 recipe: K channel-mean removal, per-128-row Q block-mean removal with the `qm K^T` correction computed in-kernel (one block-scaled MMA row per 128-key tile) instead of Sage3's dense FP32 `delta_s` (whose TMA descriptor overflows above 2^31 elements); Q/K/V E2M1 + UE4M3 per 16; QK and PV on `mma.sync.m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X`; row max brought to 448*6; two-level P quantization; 3 PDL-chained launches. Rel-L2 0.186-0.193. https://github.com/flashinfer-ai/flashinfer/pull/5545
- flashinfer#5583: block-max-first softmax over per-16-key raw block maxima, **4 independent partial row-sum chains**: +2-4.6%.
- flashinfer#6060: warp-specialized form, **8 QK/softmax warps + 4 PV warps (384 threads), 1 CTA/SM**, selected when mean segment >= 1536 tokens and SMs >= 188.
- flashinfer#6068: E2M1 P codes computed block-relative straight from `exp2((s - blockmax)*c + log2 6)`; WS kernel window 1536 <= mean segment < 16384.
- sglang#40175 (2026-09-22) `fp8_fa_sm120` CuTe-DSL: E4M3 QK/PV, 128x32 CTA tile, two-stage K/V prefetch, probabilities packed to E4M3 in registers (no smem round trip); Triton fused per-head quant; ~10 s compile per new S. https://github.com/sgl-project/sglang/pull/40175

## Block-sparse attention on SM120
- flashinfer#4259 (2026-08-14): `vsa_sm120_blk64` backend in `BlockSparseAttentionWrapper` (fp16/bf16, GQA, block mask or BSR, LSE).
- flashinfer#4691 (2026-09-17): Sage block-sparse: **custom INT8 warp MMA for QK, FP8 warp MMA for PV**, log2-domain online softmax; quant: per-32-token-group Q, mean-centred per-K64-tile K, per-channel V with a 16-token physical permutation matching the PV MMA operand layout (PRMT). MHA only, no LSE.

## vLLM SM12x attention backend selection
- vllm#49718 (2026-08-11): FlashInfer XQA decode on SM12x (uniform/ragged spec-decode, sliding window, sinks, CUDA graphs); FlashInfer 0.6.16.post1. https://github.com/vllm-project/vllm/pull/49718
- vllm#52148 (2026-08-13): generic FA2 prefill accepted `sinks` but **ignored them**; use the sink-aware paged prefill wrapper on SM12x (GPT-OSS fixed).
- vllm#46963 (2026-10-07): `--kv-cache-dtype nvfp4` on SM8x/SM12x via `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` + native fa2 read; head_size 64 refused (fa2 prefill misreads NVFP4 by 10-15% with the 128-row query tile). KV capacity 3.32x (sm_80 host).
- vllm#60452 (2026-10-10): NVFP4 KV decodes with XQA on SM12x like FP8, restoring `UNIFORM_BATCH` graphs for MTP verification (was ~15 tok/s at TP=2 MTP3).
- vllm#52017 (2026-09-01): `--attention-backend B12X_ATTN` causal paged attention (BF16/FP8 KV, sinks, sliding window, spec verify).
- vllm#58482 (2026-10-01): Triton unified attention split-K uses 64 segments instead of 16 on SM120 when `padded num_seqs x KV heads x 16 < SM count`; Llama-3.1-8B 32k prompt c1 1.050x tok/s.

## KAI Notes
- On SM120 build attention as TMA producer warp + mbarrier full/empty ring + mma.sync consumer warps (no wgmma); use FA3-style ping-pong between two consumer warp groups via named barriers.
- For batched/varlen TMA loads, add per-request `cu_seqlens` offsets to the TMA coordinates and guard padding tiles (`is_valid_tile`), or you read request 0 / OOB.
- BF16 attention is near mma.sync peak on SM120; go to FP8 (`m16n8k32 f8f6f4`) for ~1.8-1.9x or NVFP4 (`m16n8k64 mxf4nvf4 scale_vec::4X`) for ~2.3x, with K mean-centering for accuracy.
- Low-batch decode on 188-SM PRO 6000 with few KV heads under-fills the GPU: use split-KV / multi-block (more segments) when `batch x kv_heads x splits < SMs`.
- Check that masks (sliding window, sinks) are actually applied in every selected kernel path; several SM120 paths silently ignored them.
- head_dim 512 kernels need tiled O (multiple CTAs per O row) and grid.x scaled accordingly.
