---
id: blog-sm120-moe-kernels-2026h2
title: "SM120/SM121 fused-MoE kernels in FlashInfer b12x, vLLM and TensorRT-LLM (May-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/flashinfer-ai/flashinfer/pull/4285
source_category: community-note
architectures: [sm120]
tags: [moe, grouped-gemm, nvfp4, fp4, fp8, block-scale, cute-dsl, fused-kernel, persistent-kernel, decode, prefill, quantization, tile-scheduling]
retrieved_at: 2026-10-15
---

# SM120/SM121 fused-MoE kernels (2026 H2 digest)

Consumer/workstation Blackwell (SM120: RTX 5080/5090, RTX PRO 5000/6000; SM121: GB10 DGX Spark)
has no tcgen05/TMEM, ~99 KiB opt-in shared memory per block, and warp-level block-scaled MMA
(`mma.sync ... kind::mxf4nvf4 / mxf8f6f4`, e.g. `m16n8k32` E2M1 x E4M3 with UE8M0 scales). The
dominant open-source MoE stack for it is **b12x** (CuTe-DSL kernels from local-inference-lab,
vendored into FlashInfer as `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/`, later moved under
`flashinfer.experimental`). Entry points: `b12x_fused_moe(...)` and `B12xMoEWrapper`.

Kernel family after the Aug 2026 sync (flashinfer#4285):
- `MoEDirectMicroKernel` - smallest decode batches (1-2 tokens); reads top-k ids directly and skips the routing pre-pass.
- `MoEMicroKernel`, `MoEStaticKernel` - rest of decode; tile assignment fixed at launch.
- `MoEDynamicKernel` - prefill; tiles distributed by an in-kernel work queue.
- All fused: route/pack + input quant + FC1 + gated activation + intermediate quant + FC2 + weighted scatter in one kernel (no standalone sort kernel). Launched **cooperatively** (grid-wide barriers between phases).

## flashinfer#3271 - feat(moe): add SM120 W4A16 b12x kernels
- Merged 2026-05-11 (just before the KB cutoff; included for context). https://github.com/flashinfer-ai/flashinfer/pull/3271
- Adds `activation_precision` (`"fp4"` = W4A4, `"bf16"` = W4A16) to the b12x MoE dispatch; wires W4A16 static, dynamic and direct-micro kernels.
- `fc2_input_scale` required only for fp4 activations, ignored for bf16.
- Added CuTe-DSL primitives: GPU sync/atomics, non-coherent global loads, relaxed shared-memory ops, FP4/FP8/BF16 quant/dequant/pack and vectorized scatter helpers.
- Pitfall: direct W4A16 micro path only valid when local expert ids == global ids; EP remapping falls back to compact static path.

## flashinfer#3336 - Replace SM120 W4A16 MoE kernels
- Merged 2026-05-22. https://github.com/flashinfer-ai/flashinfer/pull/3336
- Replaces the static/dynamic/micro W4A16 split with a **packed-route design**: routed token rows are grouped by expert *before* GEMM work; one fused FC1/activation/FC2 kernel is driven by packed route indices, per-block expert ids and a route count.
- New `moe_w4a16_*` modules: prepared weights, route packing, FP4 helpers, host utilities, fused kernel. Adds `quant_mode` and `source_format` inputs; quant_mode-aware workspace allocation.
- CUDA-graph capture now requires preallocated outputs; incompatible quant_mode/source_format combos rejected.
- Related: flashinfer#3302 (2026-05-21) rejects EP configs (`num_local_experts != num_experts`) with a clear `NotImplementedError` instead of illegal-address crashes deep in the CuTe-DSL kernel.

## flashinfer#3744 - gelu_tanh and swiglu_oai activations for b12x NVFP4 MoE
- Merged 2026-07-01. https://github.com/flashinfer-ai/flashinfer/pull/3744
- `gelu_tanh` (GeGLU, Gemma-4) and `swigluoai_uninterleave`: `gate*sigmoid(alpha*gate)*(up+beta)` with optional clamp; defaults alpha=1.702, beta=1.0 (MiniMax-M3).
- Transparent zero-padding of non-128-aligned `intermediate_size` (Gemma-4 I=704) up to the gate/up tile width; padded weights cached.

## flashinfer#3932 - correct FP4 quantization numerics, add input_global_scale
- Merged 2026-07-31. https://github.com/flashinfer-ai/flashinfer/pull/3932
- Symptom: b12x W4A4 on GB10 produced many more reasoning tokens than Marlin at equal scores.
- Bug 1: the software decode of **subnormal E4M3 block scales** used a wrong formula (up to 4.5x off) while the tensor core decoded the byte correctly. Fix: decode with the hardware cvt (exact incl. subnormals and cheaper).
- Bug 2: precise path (`fast_math=False`) inverted the pack multiplier and quantized everything to zero.
- API: `w1_alpha` was doing two jobs (input-quant scale and output multiplier); new optional `input_global_scale` decouples them, folded internally into the output multiplier and cached (no per-call alloc during graph capture).

## flashinfer#4255 - sync SM12x W4A16 fused MoE family to b12x HEAD
- Merged 2026-08-03. https://github.com/flashinfer-ai/flashinfer/pull/4255
- Cooperative persistent launches for fused FC1/FC2 with occupancy-aware tile selection and launch bounds.
- Tensor-core decode path for small batches on packed serving weights; top-k sum folded into the FC2 store epilogue (one launch fewer). FlashInfer caps this path at m<=4 (upstream m<=8) because the crossover differs per card.
- Shape-stable route packing: Triton packing kernels specialize on a **power-of-two capacity** instead of exact token count, so changing decode batch no longer recompiles.
- New `fp4_e8m0_k32` (MXFP4 K/32) weight source format with scale-tail handling for non-128-aligned TP shards.
- Perf (RTX 5080 / RTX PRO 6000 SE / GB10, Nemotron MoE shape): m=1 relu2 1.07x/1.15x/1.11x; m=2048 12.91x/18.52x/1.00x. The m=2048 wins are a **stall fix**: the kernel used grid-wide barriers but was not launched cooperatively, so non-co-resident CTAs spun forever-ish. GB10 kept the whole grid resident and never stalled.

## flashinfer#4285 - sync SM12x NVFP4 fused-MoE kernels to b12x HEAD
- Merged 2026-08-03. https://github.com/flashinfer-ai/flashinfer/pull/4285
- Adopts the real `MoEDirectMicroKernel` (previous copy was never routed to); adds gelu_tanh to it.
- `MoEDynamicKernel` may use tiles **smaller than 128 rows**, so sparse routing no longer pads every expert to 128.
- All kernels launched cooperatively (grid sync deadlocks if concurrent work keeps part of the grid off the GPU).
- Pipeline stages sized from real smem footprint; vectorized scatter epilogue in `MoEStaticKernel`.
- Fast FP4 quantizer restricted to gated activations: relu2 squared outputs need the exact quantizer.
- Geomean speedups over 38 MoE shapes (DGX Spark / RTX 5080 / RTX Pro 6000 SE): decode bs1-2 1.06x/1.30x/1.40x (up to 2.3x per shape); prefill bs512-2048 1.10x/1.10x/1.09x (up to 1.26x).
- Pitfall: upstream kernel takes scale multipliers, FlashInfer API takes reciprocal-form scales, so dispatch inverts them; workspace cache is keyed by tile size.

## flashinfer#4290 - native MXFP4 W4A4 fused MoE
- Merged 2026-08-10. https://github.com/flashinfer-ai/flashinfer/pull/4290
- `quant_mode="mxfp4"`: E2M1 data, block-32 UE8M0 scales, `MmaMXF4Op`. Works on micro/static/dynamic schedules and CUDA graphs. MoE K tile stays 128 to keep the FC1-epilogue / FC2-K-tile coupling.
- RTX PRO 6000 SE, H=7168, I=2048, E=256, top-8, full fused MoE: T=1 NVFP4 0.2090 ms vs MXFP4 0.2022 ms; T=8 1.1175 vs 1.0557; T=512 4.7641 vs 4.5523; T=2048 5.6317 vs 5.5596 (MXFP4 1.01-1.06x faster).
- Accuracy: end-to-end MoE rel-L2 vs BF16 NVFP4 0.2306, MXFP4 0.2923 (MXFP4 coarser scales).

## flashinfer#4329 - optimize gated SM12x dynamic NVFP4 MoE
- Merged 2026-08-07. https://github.com/flashinfer-ai/flashinfer/pull/4329
- Splits `_moe_dynamic/generic.py` and `_moe_dynamic/gated.py`. Gated FC1 (silu, gelu_tanh, swiglu-oai) **pairs gate and up projections in the N64 FC1 dataflow** to shorten accumulator lifetimes and staging; FC2 stays on the non-gated path. Keeps a tuned inline-PTX reciprocal for SiLU.
- RTX PRO 5000 72GB (SM120), CuTeDSL 4.6.1, CUDA graph, 192 MiB L2 flush: Qwen3.5-35B TP1 M=8192 1749 -> 928 us (-47%); Qwen3.5-122B TP2 M=8192 2518 -> 1441 us (-43%); Qwen3.5-397B TP8 M=8192 2331 -> 1857 us (-20%). M<96 uses static/micro, M>=96 dynamic.

## flashinfer#4718 - optimize and unify static MoE
- Merged 2026-08-31. https://github.com/flashinfer-ai/flashinfer/pull/4718
- Oversized routed experts split into **32-row virtual tasks** (skewed experts could previously exceed the tile's computed M extent).
- Retains two FC1 N-slices per scheduled work group; compact O(1)-claim scheduler; NVFP4 static cutover raised 640 -> 1024 routed rows (M<=128 at top-8).
- Qwen3.5-35B TP1 (E=256, H=2048, I=512, top-8), Zipf routing, SM120: M=8 112 -> 96 us (-14.5%), M=32 290 -> 229 us (-21%), M=64 383 -> 325 us (-15%), M=128 451 -> 415 us.

## flashinfer#4302 - expert parallelism for SM12x W4A16 fused MoE
- Merged 2026-09-29. https://github.com/flashinfer-ai/flashinfer/pull/4302
- Every rank gets the full batch, computes only its experts (`expert_map`), caller sums partial outputs; no token exchange. Verified on 2x RTX Pro 6000. NVFP4 path still rejects EP (indexes per-expert buffers by global top-k id).

## flashinfer#5451 / #5977 - skip unrouted (-1) expert ids
- Merged 2026-09-30 / 2026-10-05. https://github.com/flashinfer-ai/flashinfer/pull/5451
- vLLM writes -1 in CUDA-graph padding rows of `topk_ids`. b12x indexed histogram / `expert_write_rows` / `expert_tile_base` / static allocator / micro weight reads with it -> illegal address or silent corruption on SM120.
- Fix: negative id = unrouted pair (no packed row, no compact expert, finalizer ignores slot; all-negative token -> exact-zero row). Micro kernels keep dense layout, read expert 0 with zero routing weight. #5977 applies the same to the fp8 / mxfp8_mxfp4 grouped route.

## flashinfer#5168 - perf(sm12x): optimize fused MoE kernels
- Merged 2026-10-13. https://github.com/flashinfer-ai/flashinfer/pull/5168
- Static: **true-extent up/gate/down weight views** (branch-major) so TMA zero-fills OOB instead of storing padded FP4 channels (I=320 used to be materialized at 384/512); weight prefetch during routing; merge intermediate slices.
- Dynamic: parallel expert-tile prefix scan; skip invalid tail MMA and empty stages; choose A16/A32/A64/A128 activation transfers from valid row count; L2 evict policy for streamed weights.
- E=512, H=2560, I=320, top-10, SM120, CuTe DSL 4.7: geomean latency -27.7% over M=1..8192. M=64 744 -> 480 us; M=1024 993 -> 716 us; M=8192 1729 -> 1317 us.

## flashinfer#5242 / #5735 - native/canonical-layout NVFP4 W4A16 in CuTe DSL
- #5242 merged 2026-09-23 (dense, `mm_bf16_fp4(..., backend="cute-dsl-native")`); #5735 merged 2026-10-02 (MoE, `SM12xNvfp4Bf16Config`). https://github.com/flashinfer-ai/flashinfer/pull/5735
- Key idea: W4A16 reads the **same packed E2M1 + 128x4-swizzled E4M3 scales** as W4A4, no second weight copy, so a server can pick W4A16 (decode) or W4A4 (prefill) per call.
- Weights decoded in registers from 128-bit loads; split-K when grid too small; for M>16 the staged kernel reuses decoded weights across 64/128 activation rows.
- #5242, RTX PRO 6000 Max-Q (300 W), Qwen3.8-27B down proj (N=5120, K=17408) M=1: 38.8 us; 1.348x vs CuTe W4A4, 1.253x vs Marlin; vLLM TPOT -6.6 to -7.2%.
- #5735, DGX Spark, Qwen3.8-Flash-Next MoE (H=2560, I=640, 512 experts, top-10): 1 token 159.2 us = 1.36x vs Marlin, 1.24x vs old CuTe W4A16, 2.06x vs cuTile; parity at 64 tokens; at 1024 tokens 0.41x vs Marlin (MoELayer autotune picks cuTile there). Small M = one persistent kernel with work-stealing; large M = expert-sorted per-expert tiled GEMMs.

## flashinfer#4720 - SM12x fp8 + mxfp8_mxfp4 grouped-MoE kernels
- Merged 2026-09-22. https://github.com/flashinfer-ai/flashinfer/pull/4720
- 8 CuTe-DSL ops: `cute_dsl_sm12x_moe_gemm_{fp8,mxfp8_mxfp4}`, `..._fc1_act_*` (gate-up GEMM + SwiGLU/SiTU-GLU), `..._fc1_act_q1_*` (+ fused fp8/mxfp8 requant for FC2), `..._fc2_finalize_*` (down GEMM + routing-weighted finalize). Benchmarked vs DeepGEMM on DSv4 shapes (figures only).

## flashinfer#4008 - prune SM120 fused-MoE tile candidates that can never run
- Merged 2026-09-16. https://github.com/flashinfer-ai/flashinfer/pull/4008
- Removed grouped tiles `128x128x256B` and `256x128x128B` (exceed SM120 smem). `...x64B` tiles removed for FP8 activations (break K alignment of mixed FP8xFP4 load); kept for FP4 activations.

## flashinfer#4387 / #4632 / #6346 - SM120 MegaMoE (fused dispatch + grouped GEMM + combine)
- #4387 (2026-09-02): MXFP8 swap-AB CuTeDSL megakernel `sm120_mxfp8_mxfp8_bf16_cutedsl`; K-major + interleave-8 weight preprocessing; single-GPU rank-sharing via gloo for testing. https://github.com/flashinfer-ai/flashinfer/pull/4387
- #4632 (2026-10-09): W4A8 = MXFP4 E2M1 weights x MXFP8 E4M3 activations, E8M0 K32 scales, `m16n8k32` mixed block-scaled QMMA, packed-E2M1 TMA/LDSM path; FC1 output re-quantized to MXFP8. 4x RTX Pro 5000, DSV4-Flash (top-6, H=4096, E=256): 16 tok/rank 741 us; 1024 tok/rank 1869 us (165 TFLOP/s); 16384 tok/rank 21.3 ms (232 TFLOP/s).
- #6346 (2026-10-13): Cake-generated `sm120_mxfp4_mxfp8_bf16_cake`: dispatch, FC1+SwiGLU+FP8 requant, FC2, return and combine in **one cooperative persistent kernel issuing NCCL GIN puts** (NCCL 2.30.7 Device API) for multi-host EP; LSA (peer load/store via copier CTAs) intra-host. Prefill-only scope (>=512 tok/rank). Requires CUDA 13.

## vllm#40082 - integrate FlashInfer b12x MoE and FP4 GEMM for SM120/121
- Merged 2026-05-20. https://github.com/vllm-project/vllm/pull/40082
- `FlashInferCuteDSLSM12xExperts` (MoE backend `flashinfer_cutedsl_sm12x`), auto-selected by the NVFP4 oracle on SM120 family; BF16 hidden states passed directly (activation quant fused).
- `a2_gscale` reset to 1.0: SM12x uses dynamic per-block quantization for FC2 input; calibrated ModelOpt value causes saturation.
- Dense `FlashInferB12xNvFp4LinearKernel` (`VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x`). DGX Spark Qwen3-30B-A3B-NVFP4: +1.8% (1 prompt), +6.0% (8 prompts) vs flashinfer-cutlass.
- Pitfall (at the time): `nvidia-cutlass-dsl` 4.5.0 generated bad PTX for SM121; needed 4.4.2 and a patch so `warp/mma.py` accepts `sm_121a`.

## vllm#52016 / #52018 - standalone b12x dense linear and FP4 MoE backends
- Merged 2026-08-14 / 2026-08-21. https://github.com/vllm-project/vllm/pull/52018
- `vllm[b12x]` optional dep (pure-Python CuTe DSL). Linear: per-tensor FP8, 128x128 block FP8, MXFP8, NVFP4, MXFP4 (`--linear-backend b12x`). MoE: `--moe-backend b12x`, NVFP4/MXFP4 W4A4, W4A16, dynamic W4A8 (MXFP4 defaults to MXFP8 activations); `VLLM_B12X_MOE_FP4_FORCE_A16=1`. No EP. Warmup covers all CUDA-graph sizes + `max_num_batched_tokens`.

## vllm#55170 - prefer W4A4 linear kernels over weight-only on SM120/121
- Merged 2026-09-08. https://github.com/vllm-project/vllm/pull/55170
- Kernel list walked in order; the CuTe-DSL W4A4 kernel is sm_10x-only, so SM120 fell through to the weight-only W4A16 kernel ahead of three working W4A4 kernels.
- Qwen3.8-27B-NVFP4, RTX PRO 6000 Max-Q, BS1, 32K prefix: TTFT 1455 -> 926 ms, 74.8 -> 92.6 tok/s (+23.8%) with flashinfer_cutlass W4A4. W4A4 loses BS1 decode by a few percent.

## vllm#54427 - weight-only NVFP4 checkpoints routed through W4A16
- Merged 2026-08-30. https://github.com/vllm-project/vllm/pull/54427
- ModelOpt checkpoints labelled NVFP4 with `input_activations: null` made W4A4 MoE fold an **uninitialized** `input_scale` into alphas. On GB10 the buffer read 0.0 -> all experts dead, garbage tokens; on RTX PRO 6000 it happened to read 1.0. Now promoted to `W4A16_NVFP4` (Marlin).

## vllm#58877 - enable OAI Triton MXFP4 MoE on SM12x
- Merged 2026-10-06. https://github.com/vllm-project/vllm/pull/58877
- RTX 5090, GPT-OSS: decode equal to Marlin, prefill ~10% slower (smem too small for larger Triton tiles), +8% KV-cache pool (Marlin needs alignment padding). Marlin stays default; `--moe-backend triton` opt-in.

## TensorRT-LLM#13773 - FlashInfer NVFP4 MoE backend (SM120/SM121)
- Merged 2026-05-25. https://github.com/NVIDIA/TensorRT-LLM/pull/13773
- Hybrid `FlashInferNvfp4Sm12xFusedMoE`: prefill (tokens >= 64) on CUTLASS NVFP4 grouped GEMM ("b12x's 12-CTA-per-token MMA pattern is suboptimal at large m"), decode (<64) on `B12xMoEWrapper.run`.
- RTX PRO 6000, Nemotron-Super-120B-NVFP4, ISL 2048/OSL 1024, conc 1: TPOT 13.97 -> 11.49 ms, throughput 70.6 -> 85.9 tok/s, TTFT unchanged (154.5 ms); tokens/W +20.4%. GSM8K 92.42 -> 92.27%.

## TensorRT-LLM#12704 - filter CUTLASS MoE tile configs by device smem on SM121
- Merged 2026-07-23. https://github.com/NVIDIA/TensorRT-LLM/pull/12704
- Runtime check of kernel `SharedStorage` against `cudaDevAttrMaxSharedMemoryPerBlockOptin` in `moe_gemm_tma_ws_launcher.inl`; candidate list in `get_candidate_configs_sm120()` filtered; mixed FP8xFP4 gets a smaller fallback tile for 99 KiB devices.

## TensorRT-LLM#16511, #17976, #18140, #18898 - W4A16 NVFP4 and SM120 MoE fixes
- #16511 (2026-08-07): W4A16 NVFP4 (packed FP4 weights + FP8 group scales, BF16 activations) for dense, LM head and MoE on SM120/121 via CUDA-core, Triton, Marlin, CUTLASS and CuteDSL backends. https://github.com/NVIDIA/TensorRT-LLM/pull/16511
- #17976 (2026-08-28): Qwen3.6-35B-A3B-NVFP4: XQA multi-block decode with FP8 KV on SM120; fixed V-cache page cursor advanced per `xIter` instead of per V tile at head_dim 256 (`cacheVTileSeqLen==32`); fixed output race (merge gated by `isLastCta` but write was not). LM head kept W4A16 (Marlin). RTX PRO 6000 600 W, ISL 8000: ITL BS1 5.53 -> 3.74 ms (-32%), BS16 -6.3% (2 KV heads -> only 2xB CTAs vs 188 SMs without multi-block).
- #18140 (2026-08-26): SM120 FP8 block-scale MoE autotuner constraint pads M to multiple of 4 on scale tensor dim 0.
- #18898 (2026-09-16): FlashInfer >= 0.6.17 auto-selects FC2 `tile_n=512/tile_k=32` for small-m W4A16 decode, then rejects it in `force_tile_config` (`tile_k>=64` floor); shim accepts that ultra-wide tile.

## KAI Notes
- On SM120 a fused MoE that uses grid-wide barriers MUST be launched cooperatively (or sized to co-residency); otherwise large-M runs stall (18x "speedup" in flashinfer#4255 was just this fix).
- Treat `topk_ids < 0` as "unrouted" in every MoE routing/histogram/scatter path; CUDA-graph padding rows from vLLM contain -1.
- Do not pad I to 128 in memory: use true-extent TMA views and let TMA zero-fill OOB (flashinfer#5168, -28% geomean); allow dynamic tiles smaller than 128 rows for sparse routing.
- Decode scale bytes (E4M3/UE8M0) with hardware cvt, not manual formulas; subnormal E4M3 scales are easy to get wrong (flashinfer#3932).
- Pick W4A16 for M<=~16 decode and W4A4/MXFP4 for prefill; share one packed weight layout so the switch is free (flashinfer#5242/#5735, vllm#55170).
- Filter tile candidates against `cudaDevAttrMaxSharedMemoryPerBlockOptin` (~99 KiB on SM12x); 128x128x256B and 256x128x128B grouped tiles do not fit.
