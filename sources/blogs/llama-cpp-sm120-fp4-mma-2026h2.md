---
id: blog-llama-cpp-sm120-fp4-mma-2026h2
title: "llama.cpp CUDA backend on SM120/SM121: native NVFP4/MXFP4 block-scaled MMQ, PDL and decode tuning (May-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/ggml-org/llama.cpp/pull/29857
source_category: community-note
architectures: [sm120]
tags: [nvfp4, fp4, block-scale, gemm, gemv, moe, quantization, pdl, ptx, cuda-cpp, ldmatrix, flash-attention, attention, decode, prefill, kernel-fusion, epilogue-fusion, tile-scheduling]
retrieved_at: 2026-10-15
---

# llama.cpp CUDA backend on Blackwell GeForce / RTX PRO / DGX Spark (2026 H2 digest)

llama.cpp's MMQ (quantized matmul) kernels have a native FP4 path on Blackwell (`BLACKWELL_MMA_AVAILABLE`,
i.e. SM120/SM121 builds) using warp-level block-scaled MMA. The current helper in
`ggml/src/ggml-cuda/mma.cuh` (read from master on 2026-10-15):

```cpp
// tile<16,8,float> D += A(tile<16,8,int>) * B(tile<8,8,int>), one 32-bit scale register per operand
if constexpr (type == GGML_TYPE_MXFP4) {
  asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};" ...);
} else { // NVFP4
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};" ...);
}
```
- MXFP4 (block 32, UE8M0) uses `kind::mxf4 ... scale_vec::2X`; NVFP4 (block 16, UE4M3) uses `kind::mxf4nvf4 ... scale_vec::4X`. A fragment = 4 x 32-bit regs (32 E2M1 values each lane pair), B = 2 regs, D = 4 fp32. The `{0,0}` operands are the byte-id / thread-id selectors for the scale registers.
- `block_fp4_mmq` has the same byte size as `block_q8_1_mmq` but holds twice the values (llama.cpp#25613).

## llama.cpp#29857 - CUDA: optimize accumulation in MMQ for NVFP4
- Merged 2026-10-05. https://github.com/ggml-org/llama.cpp/pull/29857
- Change (in `mmq-vec-dot.cuh`): instead of `tile_C C = {}; mma_block_scaled_fp4(C, ...); for l: sum[...] += C.x[l];` per fragment, reinterpret the output accumulator slice as `tile_C &` and let the MMA accumulate **in place** (`D` is both C and D operand).
- Nsight Compute over 12 matched NVFP4 launches: executed instructions 995M -> 769M, registers/thread 255 -> 232-236, **register-spill instructions 17.76M -> 0**, occupancy unchanged (16.5%).
- RTX PRO 6000 Blackwell, Qwen3.6-35B-A3B-NVFP4, ubatch 4096: NVFP4 MMQ time per 8192-token prompt 153.0 -> 142.7 ms (-6.7%), prompt throughput 14521 -> 14907 tok/s (+2.65%). Other models: Qwen3.6-27B NVFP4 -7.7%, Gemma-4-26B-A4B -10.3%, Nemotron-3-Nano-Omni -10.2% MMQ time. Perplexity slightly lower (less intermediate rounding).

## llama.cpp#25730 - improve NVFP4 W4A4 activation quantization
- Merged 2026-07-22. https://github.com/ggml-org/llama.cpp/pull/25730
- Quality: per-channel amax scaling as an alternative to pre-calibrated scales; scale search kept (removing it lost quality, costs ~5% E2E).
- Speed: fused kernel computing per-channel amax first; **256-bit loads (one thread = one 32-byte sector)** for coalescing without templating; hardware NVFP4 conversion intrinsics (dedicated cvt instructions); scale applied in the **MMQ epilogue** before the GMEM write (restricted to NVFP4 because the ternary hurt predicate efficiency for other quants).
- RTX PRO 6000 (B6000) pp8192: Qwen3.5-35B-A3B NVFP4 8246 -> 8634 tok/s (1.05x), Qwen3.5-27B NVFP4 1.04x, Nemotron-H MoE NVFP4 1.03x; DGX Spark Gemma-4 26B-A4B NVFP4 2778 -> 2940 (1.06x).

## llama.cpp#24481 - fuse MMVQ post-scale for NVFP4
- Merged 2026-07-07. https://github.com/ggml-org/llama.cpp/pull/24481
- Fuses the per-tensor weight scale `w_s` multiply into the MMVQ (decode GEMV) kernel, with the bias/scale ordering from #24331. Decode tg128@d32768 on RTX PRO 6000: Qwen3.5-35B-A3B NVFP4 193.4 -> 208.0 tok/s (1.08x), Gemma-4 NVFP4 1.02x.

## llama.cpp#24364 - force W4A8 path for NVFP4_W4A16 layers on Blackwell
- Merged 2026-09-25. https://github.com/ggml-org/llama.cpp/pull/24364
- GGUF metadata marks layers that were W4A16 in the HF checkpoint; hint `GGML_HINT_NO_QUANT_SRC1` dispatches them (dense `mul_mat` and MoE `mul_mat_id`) to a W4A8 path instead of native W4A4 on Blackwell, improving quality (W4A4 is hard for small LLMs).

## llama.cpp#25613 / #29953 / #29173 - buffer sizing and correctness
- #25613 (2026-07-15): native FP4 path sized the src1 quantization buffer using the Q8_1 layout -> 2x over-allocation in `MUL_MAT` and `MUL_MAT_ID`. Verified on RTX 5090 with compute-sanitizer memcheck.
- #29953 (2026-10-08): `ggml_cuda_mmq_get_J_max` rounded the column count down for allocation but up for config selection -> MMQ out-of-bounds reads; tile size now chosen first and used directly to compute padding.
- #29173 (2026-10-01): cuBLAS path for NVFP4 overflowed when compute type was FP16; compute type now handled (use FP32 accumulation).

## llama.cpp#22522 / #23530 / #24030 - Programmatic Dependent Launch (PDL)
- #22522 merged 2026-05-20. https://github.com/ggml-org/llama.cpp/pull/22522
- PDL for CC >= 90 (not Ada). Kernels gain a sync barrier `GGML_CUDA_PDL_SYNC` (placed before the first real read of an input tensor, i.e. `cudaGridDependencySynchronize`) and a launch signal `GGML_CUDA_PDL_LC` (trigger), launched via `ggml_cuda_kernel_launch()`. PDL + CUDA graphs are additive. Token generation +10% typical on RTX PRO 6000, +4-5% on DGX Spark; prefill neutral. Early trigger placement can be negative (notably `mul_mat_vec_q`).
- #23530 (2026-05-29): `__CUDA_ARCH_LIST__` cannot distinguish sm_90 vs sm_90a/f PTX; a kernel forward-JITed from older PTX must not opt into PDL. Guard at runtime with `cudaFuncAttributes::ptxVersion >= 90`.
- #24030 (2026-06-03): **PDL and `__restrict__` cannot coexist**: `__restrict__` lets the compiler emit `ld.global.nc` (non-coherent read-only path, outside the PTX memory model PDL relies on) and hoist those loads above the PDL wait -> races. `__restrict__` disabled in PDL-enabled device code; ~1% gain on Blackwell because FA kernels could be re-enrolled.

## llama.cpp#26705 - branchless Q4_K/Q5_K unpack + L2 prefetch on DGX Spark
- Merged 2026-09-07. https://github.com/ggml-org/llama.cpp/pull/26705
- Runtime branch in the K-quant scale unpack was predicated and replicated per `ncols_dst` column; mask-select version: predicated instructions at ncols_dst=8 226 -> 34. Gains from batch >= 4 on RTX 5090.
- Spark (low bandwidth, long_scoreboard 1.5-2.3x higher) regressed slightly at batch 4-6, so a `prefetch.global.L2` of upcoming weights is enabled **only on Spark**: +4.81% median decode on Spark, while on RTX 5090 it costs -1.57% (extra LSU/queue pressure, `lg_throttle` up, no latency left to hide at ne11=8). MMVQ->MMQ switch points (#26079) retuned.

## FlashAttention kernels in llama.cpp relevant to SM120
- llama.cpp#25635 (2026-08-31): XOR swizzle of K/V fp16 smem tiles in `fattn-mma-f16.cuh` (pow2 stride with per-row XOR remap) instead of +4 row padding, removing bank conflicts for `cp.async` stores and `ldmatrix` loads; helps very long contexts (65K) on DGX Spark. https://github.com/ggml-org/llama.cpp/pull/25635
- llama.cpp#29435 (2026-10-05): prefer whole-tile scheduling over Stream-K for FA prefill when the kernel has exactly two loading stages and tile/wave efficiency >= 75%; Stream-K started blocks for the same KV head at different K offsets, and causal truncation desynchronised them (worse L2 reuse). Large wins on DGX Spark long context, smaller on RTX 5090.

## MoE and decode fusions (measured on RTX PRO 6000 Blackwell)

### llama.cpp#27621 - extend MoE fusion to speculative decode (2-8 tokens)
- Merged 2026-08-31. https://github.com/ggml-org/llama.cpp/pull/27621
- `mul_mat_vec_q_moe` now folds the gate projection, biases, per-expert NVFP4 scales, the gate scale and the GLU into the up-projection kernel for 2-8 token batches (MTP / DFlash drafts), not only BS=1.
- The top-k router fusion (softmax -> topk -> get_rows) is extended from 1 row to batches of 4 (extensible to 8).
- 2-22% end-to-end decode speedup depending on model, quant and draft width; largest at draft widths 2 and 4, and larger with NVFP4 because the scale multiply is fused too.

### llama.cpp#25952 - fuse MoE weighted expert reduction
- Merged 2026-09-01. https://github.com/ggml-org/llama.cpp/pull/25952
- The MoE combine tail used to write weighted expert outputs to global memory and then reduce them, as two fused kernels for top-k=8. Now one kernel weights and reduces in order: `experts * router_weights`, or `(experts * expert_scale) * router_weights`.
- One runtime-k kernel covers k = 2..15. Graph matching checks shapes, strides, views and add-chain topology, so it does not depend on the model. It needs no scratch allocation.

### llama.cpp#29633 - MMVF for thin f16/bf16 mul_mat at small batch
- Merged 2026-10-05. https://github.com/ggml-org/llama.cpp/pull/29633
- Thin f16/bf16 matmuls at microbatch 1-8 use the MMVF vector kernel instead of the slow cuBLAS path (measured with `-ub 1..8` on RTX PRO 6000 Blackwell).

## KAI Notes
- For native FP4 on SM120 use `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64...ue4m3` (NVFP4) or `kind::mxf4.block_scale.scale_vec::2X...ue8m0` (MXFP4); pass one packed 32-bit scale register per operand.
- Accumulate MMA results in place into the final accumulator registers; separate temporary C tiles plus adds pushed MMQ to 255 registers with spills.
- When enabling PDL, place `griddepcontrol.wait` before the first dependent load and drop `__restrict__`/`ld.global.nc` on inputs written by the previous kernel; gate PDL on the loaded kernel's `ptxVersion`.
- Use L2 prefetch for decode GEMV only on bandwidth-poor GB10; it hurts RTX 5090.
- Fuse per-tensor NVFP4 scales into the GEMV/MMQ epilogue instead of a separate scaling kernel.
