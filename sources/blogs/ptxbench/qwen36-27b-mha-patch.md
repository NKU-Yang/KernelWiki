---
id: blog-ptxbench-qwen36-27b-mha-patch
title: 'PTXBench: Kernel contract checklist for TMA/WGMMA attention (Qwen3.6-27B MHA task patch)'
author: PTXBench contributors
url: https://github.com/zhang677/PTXBench/blob/163e3734eed4ae66af8cfb7ac55976178d576603/multiturn/prompts/fragments/structural_doc/patterns/qwen36-27b-mha-patch.md
source_category: community-note
architectures:
- sm90
tags:
- tma
- mbarrier
- wgmma
- swizzling
- attention
retrieved_at: '2026-10-15'
description: Verbatim PTXBench structural_doc prompt fragment. Constraints stated inside it (for example autotune bans or fixed task shapes) were PTXBench task rules, not general KernelWiki guidance.
---

# PTXBench: Kernel contract checklist for TMA/WGMMA attention (Qwen3.6-27B MHA task patch)

1. In your thinking, write a short "kernel contract" block listing tensor shapes, flattened offsets, tile sizes, TMA descriptor dimensions/strides/boxDim/swizzle, mbarrier counts, and launch shared-memory bytes. The code must match this block exactly.
2. For TMA BF16 with 128B swizzle and flattened `[B*H*S, D]`, use `globalDim={D, B*H*S}`, `globalStrides={D*2}`, and `boxDim={64,128}` unless the kernel explicitly proves a different layout.
3. Every host-side CUDA/driver call must be checked: `CU_CHECK(cuTensorMapEncodeTiled(...))`, `CUDA_CHECK(cudaFuncSetAttribute(...))`, `CUDA_CHECK(cudaGetLastError())`, and `CUDA_CHECK(cudaStreamSynchronize(stream))`.
4. If dynamic shared memory exceeds the default limit, the kernel must opt in with `cudaFuncSetAttribute` using the same byte count as the launch.
5. The online softmax update must explicitly maintain `m_i`, `l_i`, and rescale old `O` accumulators when the row max changes. Any kernel that computes a tile softmax independently and then accumulates `P @ V` is wrong for full attention.
6. Prohibit unsynchronized shared accumulator updates. Each accumulator element must have a single owner thread/warp, or use an explicit reduction with a documented synchronization point.
7. Require a self-audit checklist at the end of generation: descriptor validity, barrier arrival/tx-count matching, WGMMA descriptor LBO/SBO, output/LSE indexing, and launch shared memory.