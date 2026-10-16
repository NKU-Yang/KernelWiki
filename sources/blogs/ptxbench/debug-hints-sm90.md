---
id: blog-ptxbench-debug-hints-sm90
title: 'PTXBench: Common Causes of Incorrect Results'
author: PTXBench contributors
url: https://github.com/zhang677/PTXBench/blob/163e3734eed4ae66af8cfb7ac55976178d576603/multiturn/prompts/fragments/structural_doc/document/debug_hints_sm90.md
source_category: community-note
architectures:
- sm90
tags:
- tma
- mbarrier
- wgmma
- cluster
- swizzling
retrieved_at: '2026-10-15'
description: Verbatim PTXBench structural_doc prompt fragment. Constraints stated inside it (for example autotune bans or fixed task shapes) were PTXBench task rules, not general KernelWiki guidance.
---

# PTXBench: Common Causes of Incorrect Results

Below are some common errors and solutions.
# Common Causes of Incorrect Results
1. Wrong wgmma descriptor stride: Revise the tma doc

2. Swizzle mismatch between TMA and wgmma descriptor

3. Missing fence.proxy.async (manual loads before WGMMA)
```cpp
// WRONG
smem[i] = global[i];
__syncthreads();
wgmma_m64n64k16_fn_(acc, da, db);  // WGMMA may see stale data!
// CORRECT
smem[i] = global[i];
__syncthreads();
fence_proxy_async_fn()
__syncwarp();
wgmma_fence_fn();
wgmma_m64n64k16_fn_(acc, da, db);
```

4. Barrier race conditions
```cpp
// WRONG: empty arrives before math finishes
mbarrier_arrive_fn(empty_barriers[s]);
wgmma_wait_fn();  // Race!
// CORRECT
wgmma_wait_fn();
mbarrier_arrive_fn(empty_barriers[s]);
```

# Execution timeout
Hangs and Deadlocks
Cause: Barrier expected arrivals don't match actual arrivals.
Debug:
```cpp
  // Add debug prints (disable in production!)
  if (tid == 0) {
      printf("Block %d: init barrier %d with %d arrivals\n",
             blockIdx.x, s, expected_arrivals);
}
```
Common mistakes: - init_smem_barrier_fn(empty_barriers[s], 4) but only 3 consumer warps call arrive() - Cluster barriers initialized with wrong count - Not all threads reach arrive() due to early exit.

# Register Spill
Symptom: Performance drops dramatically
Causes: 1. Too many accumulators live simultaneously 2. Loop unrolling creates too many variables 3. Missing #pragma unroll causing inefficient code
Solutions: 1. Process tiles sequentially with accumulator reuse 2. Use #pragma nounroll where appropriate 3. Reduce NUM_MATH_REGS request

Debug hints end.