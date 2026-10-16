---
id: doc-nvidia-cuda-programming-guide-sm120
title: "CUDA Programming Guide: Compute Capability 12.x Feature Targets"
url: https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/compute-capabilities.html
source_category: official-doc
architectures: [sm120]
tags: [ptx, fp4, fp6, fp8, cuda-cpp]
retrieved_at: 2026-05-26
---

# CUDA Programming Guide: Compute Capability 12.x Feature Targets

## Overview

The CUDA Programming Guide documents baseline, family-specific, and architecture-specific compiler targets for recent architectures. For SM120, the relevant target names include `compute_120`, `compute_120f`, and `compute_120a`, with `sm_120`, `sm_120f`, and `sm_120a` as PTX/SASS target strings.

## Feature Notes

Compute capability 12.x is listed as supporting modern Tensor Core input types including FP8, FP6, FP4, BF16, FP16, TF32, INT8, and INT4. It also retains modern CUDA features such as TMA, thread block clusters, Distributed Shared Memory, hardware `memcpy_async`, split arrive/wait barriers, and L2 cache residency controls.

## KAI Notes

- Baseline `sm_120` is not always enough for architecture-specific Tensor Core features; check whether `sm_120a` or `sm_120f` is required.
- Prefer `-gencode=arch=compute_120,code=sm_120` plus PTX retention for normal kernels, and use architecture/family targets only when the instruction being emitted requires them.
- If handwritten PTX fails to assemble, inspect whether the target suffix is wrong before changing the algorithm.

## Correction (2026-10-15)

The CUDA 13.4 Programming Guide's CC 12.x tensor-core input list is TF32, BF16, FP16, FP8, FP6, FP4 and INT8 (no FP64 and no INT4 tensor-core input). Per-SM limits for CC 12.x: 100 KB shared memory/SM, 99 KB/block opt-in, 24 blocks/SM, 48 warps/SM. See `doc-cuda-programming-guide-cc12`.
