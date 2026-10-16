---
id: doc-nvidia-blackwell-tuning-guide-sm120
title: "NVIDIA Blackwell Tuning Guide Notes for Compute Capability 12.0"
url: https://docs.nvidia.com/cuda/archive/13.1.1/blackwell-tuning-guide/index.html
source_category: official-doc
architectures: [sm120]
tags: [tma, cluster, cuda-cpp]
retrieved_at: 2026-05-26
---

# NVIDIA Blackwell Tuning Guide Notes for Compute Capability 12.0

## Overview

The Blackwell tuning guide covers multiple Blackwell compute capabilities, including 10.0, 10.1, 12.0, and 12.1. For compute capability 12.0, the guide reports a different SM resource envelope from B200/SM100: 48 resident warps per SM, 64K 32-bit registers per SM, 32 resident blocks per SM, 128 KB shared memory per SM, and 99 KB maximum shared memory per thread block.

## Memory and Cluster Notes

Thread block clusters and Distributed Shared Memory remain supported on Blackwell. The guide recommends using CUDA occupancy APIs for cluster launches. For SM120, the shared-memory capacity is the key constraint to re-check when porting kernels that were written around SM100's larger shared-memory budget.

## KAI Notes

- When porting an SM100 kernel to SM120, first re-check shared-memory usage and occupancy instead of assuming the B200 resource envelope.
- If the kernel uses clusters, recompute cluster occupancy rather than inheriting B200 launch parameters.
- Treat TMA and general Blackwell memory guidance as relevant, but validate feature-specific availability against PTX/CUTLASS for SM120.

## Correction (2026-10-15)

The "32 resident blocks per SM" and "128 KB shared memory per SM" figures above are superseded. The CUDA 13.4 Programming Guide lists 24 blocks/SM and 100 KB shared memory/SM for CC 12.x (128 KB is the unified L1 + shared array), and `cudaDeviceGetAttribute` on an RTX PRO 6000 Blackwell (CC 12.0) returns `MaxSharedMemoryPerMultiprocessor = 102400`, `MaxSharedMemoryPerBlockOptin = 101376`, `MaxBlocksPerMultiprocessor = 24`. See `doc-cuda-programming-guide-cc12` and `hw-sm120-rtx-blackwell`.
