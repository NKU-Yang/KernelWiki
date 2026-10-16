---
id: doc-cutlass-sm120
title: "CUTLASS Blackwell SM120 GEMM Functionality"
url: https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
source_category: official-doc
architectures: [sm120]
tags: [gemm, fp4, fp6, fp8, block-scale, cuda-cpp, cute-dsl]
retrieved_at: 2026-05-26
---

# CUTLASS Blackwell SM120 GEMM Functionality

## Overview

CUTLASS documents SM120 GEMMs separately from SM100 GEMMs. For SM120 dense narrow-precision tensor core kernels, CUTLASS uses the `mma.sync.aligned.kind::f8f6f4` instruction family. For sparse narrow-precision tensor core kernels, CUTLASS uses the `mma.sp::ordered_metadata.sync.aligned.kind::f8f6f4` instruction family.

## Key SM120 GEMM Constraints

- Dense SM120 narrow-precision MMA shape: 1-SM `m16n8k32`.
- Sparse SM120 narrow-precision MMA shape: 1-SM `m16n8k64`.
- Cluster shape is fixed to `1x1x1` for GeForce-series SM120 paths because multicast is not available there.
- Only TN tensor layout is supported in the documented SM120 path: A row-major, B column-major.
- Input tensors must satisfy CUTLASS alignment requirements; the documented mainloop assumes 16-byte alignment.

## KAI Notes

- For SM120 RTX Blackwell, start from CUTLASS SM120 atom/builders or the documented `mma.sync` / `mma.sp::ordered_metadata` families.
- Do not use SM100 `tcgen05.mma`, TMEM, or 2-SM cooperative assumptions as the default for SM120 GEMM.
- If the task is sparse FP4/MXFP4/NVFP4 GEMM, the key semantic issue is ordered sparse metadata, not simulating sparsity with ordinary dense loads.
