---
id: doc-nvidia-cuda-gpus-sm120
title: "NVIDIA CUDA GPUs: Compute Capability 12.0 Devices"
url: https://developer.nvidia.com/cuda/gpus
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp]
retrieved_at: 2026-05-26
---

# NVIDIA CUDA GPUs: Compute Capability 12.0 Devices

## Overview

NVIDIA's CUDA GPU compute capability table identifies SM120 / compute capability 12.0 as the target for RTX Blackwell workstation and consumer GPUs. This includes the RTX PRO 6000 Blackwell workstation and server family, RTX PRO 5000/4500/4000/2000 Blackwell products, and the GeForce RTX 50 series such as RTX 5090 and RTX 5080.

## Why This Matters

Use this source to distinguish SM120 RTX Blackwell devices from SM100/SM100a datacenter Blackwell devices such as B200 and GB200. A kernel tuned for B200 should not automatically assume the same tensor-core instruction path, memory system, cluster behavior, or product-level bandwidth on RTX PRO 6000 Blackwell.

## KAI Notes

- Treat `sm120`, `sm_120`, `compute_120`, RTX PRO 6000 Blackwell, and GeForce RTX 50 as the same architecture family for lookup.
- Do not map SM120 to B200 unless the optimization is explicitly described as portable across Blackwell families.
- Prefer runtime detection through CUDA device attributes or `nvidia-smi --query-gpu=name,compute_cap` when a run can target multiple GPUs.
