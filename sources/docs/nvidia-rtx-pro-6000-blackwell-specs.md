---
id: doc-nvidia-rtx-pro-6000-blackwell-specs
title: "NVIDIA RTX PRO 6000 Blackwell Architecture and Product Specifications"
url: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf
source_category: official-doc
architectures: [sm120]
tags: [fp4, fp8, cuda-cpp]
retrieved_at: 2026-05-26
---

# NVIDIA RTX PRO 6000 Blackwell Architecture and Product Specifications

## Overview

NVIDIA's RTX PRO Blackwell architecture material identifies the workstation-class RTX PRO 6000 Blackwell as a GB202-based RTX Blackwell GPU with 188 SMs, 24,064 CUDA cores, 752 fifth-generation Tensor Cores, 96 GB GDDR7, 512-bit memory interface, and 1,792 GB/s memory bandwidth for the workstation edition. NVIDIA's product family pages separately document the workstation, server, and Max-Q editions and their power envelopes.

## KAI Notes

- RTX PRO 6000 Blackwell has far less memory bandwidth than B200 HBM parts, so B200 bandwidth-bound conclusions do not transfer directly.
- Re-tune kernels that were written for B200's HBM and larger shared-memory budget.
- For E2E optimization, memory movement, launch overhead, CPU/GPU synchronization, and framework overhead may dominate even when raw Tensor Core capability is high.
