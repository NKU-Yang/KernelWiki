---
id: doc-nvidia-tensorrt-rtx-sm12
title: "TensorRT-RTX Support Matrix: SM 12.x Blackwell GeForce"
url: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/support-matrix.html
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, fp4, fp8]
retrieved_at: 2026-05-26
---

# TensorRT-RTX Support Matrix: SM 12.x Blackwell GeForce

## Overview

TensorRT-RTX documentation separates Blackwell RTX/GeForce devices from Blackwell datacenter devices. It labels SM 12.x as Blackwell GeForce, covering GeForce RTX 50 series and NVIDIA RTX PRO 6000 Blackwell, while SM 10.x covers Blackwell datacenter products such as B200 and GB200.

## KAI Notes

- Use this as a high-level product-family guardrail: RTX PRO 6000 Blackwell is SM12.x, not the B200 SM10.x path.
- When an optimization suggestion depends on SM100 datacenter-only assumptions, require a source that explicitly says it applies to SM120.
