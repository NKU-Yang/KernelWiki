---
id: blog-ptxbench-arch-param-b200
title: 'PTXBench: B200 (SM100) hardware parameters'
author: PTXBench contributors
url: https://github.com/zhang677/PTXBench/blob/163e3734eed4ae66af8cfb7ac55976178d576603/multiturn/prompts/fragments/structural_doc/document/arch_param_b200.md
source_category: community-note
architectures:
- sm100
tags:
- tmem
- cluster
- fp8
retrieved_at: '2026-10-15'
description: Verbatim PTXBench structural_doc prompt fragment. Constraints stated inside it (for example autotune bans or fixed task shapes) were PTXBench task rules, not general KernelWiki guidance.
---

# PTXBench: B200 (SM100) hardware parameters

Below is the hardware parameter of NVIDIA B200 (SM100)
| Component | Value |
|-----------|---------------|
| SM Count | 148 |
| Peak MMA FP8 | ~4500 TFLOPS |
| Peak MMA BF16 | ~2250 TFLOPS (8192 ops/cycle/SM) |
| Peak MUFU (multifunction unit) | 16 ops/cycle/SM |
| Cluster Size | Portable: 8 CTAs; B200 nonportable: up to 16 CTAs |
| Max Registers / CTA | 65536 32-bit|
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 180 GB | ~400 cycles | 8 TB/s |
| L2 Cache | 126 MB | ~100 cycles | ~16TB/s |
| Shared Memory | 0, 8, 16, 32, 64, 100, 132, 164, 196, and 228 KB per SM. A single CTA can address up to 227 KB of shared memory as 1 KB is reserved by CUDA | ~20 cycles | 128 B/cycle/SM |
| Tensor Memory (TMEM) | 256 KB (512 columns × 128 rows x 32-bit cells) / SM |  ~10 cycles | Tensor Core internal |
| Registers | 256 KB / SM  | 1 cycle | Unlimited (matched with compute) |