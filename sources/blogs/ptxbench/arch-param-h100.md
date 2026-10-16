---
id: blog-ptxbench-arch-param-h100
title: 'PTXBench: H100 (SM90) hardware parameters'
author: PTXBench contributors
url: https://github.com/zhang677/PTXBench/blob/163e3734eed4ae66af8cfb7ac55976178d576603/multiturn/prompts/fragments/structural_doc/document/arch_param_h100.md
source_category: community-note
architectures:
- sm90
tags:
- cluster
- fp8
retrieved_at: '2026-10-15'
description: Verbatim PTXBench structural_doc prompt fragment. Constraints stated inside it (for example autotune bans or fixed task shapes) were PTXBench task rules, not general KernelWiki guidance.
---

# PTXBench: H100 (SM90) hardware parameters

Below is the hardware parameter of NVIDIA H100
| Component | Value |
|-----------|---------------|
| SM Count | 132 |
| Peak FP8 | ~1978 TFLOPS |
| Peak BF16 | ~989 TFLOPS |
| Cluster Size | Up to 16 SMs |
| Max Registers / CTA | 65536 |
| Max Registers / Thread | 255 |
| Max CTA size (# of threads) | 1024 |

| Level | Capacity | Latency | Bandwidth |
|-------|----|------|---------|
| Global (HBM) | 80 GB | ~500 cycles | 3.35 TB/s |
| L2 Cache | 50 MB | ~100 cycles | ~12TB/s |
| Shared Memory | 256 KB / SM (228 KB usable with 28KB L1 Cache) | ~30 cycles | 128 B/cycle/SM |
| Registers | 256 KB / SM  | 1 cycle | Unlimited (matched with compute) |