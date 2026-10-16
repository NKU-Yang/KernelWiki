---
id: paper-blackwell-sm120-microbenchmarking-rtx5080
title: "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks"
url: https://arxiv.org/abs/2507.10789
source_category: paper
architectures: [sm120]
tags: [gemm, fp4, fp6, fp8, cuda-cpp]
retrieved_at: 2026-05-26
---

# Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks

## Overview

This public microbenchmark paper studies a GeForce RTX 5080 Blackwell GPU and compares it with H100 PCIe. The paper examines memory hierarchy, SM execution pipelines, SM sub-core units, fifth-generation Tensor Cores, FP4/FP6 behavior, scheduling, cache behavior, latency, throughput, and energy behavior.

## KAI Notes

- Treat this as an SM120 microarchitecture source, not an RTX PRO 6000 product-spec source.
- Use it to reason about architectural bottlenecks and precision behavior on consumer/workstation Blackwell.
- Validate absolute performance claims on the local RTX PRO 6000 hardware before encoding them as assumptions.
