# Blackwell Kernel Optimization Knowledge Base

> Comprehensive knowledge base for GPU kernel optimization on NVIDIA Blackwell (SM100) and Hopper (SM90).
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and conventions.
> **For Claude Code agents**: this repository is a Claude Code skill — see [SKILL.md](SKILL.md).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for 10 worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Browse exact, family-only, or unknown architecture evidence | [queries/by-architecture.md](queries/by-architecture.md) |
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |

## Hardware Features

- [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md) — Blackwell MMA instruction (replaces wgmma)
- [hw-tmem](wiki/hardware/tmem.md) — Tensor Memory (CTA-visible 128-lane × 512-column view)
- [hw-clc](wiki/hardware/clc.md) — Cluster Launch Control (dynamic tile scheduling)
- [hw-tma](wiki/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads)
- [hw-2sm-cooperative](wiki/hardware/2sm-cooperative.md) — Two-SM cooperative MMA
- [hw-nvfp4](wiki/hardware/nvfp4.md) — NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — Programmatic Dependent Launch / Grid Dependency Control
- [hw-sm120-rtx-blackwell](wiki/hardware/sm120-rtx-blackwell.md) — RTX/GeForce Blackwell compute capability 12.0 target

## Optimization Techniques

- [technique-warp-specialization](wiki/techniques/warp-specialization.md) — Warp role assignment
- [technique-persistent-kernels](wiki/techniques/persistent-kernels.md) — Persistent kernel patterns with CLC
- [technique-swizzling](wiki/techniques/swizzling.md) — Shared memory swizzling
- [technique-pipeline-stages](wiki/techniques/pipeline-stages.md) — Software pipelining
- [technique-epilogue-fusion](wiki/techniques/epilogue-fusion.md) — Fusing epilogue with mainloop
- [technique-tile-scheduling](wiki/techniques/tile-scheduling.md) — Tile scheduling strategies
- [technique-double-buffering](wiki/techniques/double-buffering.md) — Double/multi-buffering
- [technique-software-exp](wiki/techniques/software-exp.md) — Software-emulated exponential
- [technique-fine-grained-quantization](wiki/techniques/fine-grained-quantization.md) - Fine-grained FP8/FP4 quantization
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md) — Wide vectorized loads and cache policies
- [technique-sm120-blockscaled-mma-sync](wiki/techniques/sm120-blockscaled-mma-sync.md) — SM120 NVFP4/MXFP4/MXFP8 `mma.sync` GEMM: fragments, scale lanes, TMA pipeline, verified test

## Kernel Case Studies

- [kernel-flash-attention-4](wiki/kernels/flash-attention-4.md) — FlashAttention-4 (up to 1613 TFLOPS on B200 in the paper's benchmark sweep)
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — DeepGEMM FP8 GEMM (1550 TFLOPS on H800)
- [kernel-flashmla](wiki/kernels/flashmla.md) — FlashMLA sparse/dense MLA decoding
- [kernel-nsa](wiki/kernels/nsa.md) — Native Sparse Attention (9x fwd speedup)
- [kernel-gated-delta-net](wiki/kernels/gated-delta-net.md) — Gated Delta Net linear attention
- [kernel-nvfp4-gemm](wiki/kernels/nvfp4-gemm.md) — NVFP4 GEMM from GPU Mode hackathon
- [kernel-nvfp4-gemv](wiki/kernels/nvfp4-gemv.md) — NVFP4 batched GEMV optimization
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — Fused MoE with FP8
- [kernel-sm120-blockscaled-gemm-moe](wiki/kernels/sm120-blockscaled-gemm-moe.md) — SM120/SM12x blockscaled GEMM, b12x, and fused MoE
- [kernel-sm120-attention](wiki/kernels/sm120-attention.md) — SM120 attention: TMA + mma.sync recipe, FP8/NVFP4 attention, decode pitfalls

## Problem → Solution Patterns

- [pattern-low-sm-utilization](wiki/patterns/low-sm-utilization.md) — SM utilization is low
- [pattern-memory-bound](wiki/patterns/memory-bound.md) — Kernel is memory bandwidth limited
- [pattern-register-pressure](wiki/patterns/register-pressure.md) — Too many registers → low occupancy
- [pattern-compute-bound](wiki/patterns/compute-bound.md) — Not reaching peak FLOPS
- [pattern-tail-effect](wiki/patterns/tail-effect.md) — Last wave underutilizes GPU
- [pattern-sm120-porting-failures](wiki/patterns/sm120-porting-failures.md) — SM100 assumptions or SM12x guards break RTX Blackwell kernels

## Languages & DSLs

- [lang-cute-dsl](wiki/languages/cute-dsl.md) — CuTe DSL for Blackwell
- [lang-cuda-cpp](wiki/languages/cuda-cpp.md) — CUDA C++ with PTX inline
- [lang-ptx](wiki/languages/ptx-sm100.md) — PTX instructions for SM100
- [lang-ptx-sm120](wiki/languages/ptx-sm120.md) — PTX/CUTLASS instruction families for SM120
- [lang-cutlass-sm120-cute](wiki/languages/cutlass-sm120-cute.md) — CuTe API naming guardrails for SM120 MMA
- [lang-triton](wiki/languages/triton-blackwell.md) — Triton on Blackwell

## Migration Guides

- [migration-wgmma-to-tcgen05](wiki/migration/wgmma-to-tcgen05.md) — Hopper wgmma → Blackwell tcgen05
- [migration-register-to-tmem](wiki/migration/register-to-tmem.md) — Register accumulators → TMEM
- [migration-sm100-to-sm120](wiki/migration/sm100-to-sm120.md) — B200/SM100 assumptions → RTX Blackwell/SM120 guardrails

## Source Repositories

| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | CUTLASS 4.x Blackwell support |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | SGLang Blackwell integration |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | vLLM Blackwell support |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | FlashInfer Blackwell kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | PyTorch/Inductor Blackwell |

## Competitions

- [GPU Mode NVFP4 Hackathon](sources/contests/gpu-mode-nvfp4/) — 4 NVFP4 kernel challenges on B200
- [FlashInfer MLSys 2026](sources/contests/flashinfer-mlsys26/) — MoE, Sparse Attention, GatedDeltaNet

## PTXBench Structural Docs

Verbatim PTXBench (zhang677/PTXBench @ 163e3734) `structural_doc` prompt fragments, imported as `community-note` source pages under `sources/blogs/ptxbench/` (40 pages) with the helper headers, LICENSE and NOTICE in `artifacts/blogs/ptxbench-structural-doc/code/`. Task rules inside them (autotune bans, fixed shapes) were PTXBench task constraints, not general guidance.

- Hardware parameters: `blog-ptxbench-arch-param-b200`, `blog-ptxbench-arch-param-h100`
- PTX excerpts: `blog-ptxbench-ptx-knowledge-sm100-plus`, `blog-ptxbench-ptx-knowledge-sm90-plus`, `blog-ptxbench-debug-hints-sm90`
- Triton contracts: `blog-ptxbench-triton-knowledge-sm100-plus`, `blog-ptxbench-triton-knowledge-sm90-plus`
- Optimization patterns: FA3/FA4 forward/backward, persistent kernels, warp specialization, TMA store, cluster multicast, L2 swizzle (`python3 scripts/query.py ptxbench --compact`)
- Helper headers with PTX usage notes: `blog-ptxbench-header-{barrier,simt,tma-bf16,tma-bf16-sm100,umma-bf16,wgmma-bf16}`
