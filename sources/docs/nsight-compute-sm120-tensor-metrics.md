---
id: doc-nsight-compute-sm120-tensor-metrics
title: "Nsight Compute 2026.3: SM120 (GB20x) support and tensor-pipe metrics for mma.sync kernels"
url: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
source_category: official-doc
architectures: [sm120]
tags: [gemm, attention, tma, warp-specialization, pipeline-stages, shared-memory-optimization]
retrieved_at: 2026-10-15
---

# Nsight Compute 2026.3.x: profiling SM120 tensor-core kernels

Sources: Nsight Compute Release Notes (v2026.3.1), Profiling Guide, and the Compute Triage Guide. SM120
consumer/workstation chips appear in Nsight Compute as **GB20x** (GB202 etc.); GB10 / DGX Spark appears in
the cuBLAS notes as "DGX Spark" (CC 12.1). Nsight Compute chip names GB20b / GB20c are also listed as
supported; GB10x = data-center Blackwell (CC 10.x), GB10b = Thor.

## SM120 support history in Nsight Compute

- 2024.4: "Added support for the Blackwell architecture."
- 2025.1.1: "Fixed the missing roofline ceilings in the Floating Point Operations Roofline for GB20x chips."
- 2025.2: "Fixed that some metrics were not listed for collection in section files for GB20x GPUs."
- 2026.2: added support for CUDA 13.3; added SASS instruction-size metrics; added profiling for previously
  unsupported SKUs "with reduced metric support, including GPUs such as RTX 6000D BSE (GB202-891), RTX 5090DD
  (GB202-240), and RTX 5090D (GB202-250)"; Instruction Statistics shows warp-can't-issue stall samples per HW
  warp ID slot.
- 2026.3.1: limited C2C metrics for GB20B and GB20C. Known issue: on GB20B/C, C2C
  `pct_of_peak_sustained_elapsed` metrics report NaN.
- Supported GPU table (2026.3.1): Turing TU1xx, GA100, GA10x, GA10b, AD10x, GH100, GB10x, GB10b, GB11x,
  GB20x, GB20b, GB20c, GR100 = Yes; Maxwell/Pascal/Volta = No.

## SM120 tensor pipeline metric names (mma.sync path)

From the Compute Triage Guide: "sm__pipe_tensor_cycles_active applies to SM 7.0-12.0, while SM 10.x+ adds
sm__pipe_tc_cycles_active for UTCMMA-capable chips." SM120 has no UTCMMA (tcgen05), so on SM120 the
relevant counters are the **tensor** pipe ones:

```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed   # tensor pipe utilization
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed   # exact name unverified; doc gives generic sm__inst_executed_pipe_*
sm__pipe_*_cycles_active.avg.pct_of_peak_sustained_elapsed        # find the limiting pipe
sass__inst_executed_per_opcode                                    # confirm MMA SASS opcodes are present (opcode names not documented here)
```

Pipeline definitions (Profiling Guide): "tensor - The Tensor pipeline executes various MMA instructions. It is
different from the Tensor Core pipeline." "tc - Tensor Core. The TC pipeline executes UTCBAR, UTCCP, UTC*MMA,
UTCSHIFT and UTC*SWS instructions." "tmem - executes FENCE.VIEW.ASYNC.T, LDT(M) and STT(M)". On SM120 expect
`tc` and `tmem` to be absent or zero. "tma - Tensor Memory Accelerator ... between global and shared memories."

The guide warns metric names vary by architecture: "Run `ncu --query-metrics | grep sm__pipe_` on your target
GPU to discover the available pipeline metrics before writing analysis scripts or setting thresholds." Older
per-type sub-metrics such as `sm__inst_executed_pipe_tensor_op_imma.avg.pct_of_peak_sustained_active` may not
exist on a given chip (n/a = metric does not exist). Exact SM120 sub-pipe names (e.g. per-dtype tensor_op_*)
were not verified - query them.

For datatype/sparsity breakdown of tensor work use `sm__ops_*` metrics (documented in the Metrics Decoder).

## SM120 warp-level MMA counter semantics

From "MMA fundamentals" (Compute Triage Guide):

- "Warp-level MMA (Volta+): runs on a single warp scheduler using all 32 threads of one warp. All inputs and
  outputs reside in the register file. Cannot be predicated - all 32 threads must be active... Source
  counters: 1 warp instruction with 32 threads predicated on average (cannot be lower)." This is the only MMA
  family on SM120.
- WGMMA (Hopper) shows 4 warp instructions per logical MMA; 5th-gen MMA (Blackwell DC) shows 1 warp instruction
  with 1 active thread - neither applies to SM120.
- Each MMA does about `2*M*N*K` FLOPs; e.g. one `m16n8k64` FP4 mma = 16384 FLOPs, one `m16n8k32` FP8 mma = 8192
  FLOPs (arithmetic from the formula).
- Tensor kernels "typically show low scheduler issue-active but very high pipeline throughput per issued
  instruction ... a low issue rate is expected"; treat HighPipeUtilization (tensor pipe) as the primary signal.

## SM120 triage thresholds and pipelining guidance

- Limiter identification: highest `sm__pipe_*` value; "Above ~60% is high pressure; above ~80% is likely the
  binding constraint."
- "Computing global-to-shared address indexing for tiles is often a real bottleneck ... If address-math
  pipelines (e.g. ADU) saturate before the tensor pipe, offload indexing to TMA."
- "TMA copies are issued by a single thread and free warps from address-math work"; pipeline next-tile TMA
  with current-tile MMA; "Async MMA + async TMA only pay off if the kernel actually overlaps them".
- Warp specialization: dedicate producer warps to TMA loads and consumer warps to MMA.
- Use the Tensor roofline section (`SpeedOfLight_HierarchicalTensorRooflineChart`) for tensor kernels, not
  the FP32 roofline; roofline mismatch can come from clock basis (cross-check reported SM frequency).
- LSU writeback disambiguation: low instruction throughput + high LSU writeback = wide loads saturating the
  return path; high instruction throughput + high writeback = too many memory instructions.
- Shared memory: 32 banks x 4 B; a 128 B row matches the L1 line. Memory tables: Instructions -> Requests ->
  Wavefronts -> Sectors to find where inflation starts.

## SM120 practical ncu commands

```
ncu --query-metrics | grep sm__pipe_                      # discover pipes on the RTX 50 / RTX PRO 6000
ncu --set full -k regex:my_gemm ./app
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_elapsed ./app
ncu --section SpeedOfLight_HierarchicalTensorRooflineChart ./app
```

Caveats: `--clock-control` cannot set clocks on a MIG partition and is not supported on GA10b/GB10b Linux
aarch64; reduced metric support on some GB202 SKUs (RTX 5090D/5090DD/6000D); profiling requires
performance-counter permission (ERR_NVGPUCTRPERM; on WSL enable it in the Windows NVIDIA Control Panel).

## References

- https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (Pipelines table, FAQ)
- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html (3.4 Compute, MMA fundamentals)

## KAI Notes

- On SM120 judge tensor-core use by `sm__pipe_tensor_cycles_active` (not `sm__pipe_tc_*`, which belongs to tcgen05 chips); a well-tuned FP4/FP8 mma.sync GEMM should push it toward 80%+.
- Low issue-slot utilization in an mma.sync GEMM is normal; look instead for ADU/LSU/ALU pipes saturating before the tensor pipe (address math, ldmatrix count).
- Confirm in `sass__inst_executed_per_opcode` that the expected MMA opcodes appear and that no FP4/FP8 conversion emulation sequences dominate.
- Always run `ncu --query-metrics` on the actual GB20x device before hard-coding metric names; some GeForce SKUs have reduced metric support.
- Profiles inform optimization but timing must come from repeated measured runs, not ncu duration (ncu replays and may lock clocks).
