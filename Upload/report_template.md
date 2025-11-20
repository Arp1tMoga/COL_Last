# Report Template — Accelerator Leaderboard Submission


## Team: <team name>


## 1. Problem description
- Application: GEMM (dense matrix multiply)
- Input sizes tested: N = ...


## 2. Baseline
- Description of baseline implementation (Python multiprocessing + NumPy)
- Baseline timings and environment (CPU model, cores, OS)


## 3. Optimizations implemented
- Multithreading (how: OpenMP / tasks / partitioning)
- Cache blocking / tiling (explain block sizes and rationale)
- SIMD intrinsics used (AVX2/AVX-512) and code excerpts
- NUMA-aware allocation and thread pinning
- Prefetching and unrolling


## 4. Experimental methodology
- Commands used to run baseline and optimized
- Number of runs, how median chosen, perf counters collected


## 5. Results
- Tables and plots: runtime, GFLOPs, speedup vs baseline, scaling with threads
- Microarchitecture counters: L1/L2/L3 misses, bandwidth, IPC


## 6. Analysis
- Explain why the optimizations succeeded or failed.
- Identify bottlenecks and suggest further improvements.


## 7. Reproducibility
- How to reproduce runs (commands, environment, git hash)
