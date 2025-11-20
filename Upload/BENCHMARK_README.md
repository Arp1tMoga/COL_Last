# Smith-Waterman Benchmark Setup

## Overview
This document describes the updated build and benchmark system for the Smith-Waterman algorithm implementations.

## Files Updated

### 1. Makefile
**Location:** `/home/arpit/coding/final_ASS/Upload/Makefile`

**Changes:**
- Added compilation rules for both baseline C and optimized C++ implementations
- Baseline: `baseline/sw_baseline` (compiled with gcc)
- Optimized: `optimized/sw_optimized` (compiled with g++ with OpenMP and C++17)

**Usage:**
```bash
make              # Build both baseline and optimized
make clean        # Remove compiled binaries
```

### 2. run.sh
**Location:** `/home/arpit/coding/final_ASS/Upload/run.sh`

**Changes:**
- Updated to run Smith-Waterman implementations instead of GEMM
- Supports both baseline and optimized modes
- Takes sequence length as parameter

**Usage:**
```bash
./run.sh baseline 512     # Run baseline with N=512
./run.sh optimized 1024   # Run optimized with N=1024
```

### 3. benchmark.sh (NEW)
**Location:** `/home/arpit/coding/final_ASS/Upload/benchmark.sh`

**Features:**
- Runs both baseline and optimized implementations
- Tests multiple input sizes: 512, 1024, 2048, 4096, 8192, 16384, 32768
- Executes each configuration 10 times
- Calculates statistics: Average, Median, Standard Deviation
- Displays results in real-time with color-coded terminal output
- Saves all results to `results.csv`
- Shows speedup comparison

**Usage:**
```bash
./benchmark.sh            # Run full benchmark suite
```

## Benchmark Output

### Terminal Output
The benchmark displays:
- **Green**: Baseline (C) implementation results
- **Blue**: Optimized (C++) implementation results
- **Yellow**: Speedup calculations
- Real-time statistics as tests complete
- Formatted in a clean tabular layout

### CSV Output
File: `results.csv`

Structure:
```
Implementation,Input Size,Run,Time (s),Average (s),Median (s),Std Dev (s)
Baseline (C),512,1,0.001049,,,
...
Baseline (C),512,Summary,-,0.001189,0.001052,0.000187
Optimized (C++),512,1,0.000065,,,
...
Optimized (C++),512,Summary,-,0.000064,0.000064,0.000001
Speedup,512,-,-,18.57x,-,-
```

## Quick Start

1. **Compile the code:**
   ```bash
   cd /home/arpit/coding/final_ASS/Upload
   make
   ```

2. **Test individual runs:**
   ```bash
   ./run.sh baseline 512
   ./run.sh optimized 512
   ```

3. **Run full benchmark:**
   ```bash
   ./benchmark.sh
   ```

4. **View results:**
   ```bash
   cat results.csv
   # Or open in a spreadsheet application
   ```

## Performance Notes

From initial testing with N=512:
- **Baseline:** ~0.001189 seconds
- **Optimized:** ~0.000064 seconds
- **Speedup:** ~18.57x

The optimized implementation uses:
- SIMD instructions (AVX2/AVX512 when available)
- Striped Smith-Waterman algorithm
- Cache-optimized memory access patterns

## Input Sizes

The benchmark tests the following sequence lengths (powers of two):
- 512
- 1024
- 2048
- 4096
- 8192
- 16384
- 32768

Each size is tested 10 times to ensure statistical reliability.

## Statistics Calculated

For each configuration:
- **Average:** Mean execution time across all runs
- **Median:** Middle value of sorted execution times
- **Standard Deviation:** Measure of variance in execution times

These statistics help identify:
- Typical performance
- Outliers
- Consistency of results
