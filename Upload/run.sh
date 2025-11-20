#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-baseline}
N=${2:-1024}

if [ "$MODE" = "baseline" ]; then
    echo "Running baseline C implementation"
    if [ ! -x baseline/sw_baseline ]; then
        echo "Baseline binary not found. Run 'make' first."
        exit 1
    fi
    baseline/sw_baseline $N
elif [ "$MODE" = "optimized" ]; then
    echo "Running optimized C++ implementation"
    if [ ! -x optimized/sw_optimized ]; then
        echo "Optimized binary not found. Run 'make' first."
        exit 1
    fi
    optimized/sw_optimized $N
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {baseline|optimized} [sequence_length]"
    exit 1
fi
