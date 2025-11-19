#!/bin/bash
# Complete profiling workflow for line-level analysis

echo "=== Compiling with O3 and debug symbols ==="
g++ optimized/sw_optimized.cpp -O3 -g3 -fno-omit-frame-pointer -mavx2 -fopenmp -std=c++17 -o optimized/sw_optimized

echo ""
echo "=== Running Callgrind (this will take a while...) ==="
valgrind --tool=callgrind \
  --dump-instr=yes \
  --collect-jumps=yes \
  --cache-sim=yes \
  --branch-sim=yes \
  ./optimized/sw_opt_profile

# Get the latest callgrind file
CALLGRIND_FILE=$(ls -t callgrind.out.* | head -1)

echo ""
echo "=== Profiling complete! ===" 
echo "Output file: $CALLGRIND_FILE"
echo ""
echo "View summary:"
echo "  callgrind_annotate $CALLGRIND_FILE | less"
echo ""
echo "Launch KCacheGrind:"
echo "  kcachegrind $CALLGRIND_FILE"
