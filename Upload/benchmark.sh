#!/usr/bin/env bash
set -euo pipefail
# Configuration
ITERATIONS=10
# Input sizes from 2^9 to 2^15
INPUT_SIZES=(512 1024 2048 4096 8192 16384 32768)
RESULTS_FILE="results.csv"

# Color codes for terminal output
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if binaries exist
if [ ! -x baseline/sw_baseline ]; then
    echo "Error: baseline/sw_baseline not found. Please run 'make' first."
    exit 1
fi

if [ ! -x optimized/sw_optimized ]; then
    echo "Error: optimized/sw_optimized not found. Please run 'make' first."
    exit 1
fi

# Function to calculate statistics
calculate_stats() {
    local -n times=$1
    local count=${#times[@]}
    
    # Calculate average
    local sum=0
    for time in "${times[@]}"; do
        sum=$(echo "$sum + $time" | bc -l)
    done
    local avg=$(echo "scale=6; $sum / $count" | bc -l)
    
    # Sort array for median calculation
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
    unset IFS
    
    # Calculate median
    local median
    if [ $((count % 2)) -eq 0 ]; then
        local mid1=$((count / 2 - 1))
        local mid2=$((count / 2))
        median=$(echo "scale=6; (${sorted[$mid1]} + ${sorted[$mid2]}) / 2" | bc -l)
    else
        local mid=$((count / 2))
        median=${sorted[$mid]}
    fi
    
    # Calculate standard deviation
    local variance_sum=0
    for time in "${times[@]}"; do
        local diff=$(echo "$time - $avg" | bc -l)
        local sq=$(echo "$diff * $diff" | bc -l)
        variance_sum=$(echo "$variance_sum + $sq" | bc -l)
    done
    local variance=$(echo "scale=6; $variance_sum / $count" | bc -l)
    local stddev=$(echo "scale=6; sqrt($variance)" | bc -l)
    
    echo "$avg $median $stddev"
}

# Function to extract execution time from output
extract_time() {
    local output="$1"
    echo "$output" | grep "Execution time:" | awk '{print $3}'
}

# Initialize CSV file
echo "Implementation,Input Size,Run,Time (s),Average (s),Median (s),Std Dev (s)" > "$RESULTS_FILE"

# Print header
echo -e "${BOLD}========================================================================================================${NC}"
echo -e "${BOLD}                          Smith-Waterman Performance Benchmark${NC}"
echo -e "${BOLD}========================================================================================================${NC}"
printf "${BOLD}%-20s %-12s %-8s %-15s %-15s %-15s %-15s${NC}\n" \
    "Implementation" "Input Size" "Run #" "Time (s)" "Average (s)" "Median (s)" "Std Dev (s)"
echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"

# Run benchmarks
for size in "${INPUT_SIZES[@]}"; do
    # Baseline C implementation
    echo -e "${GREEN}${BOLD}Testing Baseline (C) with size ${size}...${NC}"
    baseline_times=()
    
    for i in $(seq 1 $ITERATIONS); do
        output=$(baseline/sw_baseline $size 2>&1)
        time=$(extract_time "$output")
        baseline_times+=("$time")
        
        # Print individual run
        printf "${GREEN}%-20s %-12s %-8s %-15s${NC}\n" \
            "Baseline (C)" "$size" "$i" "$time"
        
        # Save to CSV (without stats for individual runs)
        echo "Baseline (C),$size,$i,$time,,," >> "$RESULTS_FILE"
    done
    
    # Calculate and display statistics
    stats=$(calculate_stats baseline_times)
    avg=$(echo $stats | awk '{print $1}')
    median=$(echo $stats | awk '{print $2}')
    stddev=$(echo $stats | awk '{print $3}')
    
    printf "${GREEN}${BOLD}%-20s %-12s %-8s %-15s %-15s %-15s %-15s${NC}\n" \
        "Baseline (C)" "$size" "Summary" "-" "$avg" "$median" "$stddev"
    
    # Update CSV with summary
    echo "Baseline (C),$size,Summary,-,$avg,$median,$stddev" >> "$RESULTS_FILE"
    echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
    
    # Optimized C++ implementation
    echo -e "${BLUE}${BOLD}Testing Optimized (C++) with size ${size}...${NC}"
    optimized_times=()
    
    for i in $(seq 1 $ITERATIONS); do
        output=$(optimized/sw_optimized $size 2>&1)
        time=$(extract_time "$output")
        optimized_times+=("$time")
        
        # Print individual run
        printf "${BLUE}%-20s %-12s %-8s %-15s${NC}\n" \
            "Optimized (C++)" "$size" "$i" "$time"
        
        # Save to CSV (without stats for individual runs)
        echo "Optimized (C++),$size,$i,$time,,," >> "$RESULTS_FILE"
    done
    
    # Calculate and display statistics
    stats=$(calculate_stats optimized_times)
    avg=$(echo $stats | awk '{print $1}')
    median=$(echo $stats | awk '{print $2}')
    stddev=$(echo $stats | awk '{print $3}')
    
    printf "${BLUE}${BOLD}%-20s %-12s %-8s %-15s %-15s %-15s %-15s${NC}\n" \
        "Optimized (C++)" "$size" "Summary" "-" "$avg" "$median" "$stddev"
    
    # Update CSV with summary
    echo "Optimized (C++),$size,Summary,-,$avg,$median,$stddev" >> "$RESULTS_FILE"
    
    # Calculate speedup
    baseline_avg=$(calculate_stats baseline_times | awk '{print $1}')
    opt_avg=$(calculate_stats optimized_times | awk '{print $1}')
    speedup=$(echo "scale=2; $baseline_avg / $opt_avg" | bc -l)
    
    printf "${YELLOW}${BOLD}%-20s %-12s %-8s %-15s %-15s %-15s %-15s${NC}\n" \
        "Speedup" "$size" "-" "-" "${speedup}x" "-" "-"
    
    echo "Speedup,$size,-,-,${speedup}x,-,-" >> "$RESULTS_FILE"
    echo -e "${BOLD}========================================================================================================${NC}"
done

echo -e "\n${GREEN}${BOLD}Benchmark completed!${NC}"
echo -e "${GREEN}Results saved to: ${RESULTS_FILE}${NC}"
