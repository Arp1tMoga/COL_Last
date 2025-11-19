#!/bin/bash

# A script to benchmark different program versions with various inputs.
# It calculates the average, median, and standard deviation of execution times
# over multiple runs. Results are displayed in the console and saved to a CSV file.

# --- Configuration ---
# Programs to be benchmarked.
programs=("python baseline/sw_baseline.py" "./baseline/c_sw_baseline" "./optimized/sw_optimized")

# A range of inputs to test each program with.
inputs=(512 1024 2048 4096 8192)

# The number of times to run each program with each input to get a stable average.
runs=10

# The name of the file to save the results.
output_csv="results.csv"

# --- Function to calculate statistics ---
# Takes a list of execution times and calculates the average, median, and standard deviation.
calculate_stats() {
    local times=("$@")
    local n=${#times[@]}
    local sum=0

    # Sort times numerically to find the median.
    local sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))

    # Calculate median
    local median
    if (( n % 2 == 0 )); then
        # If even number of elements, average the middle two.
        local mid1=${sorted_times[$((n/2-1))]}
        local mid2=${sorted_times[$((n/2))]}
        median=$(echo "scale=6; ($mid1 + $mid2) / 2" | bc)
    else
        # If odd number of elements, the middle one is the median.
        median=${sorted_times[$((n/2))]}
    fi

    # Calculate sum for the average
    for t in "${times[@]}"; do
        sum=$(echo "scale=6; $sum + $t" | bc)
    done
    local avg=$(echo "scale=6; $sum / $n" | bc)

    # Calculate variance and standard deviation
    local variance=0
    for t in "${times[@]}"; do
        local diff=$(echo "scale=6; $t - $avg" | bc)
        variance=$(echo "scale=6; $variance + ($diff * $diff)" | bc)
    done
    variance=$(echo "scale=6; $variance / $n" | bc)
    local stddev=$(echo "scale=6; sqrt($variance)" | bc)

    echo "$avg $median $stddev"
}

# --- Main Execution ---

# Initialize CSV file with a header
echo "Program,Input,Average,Median,StdDev" > "$output_csv"

# Print table header to the console
printf "%-35s %-10s %-15s %-15s %-15s\n" "Program" "Input" "Average (s)" "Median (s)" "StdDev (s)"
printf "%s\n" "------------------------------------------------------------------------------------------"

# Loop through each program
for prog in "${programs[@]}"; do
    # Loop through each input size
    for input in "${inputs[@]}"; do
        declare -a times=() # Array to store execution times for the current set of runs

        # Run the program multiple times to gather data
        for ((i=1; i<=runs; i++)); do
            # Use date for high-resolution timing (seconds.nanoseconds)
            start_time=$(date +%s.%N)
            # Execute the program, redirecting its output to /dev/null
            $prog $input > /dev/null 2>&1
            end_time=$(date +%s.%N)

            # Calculate the execution time and add it to our list
            time_taken=$(echo "scale=6; $end_time - $start_time" | bc)
            times+=($time_taken)
        done

        # Calculate statistics for the collected times
        stats=($(calculate_stats "${times[@]}"))
        avg_time=${stats[0]}
        median_time=${stats[1]}
        stddev_time=${stats[2]}

        # Extract a clean program name for display
        prog_name=$(basename "$prog")

        # Print formatted results to the console as they are completed
        printf "%-35s %-10s %-15.6f %-15.6f %-15.6f\n" "$prog" "$input" "$avg_time" "$median_time" "$stddev_time"

        # Append results to the CSV file
        echo "$prog_name,$input,$avg_time,$median_time,$stddev_time" >> "$output_csv"
    done
done

echo ""
echo "Benchmarking complete. Results saved to $output_csv"