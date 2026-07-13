#!/bin/bash
set -e

export OMP_NUM_THREADS=6
EXEC="./jacobi"
DAT_FILE="cache_wall.dat"
ITERATIONS=100

if [ ! -f "$EXEC" ]; then
    echo "Compiling..."
    clang -O3 -Xpreprocessor -fopenmp -lomp -ffast-math -mcpu=apple-m2 nontemporal_stencil.c -o $EXEC
fi

echo "# Kernel NX NY Iterations Time(us/step) Bandwidth(GB/s)" > "$DAT_FILE"
echo "Running High-Resolution Cache Wall Sweep..."
echo "------------------------------------------------------"

# Generate 4 data points per octave (2^8 to 2^14.5)
SIZES=$(awk 'BEGIN {for(i=8; i<=14.5; i+=0.25) printf "%d ", 2^i}')

for SIZE in $SIZES; do
    echo "Benchmarking grid size: ${SIZE} x ${SIZE}..."

    $EXEC $SIZE $SIZE $ITERATIONS | awk -F',' '/^DATA/ {
        kernel = $2
        gsub(/^[ \t]+|[ \t]+$/, "", kernel)
        printf "%-15s %8d %8d %8d %15.2f %15.2f\n", kernel, $3, $4, $5, $6, $7
    }' >> "$DAT_FILE"
done

echo "------------------------------------------------------"
echo "Benchmark complete. Results saved to $DAT_FILE."