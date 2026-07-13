#!/usr/bin/env bash
# Sweep grid size and record bandwidth for each stencil / store variant.
# Output is a whitespace-separated table in $DAT, ready for plot.gp.
#
# Tunables (environment overrides):
#   THREADS   worker threads / GCD bands           (default 6 = M2 Pro P-cores)
#   STENCILS  space-separated: jacobi nine          (default both)
#   BACKENDS  space-separated: omp-for omp-spmd gcd  (default omp-for)
#   SIZES     space-separated grid sizes             (default 2^7 .. ~2^13.75)
#   DAT       output file                            (default results.dat)
set -euo pipefail
cd "$(dirname "$0")"

BIN=./stencil
DAT=${DAT:-results.dat}
THREADS=${THREADS:-6}
STENCILS=${STENCILS:-"jacobi nine"}
BACKENDS=${BACKENDS:-"omp-for"}

# Pin the streaming workload to the performance cores.
export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"
export OMP_PLACES="${OMP_PLACES:-cores}"

[ -x "$BIN" ] || make

# 4 points per octave from 128 up to ~13000.
SIZES=${SIZES:-$(awk 'BEGIN{for(i=7;i<=13.75;i+=0.25) printf "%d ", 2**i}')}

echo "# stencil backend store nx ny threads iters trials time_us eff_gbs dram_gbs mlups" > "$DAT"
echo "Sweeping: stencils=[$STENCILS] backends=[$BACKENDS] threads=$THREADS"
echo "-------------------------------------------------------------------------------"
for s in $STENCILS; do
  for b in $BACKENDS; do
    for N in $SIZES; do
      printf "  %-7s %-9s N=%-6s ... " "$s" "$b" "$N"
      line=$("$BIN" -n "$N" -s "$s" -b "$b" -t "$THREADS" --store both -q)
      # Strip the DATA prefix, comma -> space, into the table.
      echo "$line" | awk -F, '/^DATA/{out=$2; for(i=3;i<=NF;i++) out=out" "$i; print out}' >> "$DAT"
      # Progress: store=eff_gbs ($4=store, $11=eff_gbs).
      echo "$line" | awk -F, '/^DATA/{printf "%s=%.1f GB/s  ", $4, $11} END{print ""}'
    done
  done
done
echo "-------------------------------------------------------------------------------"
echo "Wrote $DAT. Plot with:  gnuplot plot.gp   (produces results.png)"
