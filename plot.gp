# Bandwidth vs. problem size for standard vs. non-temporal stores.
#   gnuplot plot.gp            -> results.png
# Reads results.dat produced by launch.sh. Columns:
#   1 stencil 2 backend 3 store 4 nx 5 ny 6 threads 7 iters 8 trials
#   9 time_us 10 eff_gbs 11 dram_gbs 12 mlups

set terminal pngcairo size 1200,820 font "Helvetica,12"
set output "results.png"

set multiplot layout 2,2 \
    title "2-D stencil: standard vs. non-temporal stores (Apple M2 Pro)" font ",15"

set logscale x 2
set xlabel "grid size N (N x N)"
set grid xtics ytics mxtics lc rgb "#cccccc"
set key top left box opaque samplen 1.5

# --- Approximate cache boundary: where the two N x N double arrays stop fitting
#     in the shared last-level cache. Edit slc_mb for your machine's SLC size.
slc_mb = 24.0
slc_n  = sqrt(slc_mb*1e6 / (2*8))     # 2 arrays x 8 bytes/double
set arrow 1 from slc_n, graph 0 to slc_n, graph 1 nohead dt 2 lc rgb "red"
set label 1 "2 arrays > SLC" at slc_n, graph 0.94 right offset -0.5,0 tc rgb "red"

std(s) = sprintf("< awk '$1==\"%s\" && $3==\"std\"' results.dat", s)
nt(s)  = sprintf("< awk '$1==\"%s\" && $3==\"nt\"'  results.dat", s)

# Row 1: effective (algorithmic) bandwidth = 2N moved / time. This is the
# apples-to-apples yardstick; nt should pull ahead once N passes the cache.
set ylabel "effective bandwidth (GB/s)"

set title "Jacobi 5-point"
plot std("jacobi") using 4:10 w lp lw 2 pt 7 ps 0.7 title "standard", \
     nt("jacobi")  using 4:10 w lp lw 2 pt 5 ps 0.7 title "non-temporal"

set title "9-point"
plot std("nine") using 4:10 w lp lw 2 pt 7 ps 0.7 title "standard", \
     nt("nine")  using 4:10 w lp lw 2 pt 5 ps 0.7 title "non-temporal"

# Row 2: modelled DRAM traffic bandwidth = (std 3N | nt 2N) / time. Both curves
# should flatten near the memory-controller ceiling at large N, confirming the
# kernel is memory-bound -- nt just reaches the same ceiling moving less data.
set ylabel "modelled DRAM traffic (GB/s)"
set key bottom right

set title "Jacobi 5-point -- traffic"
plot std("jacobi") using 4:11 w lp lw 2 pt 7 ps 0.7 title "standard (3N)", \
     nt("jacobi")  using 4:11 w lp lw 2 pt 5 ps 0.7 title "non-temporal (2N)"

set title "9-point -- traffic"
plot std("nine") using 4:11 w lp lw 2 pt 7 ps 0.7 title "standard (3N)", \
     nt("nine")  using 4:11 w lp lw 2 pt 5 ps 0.7 title "non-temporal (2N)"

unset multiplot
set output
