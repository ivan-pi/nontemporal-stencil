# save as plot.gp and run: gnuplot -p plot.gp
set terminal qt size 1000,600 font "Helvetica,11"

# Enable a multiplot layout to show Bandwidth and Runtime side-by-side
set multiplot layout 1,2 title "Jacobi Stencil Performance on Apple M2 Pro" font ",14"

# ----------------------------------------------------
# Plot 1: Bandwidth (GB/s)
# ----------------------------------------------------
set title "Effective Bandwidth"
set xlabel "Grid Size (NX = NY)"
set ylabel "Bandwidth (GB/s)"
set logscale x 2
set key top right box opaque
set grid xtics ytics mxtics mytics lt 0 lw 1, lt 0 lw 0.5

# Highlight the ~32MB L2 Cache boundary (NX ~ 1414)
set arrow from 1414, graph 0 to 1414, graph 1 nohead dt 2 lc rgb "red"
set label "L2 Boundary" at 1414, graph 0.05 right offset -1,0 textcolor rgb "red"

plot "< awk '$1==\"Standard\"' cache_wall.dat" \
     using 2:6 with linespoints lw 1.5 pt 7 ps 0.6 title "Standard Stores", \
     "< awk '$1==\"Non-Temporal\"' cache_wall.dat" \
     using 2:6 with linespoints lw 1.5 pt 5 ps 0.6 title "Non-Temporal Stores"

# ----------------------------------------------------
# Plot 2: Runtime (Microseconds per step)
# ----------------------------------------------------
set title "Iteration Runtime"
set xlabel "Grid Size (NX = NY)"
set ylabel "Time (microseconds per step)"
set logscale x 2
set logscale y 10
set key top left box opaque

plot "< awk '$1==\"Standard\"' cache_wall.dat" \
     using 2:5 with linespoints lw 1.5 pt 7 ps 0.6 title "Standard Stores", \
     "< awk '$1==\"Non-Temporal\"' cache_wall.dat" \
     using 2:5 with linespoints lw 1.5 pt 5 ps 0.6 title "Non-Temporal Stores"

unset multiplot