# Build the standalone DCOPY store benchmark (Fortran driver + C kernels).
#
#   make            build ./dcopy_bench
#   make run        build and run with defaults
#   make clean      remove build products
#
# Threading is handled entirely in the Fortran driver, so only FFLAGS needs
# -fopenmp; the C kernels stay single-threaded.  Run several threads to expose
# the non-temporal-store advantage, e.g.  OMP_NUM_THREADS=4 ./dcopy_bench
#
# On x86, -march=native enables AVX/AVX-512 so dcopy_nt_ emits real streaming
# stores.  On Apple silicon build with, e.g.:
#   make CC=clang FC=gfortran \
#        CFLAGS='-O3 -mcpu=apple-m2' FFLAGS='-O3 -mcpu=apple-m2 -fopenmp'

# Use gfortran unless the user overrode FC (GNU Make otherwise defaults it to f77).
ifeq ($(origin FC),default)
FC := gfortran
endif

CC     ?= cc
CFLAGS ?= -O3 -march=native
FFLAGS ?= -O3 -march=native -fopenmp

dcopy_bench: dcopy_bench.o dcopy_kernels.o
	$(FC) $(FFLAGS) $^ -o $@

dcopy_bench.o: dcopy_bench.f90
	$(FC) $(FFLAGS) -c $< -o $@

dcopy_kernels.o: dcopy_kernels.c
	$(CC) $(CFLAGS) -c $< -o $@

run: dcopy_bench
	./dcopy_bench

clean:
	rm -f dcopy_bench dcopy_bench.o dcopy_kernels.o dcopy_bench_mod.mod

.PHONY: run clean
