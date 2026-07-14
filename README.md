# nontemporal-stencil

Experiments in **non-temporal (streaming) stores** — writes that go straight to
memory, bypassing the caches and skipping the read-for-ownership (RFO) traffic
an ordinary store incurs. They pay off when a large, write-once result would
otherwise evict useful data from cache and when memory bandwidth is the
bottleneck.

Two benchmarks live here:

| Benchmark | Files | What it measures |
|-----------|-------|------------------|
| **DCOPY store test** | `dcopy_bench.f90`, `dcopy_kernels.c`, `Makefile` | A BLAS-1 `y = x` copy done four ways, isolating the store type |
| **Jacobi stencil** | `jacobi.c`, `launch.sh`, `plot.gp` | A 5-point Jacobi sweep with standard vs. non-temporal store kernels |

---

## DCOPY store test

A standalone comparison of standard stores versus non-temporal stores on the
simplest possible kernel, a unit-stride Level-1 BLAS `DCOPY` (`y(1:n) = x(1:n)`).
The benchmark driver is written in Fortran; the store kernels are in C and are
reached through the classic Fortran name-mangling ABI (trailing underscore,
by-reference arguments) — i.e. `dcopy_store_`, `dcopy_memcpy_`, `dcopy_nt_` —
just like calling a legacy BLAS library, rather than via `iso_c_binding`.

Four variants are timed under identical conditions:

| Variant | How the copy is done |
|---------|----------------------|
| `array syntax` | Fortran `y = x` |
| `memcpy` | C library `memcpy` (`dcopy_memcpy_`) |
| `standard store` | vectorised **temporal** stores (`dcopy_store_`) |
| `non-temporal store` | vectorised **streaming** stores (`dcopy_nt_`) |

`dcopy_store_` and `dcopy_nt_` are byte-for-byte identical apart from the store
instruction (`vmovapd`/`vmovupd` vs. `vmovntpd`), so the difference between them
is purely the effect being studied. On x86 the kernels use SSE2/AVX intrinsics;
on other targets Clang's `__builtin_nontemporal_store` is used (it lowers to
e.g. `stnp` on Apple silicon), with a plain-store fallback everywhere else.

### Build & run

```sh
make                      # builds ./dcopy_bench
OMP_NUM_THREADS=4 ./dcopy_bench          # defaults: 8,000,000 elements, 50 reps
OMP_NUM_THREADS=4 ./dcopy_bench 16000000 100   # custom n and repetitions
```

Threading is handled entirely in the Fortran driver (one `!$omp parallel do`
that hands contiguous chunks to whichever kernel is being timed), so only the
Fortran side needs `-fopenmp` and the C kernels stay single-threaded. Built
without `-fopenmp` the driver simply runs serially. Non-temporal stores show
their advantage most clearly once several threads are saturating memory
bandwidth, so sweep the thread count:

```sh
for t in 1 2 4; do OMP_NUM_THREADS=$t ./dcopy_bench; done
```

On Apple silicon:

```sh
make CC=clang FC=gfortran \
     CFLAGS='-O3 -mcpu=apple-m2' FFLAGS='-O3 -mcpu=apple-m2 -fopenmp'
```

### Reading the output

Each line reports best-of-N time per copy (the STREAM convention), the effective
bandwidth (counting one read + one write per element, `2 * n * 8` bytes), and a
`max|err|` correctness check that must be `0` for a pure copy. A machine-readable
`DATA,<label>,<n>,<us>,<gbps>,<err>` line accompanies each result for scripting a
size or thread sweep.

Representative run (4 threads, 61 MB per vector, shared cloud Xeon — absolute
numbers are host- and load-dependent, the *ratio* is the point):

```
array syntax             65.1 GB/s
memcpy                   64.7 GB/s
standard store           57.5 GB/s
non-temporal store      116.3 GB/s     <- ~2x the cached-store variants
```

Non-temporal stores win here because the destination is far larger than cache:
the ordinary variants pay for reading `x`, fetching `y` for ownership, and
writing `y` (~3 memory transfers), while the streaming variant only reads `x`
and writes `y` (~2). The gap widens with more threads and shrinks (or reverses)
for data that fits in cache or for a single unsaturated thread.

---

## Jacobi stencil

`jacobi.c` runs a 5-point Jacobi stencil (column-major, OpenMP, with a
configurable 2-D thread topology and leading-dimension padding) using two
kernels that differ only in their store type — an ordinary store and a
non-temporal store (via `__builtin_nontemporal_store` on Clang). It reports
per-step time and effective bandwidth so the two can be compared as the grid
grows past the cache.

```sh
clang -O3 -fopenmp -ffast-math -mcpu=apple-m2 jacobi.c -o jacobi
./jacobi --help                 # list all arguments
./launch.sh                     # sweep grid sizes -> cache_wall.dat
gnuplot -p plot.gp              # plot bandwidth & runtime vs. grid size
```

`launch.sh` sweeps grid sizes across the cache hierarchy and writes
`cache_wall.dat`; `plot.gp` renders bandwidth and runtime against grid size,
highlighting the cache boundary where non-temporal stores start to pay off.
