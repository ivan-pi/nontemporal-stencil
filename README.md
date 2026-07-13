# Non-temporal store stencil benchmark

A micro-benchmark that measures how **non-temporal (streaming) stores** affect
the memory bandwidth of 2-D stencils on Apple Silicon, as a function of problem
size. It sweeps grid sizes and produces a bandwidth-vs-size plot comparing
ordinary stores against `__builtin_nontemporal_store` (which lowers to `STNP`
on AArch64).

Two stencils, two store variants, three thread-dispatch backends, one shared
kernel body so the comparison is apples-to-apples. Written in C++17
(`stencil.cpp`).

```
make                 # build ./stencil          (Apple: needs `brew install libomp`)
make verify-asm      # confirm STNP is actually emitted (read this section!)
./launch.sh          # sweep sizes -> results.dat
gnuplot plot.gp      # results.dat -> results.png
```

### Code layout

The hot path stays a single, readable expression while the store instruction
and the stencil shape are resolved at compile time:

* **`std::vector<double, AlignedAllocator<double>>`** — owning storage, 128-byte
  aligned so each padded row starts on a cache-line boundary.
* **`ArrayView<T>`** — a thin non-owning view; `a(x, y)` indexes `data[y*lda+x]`.
  Its data pointer is `__restrict`, which is what lets the compiler assume the
  input and output don't alias — drop it and the NT vectorisation disappears.
* **`store_at<Store>()`** — `if constexpr` picks `__builtin_nontemporal_store`
  vs. an ordinary store with no runtime branch.
* **stencils are stateless lambdas** (`stencil_jacobi`, `stencil_nine`) taking a
  generic `const auto& a`; they inline into the kernel's inner loop.
* **`apply_band<Store, F>()`** is the one generic engine; four one-line
  `extern "C"` instantiations (`k_jacobi_nt`, …) give it clean, greppable
  symbol names so a single function-pointer type serves every backend and
  `make verify-asm` can find them.

Single run:

```
./stencil -n 8192 -s nine -b omp-for      # 8192^2 nine-point, OpenMP
./stencil -n 8192 -s jacobi -b gcd        # Grand Central Dispatch backend
./stencil --help
```

---

## What it measures, and why NT can help

Each sweep updates `B` from `A` (`A[y*lda + x]`, `x` unit-stride):

* **jacobi** — 5-point: `B = ¼(N + S + E + W)`
* **nine** — 9-point isotropic binomial smoother:
  `B = 1/16 (4C + 2(N+S+E+W) + (NE+NW+SE+SW))`

A stencil reads its input once (neighbours stay resident in cache) and writes a
*separate* output array it never reads back — textbook low temporal locality on
the store stream. With ordinary stores each written line is first **read** from
DRAM (write-allocate), then evicted, so a full sweep moves roughly:

| store        | DRAM traffic | reason                                  |
|--------------|--------------|-----------------------------------------|
| ordinary     | ~3·N         | read A + write-allocate B + evict B     |
| non-temporal | ~2·N         | read A + stream B (no allocate)         |

(N = `nx·ny·8` bytes.) So once the working set spills the last-level / system
level cache (SLC), NT should give up to a **1.5× speedup**. For grids that fit
in cache, NT is often *slower* — it throws away reuse the cache would have
exploited. Finding that crossover is the point of the plot.

### Metrics reported

* `eff_gbs` — **effective bandwidth**, `2·N / time`. Counts only useful bytes
  (A read once + B written once), identical for both stores, so it is the fair
  yardstick: NT wins by finishing sooner. This is the primary plotted quantity.
* `dram_gbs` — **modelled DRAM traffic**, `3·N/time` (std) or `2·N/time` (nt).
  Both curves should flatten near the memory-controller ceiling at large N,
  confirming the kernel is memory-bound.
* `mlups` — million lattice-point updates per second; no traffic assumptions.

Timing takes the **minimum of several trials** (least perturbed by OS jitter),
with the sweep count auto-sized to a ~50 ms budget per trial after warm-up.
Every run first **verifies** its output against a serial reference and prints
`max|err|` (expect `0.00e+00`).

---

## How to dispatch a stencil across threads

This is the crux of your question, so, concretely:

> **Parallelise the outer (strided) loop only, with a static, one-contiguous-
> band-per-thread decomposition. Do _not_ collapse the loops, and do _not_ use
> dynamic scheduling.**

```cpp
#pragma omp parallel for schedule(static)      // the recommended form
for (size_t y = 1; y < ny - 1; ++y)
    for (size_t x = 1; x < nx - 1; ++x)        // full unit-stride row, vectorised
        store_at<S>(&out(x, y), stencil(in, x, y));
```

Reasoning:

* **Split the strided dimension, keep whole rows intact.** With `A[y*lda + x]`
  a band of `y` values is one large *contiguous* slab of memory. That is exactly
  what streaming stores want: each thread writes long runs of full 128-byte
  cache lines, back to back. Keeping the entire unit-stride `x` row inside one
  thread also lets it vectorise cleanly.

* **`collapse(2)` is actively harmful here.** Collapsing linearises the `(y,x)`
  space and hands each thread a chunk that can *start and end mid-row*. Two
  threads then share the cache line straddling their boundary, so neither writes
  all 128 bytes of it — precisely the case the Optimization Guide warns kills NT
  performance ("all 128 bytes ... written by non-temporal stores within a brief
  window"). It also breaks the clean full-row vectorisation. You only need
  `collapse` when the outer loop has too few iterations to fill the threads;
  a stencil's `ny` is thousands of rows, so there is already ample parallelism.

* **Static beats dynamic for a uniform workload.** Every row costs the same, so
  dynamic scheduling buys you nothing but overhead — and it hands out scattered,
  unpredictable chunks that wreck streaming locality and can bounce cache lines
  between cores. Static gives each thread one contiguous band with zero runtime
  coordination. (`schedule(dynamic)` only pays off under genuine load
  imbalance, which this kernel does not have.)

The `omp-for` backend uses `schedule(runtime)` so you can **measure this
yourself** without recompiling:

```
OMP_SCHEDULE=static  ./stencil -n 8192 -b omp-for     # recommended
OMP_SCHEDULE="dynamic,64" ./stencil -n 8192 -b omp-for  # watch it regress
```

`launch.sh` exports `OMP_SCHEDULE=static`, `OMP_PROC_BIND=true`,
`OMP_PLACES=cores` by default.

`omp-spmd` is the explicit-SPMD equivalent (manual `block_decompose`, one band
per thread). It produces the *same* memory-access pattern as `omp-for` +
`static` and performs identically — use whichever you find clearer.

### The Apple Silicon P-core / E-core caveat

An equal static split assumes equal cores, but Apple Silicon is heterogeneous:
the efficiency cores have far less bandwidth and would become stragglers that
the performance cores wait on. For a clean bandwidth number, **keep the work on
the P-cores**: run with `threads = number of P-cores` (M2 Pro: 6 or 8 depending
on the bin — `launch.sh` defaults to `THREADS=6`). The benchmark reads the
P-core count from `sysctl hw.perflevel0.physicalcpu` for its default.

This is one place GCD has an edge: a high-QoS dispatch queue targets the P-cores
more reliably than OpenMP thread placement does on macOS, where affinity control
is limited.

---

## Grand Central Dispatch backend (`-b gcd`)

The `gcd` backend uses `dispatch_apply_f` as a parallel for-loop, dispatching
**one contiguous band per iteration** (band count = `-t`, default = P-cores) so
each invocation streams over its own large slab — the same decomposition as the
OpenMP path:

```cpp
dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
dispatch_apply_f(nbands, q, &ctx, gcd_band);   // gcd_band computes band i
```

`dispatch_apply_f` takes a plain `void(*)(void*, size_t)`, so a capturing
lambda can't be passed directly — the backend uses a small context struct, as
in the C API. If you prefer, the block form
`dispatch_apply(nbands, q, ^(size_t b){ … })` captures cleanly and is a one-line
swap in `run_gcd()`.

Notes:

* Dispatch **`nbands ≈ core count`**, not one iteration per row. `dispatch_apply`
  load-balances internally, but feeding it millions of tiny row-iterations wastes
  it on per-block overhead and fragments the streaming runs. A handful of large
  contiguous bands is what you want for a bandwidth kernel.
* `QOS_CLASS_USER_INITIATED` keeps the work on performance cores. The Apple docs
  recommend `DISPATCH_APPLY_AUTO`, which instead infers the queue/QoS from the
  calling context — fine for app code, but the explicit high-QoS queue is more
  controlled for benchmarking. Swap it in `run_gcd()` if you want to compare.
* GCD is compiled in only with `-DUSE_GCD` (the Makefile sets it on macOS).
  libdispatch ships in libSystem, so no extra link flags are needed.

---

## ⚠️ The compiler does **not** guarantee STNP — verify it

The Optimization Guide's fine print bites hard here: *"the compiler does not
guarantee insertion of non-temporal stores."* On AArch64, `STNP` is a **pair**
instruction, so clang keeps the non-temporal hint only when it can pair two
adjacent vector stores from an unrolled loop. Under higher register pressure —
**the 9-point kernel is the trap** — the default heuristic fails to pair and
*silently downgrades to an ordinary `STP`*. Your "nt" run then measures exactly
the same thing as "std", which is very likely why earlier results looked
inconclusive.

This benchmark works around it with an explicit interleave hint on every kernel:

```cpp
#pragma clang loop interleave_count(4)   // forces the pairing STNP needs
```

With that, both the 5-point and 9-point `_nt` kernels emit `STNP` under
`-O3 -ffast-math -fopenmp`. **Always confirm on your machine:**

```
$ make verify-asm
STNP (non-temporal pair store) count per kernel:
  k_jacobi_std   0
  k_jacobi_nt    4      <- nonzero: the hint survived
  k_nine_std     0
  k_nine_nt      4      <- nonzero: the hint survived
```

If a `_nt` kernel shows `0`, its "nt" numbers are meaningless — bump the
interleave count, or fall back to a hand-written NEON `stnp` intrinsic/asm for
that kernel. The program also prints a runtime warning if it was built with a
compiler that lacks the builtin entirely.

---

## Command-line reference

```
-n, --size N        square grid, nx = ny = N          (default 4096)
    --nx N / --ny N rectangular grid
-s, --stencil NAME  jacobi | nine                     (default jacobi)
-b, --backend NAME  omp-for | omp-spmd | gcd           (default omp-for)
    --store WHICH   std | nt | both                   (default both)
-t, --threads N     worker threads / GCD bands         (default P-cores)
-i, --iters N       max timed sweeps per trial         (default 200)
    --trials N      timed trials, minimum reported     (default 5)
    --lda N         row stride (default nx -> mult of 16, 128 B aligned)
-q, --quiet         emit only the CSV DATA line(s)
```

`launch.sh` env knobs: `THREADS`, `STENCILS`, `BACKENDS`, `SIZES`, `DAT`,
`OMP_SCHEDULE`.

## Portability

C++17; the Makefile builds with `clang++` + Homebrew libomp + GCD on macOS and
`g++` on Linux. It also builds and runs on non-Apple platforms for functional
testing: without clang's builtin, `nt` falls back to an ordinary store (flagged
at runtime); without OpenMP or GCD the backends run serially (also flagged).
Correctness is verified on every platform. Only Apple Silicon exercises the real
`STNP` path.
