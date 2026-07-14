# Non-temporal store stencil benchmark

Measures how non-temporal (streaming) stores affect the memory bandwidth of a
5-point Jacobi and a 9-point stencil versus problem size on Apple Silicon, and
plots bandwidth vs. grid size for ordinary vs. non-temporal stores.

```
make                 # build ./stencil     (Apple: brew install libomp first)
make verify-asm      # confirm STNP is actually emitted  (see below)
./launch.sh          # size sweep -> results.dat
gnuplot plot.gp      # results.dat -> results.png
./stencil --help     # all options
```

## Why non-temporal stores

A stencil reads its input once (neighbours stay cached) and writes a separate
output it never reads back. An ordinary store first reads each output line from
DRAM (write-allocate), so a sweep moves ~3N; a non-temporal store skips that, so
it moves ~2N:

| store        | DRAM traffic | note                                 |
|--------------|--------------|--------------------------------------|
| ordinary     | ~3·N         | read A + write-allocate B + evict B  |
| non-temporal | ~2·N         | read A + stream B                    |

So NT can be up to **1.5× faster once the working set spills cache**, and
*slower* while it still fits (it throws away reuse). The plot shows the
crossover. Reported metrics: `eff_gbs` (effective 2N/time, the fair yardstick),
`dram_gbs` (modelled 3N or 2N traffic), `mlups`.

## Dispatching a stencil across threads

**Parallelise the outer (strided) loop only, one contiguous band of rows per
thread — the default `static` schedule. Don't `collapse(2)`, don't use
`dynamic`.**

With `a(x, y) == data[y*lda + x]`, a band of rows is one large contiguous slab,
which is what streaming stores want: long runs of full 128-byte cache lines.

* `collapse(2)` hands a thread a chunk that starts/ends mid-row, so two threads
  share the boundary cache line and neither writes all 128 bytes of it — the
  case the Optimization Guide says kills NT. You don't need it: `ny` already has
  plenty of parallelism.
* `dynamic` only helps with load imbalance, which a uniform stencil doesn't
  have; here it just adds overhead and scatters the streaming runs.

Keep the work on the **performance cores** (threads = #P-cores; the default is
read from `sysctl hw.perflevel0.physicalcpu`) — E-cores have less bandwidth and
would just be stragglers under an equal split. GCD's high-QoS queue targets
P-cores more reliably than OpenMP placement does on macOS.

## ⚠️ The compiler does not guarantee STNP — verify it

On AArch64 `STNP` is a *pair* instruction, so clang keeps the non-temporal hint
only when it can pair two adjacent vector stores. Under register pressure — the
9-point kernel especially — it silently falls back to an ordinary `STP`, and
your "nt" run then measures the same thing as "std" (a likely cause of
inconclusive results). The kernels carry `#pragma clang loop interleave_count(4)`
to force the pairing. **Always confirm on your machine:**

```
$ make verify-asm
  k_jacobi_nt    4      <- nonzero: the hint survived
  k_nine_nt      4
  k_jacobi_std   0
  k_nine_std     0
```

If an `_nt` kernel shows `0`, its numbers are meaningless — raise the interleave
count or drop to a hand-written NEON `stnp`.

## Grand Central Dispatch backend (`-b gcd`)

`run_gcd` uses `dispatch_apply_f` over `nbands ≈ #cores` contiguous bands on a
`QOS_CLASS_USER_INITIATED` global queue (kernels have a plain C ABI, so no block
needed). Dispatch a few large bands, not one iteration per row, or the per-block
overhead dominates and the streaming runs fragment.
