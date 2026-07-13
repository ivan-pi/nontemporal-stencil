// ===========================================================================
//  stencil.c  --  Non-temporal store micro-benchmark for 2-D stencils
// ===========================================================================
//
//  Measures the effect of non-temporal (streaming) stores on the effective
//  memory bandwidth of two 2-D stencils, as a function of problem size, on
//  Apple Silicon (and portably elsewhere for functional testing).
//
//  Stencils (both are averaging operators, so the field stays bounded over
//  arbitrarily many iterations -- no overflow during long timing runs):
//    * jacobi : 5-point   Anew = 1/4 (N + S + E + W)
//    * nine   : 9-point   isotropic binomial smoother,
//                 Anew = 1/16 ( 4*C + 2*(N+S+E+W) + (NE+NW+SE+SW) )
//
//  Store variants:
//    * std : ordinary stores (write-allocate: the line is read from DRAM
//            before being overwritten, then evicted -> ~3 array passes).
//    * nt  : non-temporal stores via __builtin_nontemporal_store, which skip
//            the write-allocate read -> ~2 array passes. On AArch64 this
//            lowers to STNP; on other targets / non-clang it degrades to an
//            ordinary store (see HAVE_NT_STORE) so results stay correct.
//
//  Why NT can help: a stencil reads its input once (neighbours stay in cache)
//  and writes a distinct output array with no temporal reuse. For a large
//  grid the ordinary store's write-allocate read is pure waste; NT removes it,
//  cutting modelled DRAM traffic 3N -> 2N, i.e. up to a 1.5x speedup once the
//  working set spills the last-level cache. For small grids that fit in cache,
//  NT can be *slower* because it defeats reuse -- the crossover is exactly what
//  the bandwidth-vs-size plot is meant to reveal.
//
//  Thread dispatch backends (see run_backend):
//    * omp-for  : #pragma omp parallel for over rows, schedule(runtime).
//                 The recommended, idiomatic form. Set OMP_SCHEDULE=static
//                 (the default here) for one contiguous band per thread.
//    * omp-spmd : explicit parallel region + manual block decomposition,
//                 one contiguous band per thread (equivalent memory pattern).
//    * gcd      : Apple Grand Central Dispatch dispatch_apply_f over bands.
//
//  All backends call the SAME per-band kernel, so the arithmetic and the store
//  instruction are identical across them -- the numbers are directly
//  comparable and share one correctness check.
//
//  Storage is 2-D with x the unit-stride (contiguous) index and y strided by
//  lda: A[y*lda + x]. Work is split across the strided (y) dimension so every
//  thread owns a large contiguous slab of memory -- ideal for streaming
//  stores, which want to write whole 128-byte cache lines back to back.
// ===========================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_GCD
#include <dispatch/dispatch.h>
#define HAVE_GCD 1
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// ---------------------------------------------------------------------------
//  Non-temporal store selection
// ---------------------------------------------------------------------------
// Clang exposes the LLVM builtin; on AArch64 it becomes STNP. Everywhere else
// we fall back to a plain store so the benchmark still runs and verifies
// (the "nt" numbers just won't differ from "std" -- flagged at runtime).
#if defined(__clang__) && defined(__has_builtin)
#  if __has_builtin(__builtin_nontemporal_store)
#    define HAVE_NT_STORE 1
#  endif
#endif

#ifdef HAVE_NT_STORE
#  define NT_STORE(v, p) __builtin_nontemporal_store((v), (p))
#else
#  define NT_STORE(v, p) (*(p) = (v))
#endif
#define STD_STORE(v, p) (*(p) = (v))

// AArch64's non-temporal store (STNP) is a *pair* instruction, so clang keeps
// the non-temporal hint only when it can pair two adjacent vector stores from
// an unrolled loop. Under higher register pressure -- notably the 9-point
// kernel -- the default heuristic fails to pair and SILENTLY downgrades the
// hint to an ordinary STP, so "nt" measures the same thing as "std". An
// explicit interleave hint makes STNP emission reliable for both stencils
// (confirm with `make verify-asm`). gcc ignores the pragma and has no NT store
// anyway. Applied to every kernel so std and nt share one loop structure and
// differ only in the store instruction.
#if defined(__clang__)
#  define DO_PRAGMA(x) _Pragma(#x)
#  define STENCIL_LOOP_HINT DO_PRAGMA(clang loop interleave_count(4))
#else
#  define STENCIL_LOOP_HINT
#endif

// ---------------------------------------------------------------------------
//  Small helpers
// ---------------------------------------------------------------------------
static double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Count of performance ("P") cores, or 0 if unknown. The ideal default worker
// count for a bandwidth benchmark on Apple Silicon: E-cores have far less
// bandwidth and, with an equal static split, just become stragglers.
static int perf_cores(void) {
#ifdef __APPLE__
    int n = 0;
    size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, NULL, 0) == 0 && n > 0)
        return n;
#endif
    return 0;
}

static int default_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    int p = perf_cores();
    if (p > 0) return p;
    long m = sysconf(_SC_NPROCESSORS_ONLN);
    return m > 0 ? (int)m : 1;
#endif
}

#if defined(__GNUC__)   // gcc and clang
#  define MAYBE_UNUSED __attribute__((unused))
#else
#  define MAYBE_UNUSED
#endif

// Split [0, n) into `parts` contiguous, near-equal blocks; return block i.
// (unused by the serial fallback build; used by the omp-spmd and gcd backends.)
static MAYBE_UNUSED void band_bounds(size_t n, int parts, int i, size_t *start, size_t *end) {
    size_t q = n / (size_t)parts;
    size_t r = n % (size_t)parts;
    size_t ii = (size_t)i;
    *start = ii * q + (ii < r ? ii : r);
    *end   = *start + q + (ii < r ? 1 : 0);
}

// ---------------------------------------------------------------------------
//  Kernels
// ---------------------------------------------------------------------------
typedef struct {
    size_t lda;
    const double *restrict A;   // input
    double *restrict B;         // output
} kctx;

// A kernel updates the interior block [xs,xe) x [ys,ye) of B from A.
typedef void (*band_kernel)(size_t xs, size_t xe, size_t ys, size_t ye, const kctx *k);

// Stencil arithmetic, single-sourced so std and nt variants are bit-identical.
#define JACOBI_EXPR(A, lda, x, y)                                              \
    ( 0.25 * ( A[(y - 1) * (lda) + (x)] + A[(y + 1) * (lda) + (x)]             \
             + A[(y) * (lda) + (x - 1)] + A[(y) * (lda) + (x + 1)] ) )

#define NINE_EXPR(A, lda, x, y)                                                \
    ( (1.0 / 16.0) * ( 4.0 *  A[(y) * (lda) + (x)]                             \
        + 2.0 * ( A[(y - 1) * (lda) + (x)]     + A[(y + 1) * (lda) + (x)]       \
                + A[(y) * (lda) + (x - 1)]     + A[(y) * (lda) + (x + 1)] )     \
        +       ( A[(y - 1) * (lda) + (x - 1)] + A[(y - 1) * (lda) + (x + 1)]   \
                + A[(y + 1) * (lda) + (x - 1)] + A[(y + 1) * (lda) + (x + 1)] ) ) )

// Stamp out one kernel from a stencil expression and a store operation. The
// inner (unit-stride) loop is marked `omp simd`; keeping the whole row inside
// one thread is what lets NT stores cover full cache lines.
#define GEN_BAND_KERNEL(NAME, EXPR, STORE)                                     \
    static void NAME(size_t xs, size_t xe, size_t ys, size_t ye,               \
                     const kctx *k) {                                          \
        const double *restrict A = k->A;                                       \
        double *restrict B = k->B;                                             \
        const size_t lda = k->lda;                                             \
        for (size_t y = ys; y < ye; ++y) {                                     \
            _Pragma("omp simd") STENCIL_LOOP_HINT                              \
            for (size_t x = xs; x < xe; ++x) {                                 \
                double v = EXPR(A, lda, x, y);                                 \
                STORE(v, &B[y * lda + x]);                                     \
            }                                                                  \
        }                                                                      \
    }

GEN_BAND_KERNEL(k_jacobi_std, JACOBI_EXPR, STD_STORE)
GEN_BAND_KERNEL(k_jacobi_nt,  JACOBI_EXPR, NT_STORE)
GEN_BAND_KERNEL(k_nine_std,   NINE_EXPR,   STD_STORE)
GEN_BAND_KERNEL(k_nine_nt,    NINE_EXPR,   NT_STORE)

typedef enum { ST_JACOBI, ST_NINE } stencil_kind;
typedef enum { SB_STD, SB_NT } store_kind;
typedef enum { BK_OMP_FOR, BK_OMP_SPMD, BK_GCD } backend_kind;

static band_kernel select_kernel(stencil_kind s, store_kind st) {
    if (s == ST_JACOBI) return st == SB_NT ? k_jacobi_nt : k_jacobi_std;
    else                return st == SB_NT ? k_nine_nt   : k_nine_std;
}

// ---------------------------------------------------------------------------
//  Backends
// ---------------------------------------------------------------------------
// Worksharing loop over rows. schedule(runtime) lets you compare scheduling
// policies with OMP_SCHEDULE at no code cost: `static` gives each thread one
// contiguous band of rows (recommended); `dynamic` scatters rows and is
// measurably slower here (scheduling overhead + broken streaming locality).
static void run_omp_for(band_kernel kern, size_t nx, size_t ny,
                        const kctx *k, int nthreads) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(runtime) num_threads(nthreads)
    for (size_t y = 1; y < ny - 1; ++y)
        kern(1, nx - 1, y, y + 1, k);
#else
    (void)nthreads;
    kern(1, nx - 1, 1, ny - 1, k);
#endif
}

// SPMD: explicit parallel region, one contiguous band of rows per thread.
static void run_omp_spmd(band_kernel kern, size_t nx, size_t ny,
                         const kctx *k, int nthreads) {
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
    {
        int P = omp_get_num_threads();
        int t = omp_get_thread_num();
        size_t ys, ye;
        band_bounds(ny - 2, P, t, &ys, &ye);
        kern(1, nx - 1, ys + 1, ye + 1, k);   // shift past the y=0 halo row
    }
#else
    (void)nthreads;
    kern(1, nx - 1, 1, ny - 1, k);
#endif
}

// Grand Central Dispatch: dispatch_apply_f runs gcd_band() `nbands` times on a
// concurrent queue, one contiguous band per invocation. Using a high-QoS
// global queue keeps the work on performance cores; DISPATCH_APPLY_AUTO would
// instead let libdispatch pick the width/QoS from the calling context.
#ifdef HAVE_GCD
typedef struct {
    band_kernel kern;
    const kctx *k;
    size_t nx, ny;
    int nbands;
} gcd_task;

static void gcd_band(void *ctx, size_t b) {
    gcd_task *t = (gcd_task *)ctx;
    size_t ys, ye;
    band_bounds(t->ny - 2, t->nbands, (int)b, &ys, &ye);
    t->kern(1, t->nx - 1, ys + 1, ye + 1, t->k);
}
#endif

static void run_gcd(band_kernel kern, size_t nx, size_t ny,
                    const kctx *k, int nbands) {
#ifdef HAVE_GCD
    gcd_task t = { kern, k, nx, ny, nbands };
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply_f((size_t)nbands, q, &t, gcd_band);
#else
    (void)nbands;
    kern(1, nx - 1, 1, ny - 1, k);   // GCD not compiled in -> serial fallback
#endif
}

static void run_backend(backend_kind b, band_kernel kern, size_t nx, size_t ny,
                        const kctx *k, int nthreads) {
    switch (b) {
        case BK_OMP_FOR:  run_omp_for(kern, nx, ny, k, nthreads);  break;
        case BK_OMP_SPMD: run_omp_spmd(kern, nx, ny, k, nthreads); break;
        case BK_GCD:      run_gcd(kern, nx, ny, k, nthreads);      break;
    }
}

// ---------------------------------------------------------------------------
//  Grid setup
// ---------------------------------------------------------------------------
static double *alloc_grid(size_t lda, size_t ny) {
    void *p = NULL;
    // 128-byte alignment == Apple Silicon cache line; rows are lda-aligned too.
    if (posix_memalign(&p, 128, lda * ny * sizeof(double)) != 0) {
        fprintf(stderr, "error: allocation of %zu bytes failed\n",
                lda * ny * sizeof(double));
        exit(EXIT_FAILURE);
    }
    return (double *)p;
}

static void init_grid(size_t nx, size_t ny, size_t lda, double *A, double *B) {
    memset(A, 0, lda * ny * sizeof(double));
    memset(B, 0, lda * ny * sizeof(double));
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            double v = (double)(x + y) / (double)(nx + ny);
            A[y * lda + x] = v;
            B[y * lda + x] = v;   // keep B's halo equal to A's so it stays fixed
        }
    }
}

// ---------------------------------------------------------------------------
//  Correctness check: backend+store output vs. a serial ordinary-store sweep.
//  Returns the maximum absolute difference over the interior (expect ~0).
// ---------------------------------------------------------------------------
static double verify_once(stencil_kind s, backend_kind b, store_kind st,
                          size_t nx, size_t ny, size_t lda, int nthreads) {
    double *A   = alloc_grid(lda, ny);
    double *ref = alloc_grid(lda, ny);
    double *out = alloc_grid(lda, ny);
    init_grid(nx, ny, lda, A, ref);
    init_grid(nx, ny, lda, A, out);

    kctx kref = { lda, A, ref };
    select_kernel(s, SB_STD)(1, nx - 1, 1, ny - 1, &kref);   // serial reference

    kctx kout = { lda, A, out };
    run_backend(b, select_kernel(s, st), nx, ny, &kout, nthreads);

    double maxd = 0.0;
    for (size_t y = 1; y < ny - 1; ++y)
        for (size_t x = 1; x < nx - 1; ++x) {
            double d = fabs(out[y * lda + x] - ref[y * lda + x]);
            if (d > maxd) maxd = d;
        }
    free(A); free(ref); free(out);
    return maxd;
}

// ---------------------------------------------------------------------------
//  Timed run. Returns per-step time and derived bandwidth metrics.
// ---------------------------------------------------------------------------
typedef struct {
    double time_us;    // per stencil sweep
    double eff_gbs;    // effective (algorithmic) bandwidth: 2*N moved / time
    double dram_gbs;   // modelled DRAM traffic: (std 3N | nt 2N) / time
    double mlups;      // million lattice-point updates per second
    int    iters;      // sweeps actually timed (per trial)
    int    trials;
} result;

static result profile(stencil_kind s, backend_kind b, store_kind st,
                       size_t nx, size_t ny, size_t lda,
                       int nthreads, int req_iters, int trials) {
    double *A = alloc_grid(lda, ny);
    double *B = alloc_grid(lda, ny);
    init_grid(nx, ny, lda, A, B);

    band_kernel kern = select_kernel(s, st);
    kctx k = { lda, A, B };

    // Warm caches / TLB, then probe one sweep to size the iteration count so
    // each timed trial runs for roughly a fixed wall-time budget (~50 ms).
    for (int w = 0; w < 3; ++w) {
        k.A = A; k.B = B;
        run_backend(b, kern, nx, ny, &k, nthreads);
        double *tmp = A; A = B; B = tmp;
    }
    double p0 = wtime();
    k.A = A; k.B = B;
    run_backend(b, kern, nx, ny, &k, nthreads);
    double p1 = wtime();
    { double *tmp = A; A = B; B = tmp; }

    double one = p1 - p0;
    if (one <= 0.0) one = 1e-9;
    int iters = (int)(0.05 / one);
    if (iters < 3) iters = 3;
    if (iters > req_iters) iters = req_iters;

    // Report the minimum over trials: least perturbed by OS jitter / migration.
    double best = 1e300;
    for (int tr = 0; tr < trials; ++tr) {
        double t0 = wtime();
        for (int it = 0; it < iters; ++it) {
            k.A = A; k.B = B;
            run_backend(b, kern, nx, ny, &k, nthreads);
            double *tmp = A; A = B; B = tmp;
        }
        double t1 = wtime();
        double tot = t1 - t0;
        if (tot < best) best = tot;
    }

    double per     = best / iters;
    double Nbyte   = (double)nx * (double)ny * (double)sizeof(double);
    double updates = (double)(nx - 2) * (double)(ny - 2);

    result R;
    R.time_us  = per * 1e6;
    R.eff_gbs  = 2.0 * Nbyte / per / 1e9;
    R.dram_gbs = (st == SB_NT ? 2.0 : 3.0) * Nbyte / per / 1e9;
    R.mlups    = updates / per / 1e6;
    R.iters    = iters;
    R.trials   = trials;
    free(A); free(B);
    return R;
}

// ---------------------------------------------------------------------------
//  Naming + CLI
// ---------------------------------------------------------------------------
static const char *stencil_name(stencil_kind s) { return s == ST_JACOBI ? "jacobi" : "nine"; }
static const char *store_name(store_kind st)     { return st == SB_NT ? "nt" : "std"; }
static const char *backend_name(backend_kind b) {
    return b == BK_OMP_FOR ? "omp-for" : b == BK_OMP_SPMD ? "omp-spmd" : "gcd";
}

static void usage(const char *prog) {
    printf(
    "Usage: %s [options]\n\n"
    "Options:\n"
    "  -n, --size N        square grid, nx = ny = N        (default 4096)\n"
    "      --nx N          grid extent in unit-stride dim\n"
    "      --ny N          grid extent in strided dim\n"
    "  -s, --stencil NAME  jacobi | nine                   (default jacobi)\n"
    "  -b, --backend NAME  omp-for | omp-spmd | gcd         (default omp-for)\n"
    "      --store WHICH   std | nt | both                 (default both)\n"
    "  -t, --threads N     worker threads / GCD bands       (default P-cores)\n"
    "  -i, --iters N       max timed sweeps per trial       (default 200)\n"
    "      --trials N      timed trials, minimum reported   (default 5)\n"
    "      --lda N         leading dimension (row stride)   (default nx->mult of 16)\n"
    "  -q, --quiet         emit only the CSV DATA line(s)\n"
    "  -h, --help          this help\n\n"
    "CSV columns (prefixed 'DATA,'):\n"
    "  stencil,backend,store,nx,ny,threads,iters,trials,time_us,eff_gbs,dram_gbs,mlups\n",
    prog);
}

// Read the value for a flag that expects an argument; exits on misuse.
static const char *argval(int argc, char **argv, int *i, const char *inline_eq) {
    if (inline_eq) return inline_eq;             // --key=value form
    if (*i + 1 >= argc) {
        fprintf(stderr, "error: %s requires a value\n", argv[*i]);
        exit(EXIT_FAILURE);
    }
    return argv[++(*i)];
}

int main(int argc, char **argv) {
    size_t nx = 4096, ny = 0, lda = 0;
    stencil_kind stencil = ST_JACOBI;
    backend_kind backend = BK_OMP_FOR;
    int want_std = 1, want_nt = 1;
    int nthreads = -1, req_iters = 200, trials = 5, quiet = 0;

    for (int i = 1; i < argc; ++i) {
        char *a = argv[i];
        char *eq = strchr(a, '=');
        const char *v = eq ? eq + 1 : NULL;
        size_t klen = eq ? (size_t)(eq - a) : strlen(a);
        #define OPT(s) (strncmp(a, s, klen) == 0 && (size_t)strlen(s) == klen)

        if (OPT("-h") || OPT("--help")) { usage(argv[0]); return 0; }
        else if (OPT("-n") || OPT("--size")) { nx = ny = strtoull(argval(argc, argv, &i, v), NULL, 10); }
        else if (OPT("--nx")) { nx = strtoull(argval(argc, argv, &i, v), NULL, 10); }
        else if (OPT("--ny")) { ny = strtoull(argval(argc, argv, &i, v), NULL, 10); }
        else if (OPT("--lda")) { lda = strtoull(argval(argc, argv, &i, v), NULL, 10); }
        else if (OPT("-t") || OPT("--threads")) { nthreads = atoi(argval(argc, argv, &i, v)); }
        else if (OPT("-i") || OPT("--iters")) { req_iters = atoi(argval(argc, argv, &i, v)); }
        else if (OPT("--trials")) { trials = atoi(argval(argc, argv, &i, v)); }
        else if (OPT("-q") || OPT("--quiet")) { quiet = 1; }
        else if (OPT("-s") || OPT("--stencil")) {
            const char *w = argval(argc, argv, &i, v);
            if (!strcmp(w, "jacobi")) stencil = ST_JACOBI;
            else if (!strcmp(w, "nine")) stencil = ST_NINE;
            else { fprintf(stderr, "error: unknown stencil '%s'\n", w); return EXIT_FAILURE; }
        }
        else if (OPT("-b") || OPT("--backend")) {
            const char *w = argval(argc, argv, &i, v);
            if (!strcmp(w, "omp-for")) backend = BK_OMP_FOR;
            else if (!strcmp(w, "omp-spmd")) backend = BK_OMP_SPMD;
            else if (!strcmp(w, "gcd")) backend = BK_GCD;
            else { fprintf(stderr, "error: unknown backend '%s'\n", w); return EXIT_FAILURE; }
        }
        else if (OPT("--store")) {
            const char *w = argval(argc, argv, &i, v);
            if (!strcmp(w, "std")) { want_std = 1; want_nt = 0; }
            else if (!strcmp(w, "nt")) { want_std = 0; want_nt = 1; }
            else if (!strcmp(w, "both")) { want_std = 1; want_nt = 1; }
            else { fprintf(stderr, "error: unknown store '%s'\n", w); return EXIT_FAILURE; }
        }
        else { fprintf(stderr, "error: unknown option '%s'\n", a); usage(argv[0]); return EXIT_FAILURE; }
        #undef OPT
    }

    if (ny == 0) ny = nx;
    if (nx < 3 || ny < 3) { fprintf(stderr, "error: grid must be at least 3x3\n"); return EXIT_FAILURE; }
    if (lda == 0) lda = (nx + 15) & ~(size_t)15;   // pad to a multiple of 16 doubles (128 B)
    if (lda < nx) { fprintf(stderr, "error: lda (%zu) < nx (%zu)\n", lda, nx); return EXIT_FAILURE; }

    if (nthreads <= 0) {
        if (backend == BK_GCD) { int p = perf_cores(); nthreads = p > 0 ? p : default_threads(); }
        else nthreads = default_threads();
    }

    // Guardrails: warn loudly when a selected feature isn't actually compiled
    // in, so a serial/ordinary-store fallback isn't mistaken for the real path.
    if (backend == BK_GCD) {
#ifndef HAVE_GCD
        fprintf(stderr, "warning: built without GCD (-DUSE_GCD); 'gcd' runs SERIALLY.\n");
#endif
    } else {
#ifndef _OPENMP
        fprintf(stderr, "warning: built without OpenMP; '%s' runs SERIALLY.\n", backend_name(backend));
#endif
    }
    if (want_nt) {
#ifndef HAVE_NT_STORE
        fprintf(stderr, "warning: compiler lacks __builtin_nontemporal_store; 'nt' == 'std'.\n");
#endif
    }

    if (!quiet) {
        printf("grid %zu x %zu  (lda %zu, %.1f MB/array)  stencil=%s backend=%s threads=%d\n",
               nx, ny, lda, (double)(lda * ny * sizeof(double)) / (1024.0 * 1024.0),
               stencil_name(stencil), backend_name(backend), nthreads);
#ifdef _OPENMP
        if (backend != BK_GCD) {
            omp_sched_t sk; int chunk; omp_get_schedule(&sk, &chunk);
            // Mask off the monotonic/nonmonotonic modifier bits (high bits);
            // the base schedule (static=1,dynamic=2,guided=3,auto=4) is low.
            int base = (int)sk & 0xf;
            const char *sname = base == 1 ? "static" : base == 2 ? "dynamic"
                              : base == 3 ? "guided" : base == 4 ? "auto" : "?";
            printf("schedule(runtime) -> %s chunk=%d  (set OMP_SCHEDULE=static for one band/thread)\n",
                   sname, chunk);
        }
#endif
    }

    store_kind stores[2];
    int ns = 0;
    if (want_std) stores[ns++] = SB_STD;
    if (want_nt)  stores[ns++] = SB_NT;

    for (int j = 0; j < ns; ++j) {
        store_kind st = stores[j];
        double err = verify_once(stencil, backend, st, nx, ny, lda, nthreads);
        result R = profile(stencil, backend, st, nx, ny, lda, nthreads, req_iters, trials);

        if (!quiet) {
            printf("  %-3s | verify max|err|=%.2e %s | %9.2f us/step | "
                   "eff %7.2f GB/s | dram~%7.2f GB/s | %8.2f MLUP/s\n",
                   store_name(st), err, err <= 1e-9 ? "OK " : "!! ",
                   R.time_us, R.eff_gbs, R.dram_gbs, R.mlups);
        }
        // Machine-readable line for the sweep script / gnuplot.
        printf("DATA,%s,%s,%s,%zu,%zu,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
               stencil_name(stencil), backend_name(backend), store_name(st),
               nx, ny, nthreads, R.iters, R.trials,
               R.time_us, R.eff_gbs, R.dram_gbs, R.mlups);
    }
    return 0;
}
