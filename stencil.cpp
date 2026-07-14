// ===========================================================================
//  stencil.cpp  --  Non-temporal store micro-benchmark for 2-D stencils
// ===========================================================================
//
//  Measures how non-temporal (streaming) stores affect the memory bandwidth of
//  two 2-D stencils versus problem size, on Apple Silicon.
//
//    jacobi : 5-point   Anew = 1/4 (N + S + E + W)
//    nine   : 9-point   Anew = 1/16 (4C + 2(N+S+E+W) + (NE+NW+SE+SW))
//
//  A stencil reads its input once (neighbours stay cached) and writes a
//  separate output it never reads back. An ordinary store triggers a
//  write-allocate read of each output line (~3 array passes); a non-temporal
//  store skips it (~2 passes), so NT can be ~1.5x faster once the working set
//  spills cache -- and slower while it still fits. The plot shows the crossover.
//
//  Kernels expose a plain C ABI (raw pointers) and use ArrayView internally.
//  Backends (omp-for, gcd) split the strided (y) dimension into contiguous
//  bands so each thread streams over one large slab. Run with --help.
// ===========================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>   // sysconf

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

// Non-temporal store primitive. Clang's builtin becomes STNP on AArch64;
// gcc / other targets fall back to an ordinary store (flagged at runtime).
#if defined(__clang__) && defined(__has_builtin)
#  if __has_builtin(__builtin_nontemporal_store)
#    define HAVE_NT_STORE 1
#  endif
#endif
#ifdef HAVE_NT_STORE
#  define NT_STORE(p, v) __builtin_nontemporal_store((v), (p))
#else
#  define NT_STORE(p, v) (*(p) = (v))
#endif

// ---------------------------------------------------------------------------
//  Storage: std::vector over-aligned to a cache line, so each padded row (lda a
//  multiple of 16 doubles) starts on a 128-byte boundary. A tiny allocator is
//  the only way to over-align a std::vector in standard C++.
// ---------------------------------------------------------------------------
template <class T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <class U> AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

    T *allocate(std::size_t n) {
        return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t{128}));
    }
    void deallocate(T *p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{128});
    }
};
template <class T, class U>
bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) noexcept { return true; }
template <class T, class U>
bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) noexcept { return false; }

using AlignedVec = std::vector<double, AlignedAllocator<double>>;

// ---------------------------------------------------------------------------
//  ArrayView: non-owning 2-D indexing, a(x, y) == data[y*lda + x]. The
//  __restrict data pointer lets the compiler assume input and output do not
//  alias -- without it the non-temporal vectorisation (STNP) is dropped.
// ---------------------------------------------------------------------------
template <class T>
class ArrayView {
public:
    ArrayView(T *__restrict data, std::size_t nx, std::size_t ny, std::size_t lda) noexcept
        : data_(data), nx_(nx), ny_(ny), lda_(lda) {}

    T &operator()(std::size_t x, std::size_t y) const noexcept { return data_[y * lda_ + x]; }
    std::size_t nx() const noexcept { return nx_; }
    std::size_t ny() const noexcept { return ny_; }

private:
    T *__restrict data_;
    std::size_t nx_, ny_, lda_;
};

// ---------------------------------------------------------------------------
//  Kernels
// ---------------------------------------------------------------------------
enum class Store { Standard, NonTemporal };

// Compile-time store policy: no runtime branch in the hot loop.
template <Store S>
inline void store_at(double *__restrict p, double v) noexcept {
    if constexpr (S == Store::NonTemporal) NT_STORE(p, v);
    else                                   *p = v;
}

// One kernel per stencil, formula written inline. always_inline so the loop
// (and its STNP) lands in the C-ABI entry point below, where verify-asm looks.
//
// The `clang loop interleave_count(4)` hint is required, not cosmetic: STNP is
// a pair instruction, and without the hint clang fails to pair the 9-point
// kernel's stores and silently emits ordinary STPs (i.e. "nt" == "std"). gcc
// ignores the pragma and has no NT store anyway. Confirm with `make verify-asm`.
template <Store S>
[[gnu::always_inline]] inline void jacobi_band(
    const double *__restrict in, double *__restrict out,
    std::size_t nx, std::size_t ny, std::size_t lda, std::size_t y0, std::size_t y1) noexcept {
    ArrayView<const double> A(in, nx, ny, lda);
    ArrayView<double> Anew(out, nx, ny, lda);
    for (std::size_t y = y0; y < y1; ++y) {
        #pragma omp simd
        #ifdef __clang__
        #pragma clang loop interleave_count(4)
        #endif
        for (std::size_t x = 1; x < nx - 1; ++x)
            store_at<S>(&Anew(x, y),
                        0.25 * (A(x, y - 1) + A(x, y + 1) + A(x - 1, y) + A(x + 1, y)));
    }
}

template <Store S>
[[gnu::always_inline]] inline void nine_band(
    const double *__restrict in, double *__restrict out,
    std::size_t nx, std::size_t ny, std::size_t lda, std::size_t y0, std::size_t y1) noexcept {
    ArrayView<const double> A(in, nx, ny, lda);
    ArrayView<double> Anew(out, nx, ny, lda);
    for (std::size_t y = y0; y < y1; ++y) {
        #pragma omp simd
        #ifdef __clang__
        #pragma clang loop interleave_count(4)
        #endif
        for (std::size_t x = 1; x < nx - 1; ++x)
            store_at<S>(&Anew(x, y),
                        (1.0 / 16.0) * (4.0 * A(x, y)
                            + 2.0 * (A(x, y - 1) + A(x, y + 1) + A(x - 1, y) + A(x + 1, y))
                            +       (A(x - 1, y - 1) + A(x + 1, y - 1)
                                   + A(x - 1, y + 1) + A(x + 1, y + 1))));
    }
}

// C ABI for the kernels: a single function-pointer type serves every backend,
// and `extern "C"` keeps the symbols unmangled for `make verify-asm`.
using KernelFn = void (*)(const double *in, double *out, std::size_t nx, std::size_t ny,
                          std::size_t lda, std::size_t y0, std::size_t y1);

extern "C" {
void k_jacobi_std(const double *in, double *out, std::size_t nx, std::size_t ny,
                  std::size_t lda, std::size_t y0, std::size_t y1)
    { jacobi_band<Store::Standard>(in, out, nx, ny, lda, y0, y1); }
void k_jacobi_nt(const double *in, double *out, std::size_t nx, std::size_t ny,
                 std::size_t lda, std::size_t y0, std::size_t y1)
    { jacobi_band<Store::NonTemporal>(in, out, nx, ny, lda, y0, y1); }
void k_nine_std(const double *in, double *out, std::size_t nx, std::size_t ny,
                std::size_t lda, std::size_t y0, std::size_t y1)
    { nine_band<Store::Standard>(in, out, nx, ny, lda, y0, y1); }
void k_nine_nt(const double *in, double *out, std::size_t nx, std::size_t ny,
               std::size_t lda, std::size_t y0, std::size_t y1)
    { nine_band<Store::NonTemporal>(in, out, nx, ny, lda, y0, y1); }
}

enum class Stencil { Jacobi, Nine };

static KernelFn select_kernel(Stencil s, Store st) {
    if (s == Stencil::Jacobi) return st == Store::NonTemporal ? k_jacobi_nt : k_jacobi_std;
    else                      return st == Store::NonTemporal ? k_nine_nt   : k_nine_std;
}

// ---------------------------------------------------------------------------
//  Backends: split the strided dimension into contiguous bands of rows.
// ---------------------------------------------------------------------------
enum class Backend { OmpFor, Gcd };

struct Band { std::size_t begin, end; };

// Block i of [0, n) split into `parts` contiguous, near-equal pieces.
[[maybe_unused]] static Band split(std::size_t n, std::size_t parts, std::size_t i) noexcept {
    std::size_t q = n / parts, r = n % parts;
    std::size_t begin = i * q + (i < r ? i : r);
    return {begin, begin + q + (i < r ? 1 : 0)};
}

// Worksharing loop over rows, default (static) schedule -> one contiguous band
// of rows per thread, which is what streaming stores want.
static void run_omp_for(KernelFn k, const double *in, double *out, std::size_t nx,
                        std::size_t ny, std::size_t lda, [[maybe_unused]] int nthreads) {
#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
    for (std::size_t y = 1; y < ny - 1; ++y)
        k(in, out, nx, ny, lda, y, y + 1);
#else
    k(in, out, nx, ny, lda, 1, ny - 1);
#endif
}

// Grand Central Dispatch: dispatch_apply_f runs gcd_band() once per band on a
// concurrent queue. A high-QoS global queue keeps the work on performance
// cores (DISPATCH_APPLY_AUTO would infer QoS from the caller instead).
#ifdef HAVE_GCD
struct GcdTask {
    KernelFn k;
    const double *in;
    double *out;
    std::size_t nx, ny, lda, nbands;
};
extern "C" void gcd_band(void *ctx, std::size_t b) {
    auto &t = *static_cast<GcdTask *>(ctx);
    auto [ys, ye] = split(t.ny - 2, t.nbands, b);
    t.k(t.in, t.out, t.nx, t.ny, t.lda, ys + 1, ye + 1);
}
#endif

static void run_gcd(KernelFn k, const double *in, double *out, std::size_t nx,
                    std::size_t ny, std::size_t lda, [[maybe_unused]] int nbands) {
#ifdef HAVE_GCD
    GcdTask t{k, in, out, nx, ny, lda, static_cast<std::size_t>(nbands)};
    dispatch_apply_f(t.nbands, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &t, gcd_band);
#else
    k(in, out, nx, ny, lda, 1, ny - 1);   // GCD not compiled in -> serial fallback
#endif
}

static void run_backend(Backend b, KernelFn k, const double *in, double *out,
                        std::size_t nx, std::size_t ny, std::size_t lda, int nthreads) {
    if (b == Backend::OmpFor) run_omp_for(k, in, out, nx, ny, lda, nthreads);
    else                      run_gcd(k, in, out, nx, ny, lda, nthreads);
}

// ---------------------------------------------------------------------------
//  Topology + grid setup
// ---------------------------------------------------------------------------
// Number of performance ("P") cores, or 0 if unknown -- the right default for a
// bandwidth benchmark, since E-cores have less bandwidth and become stragglers.
static int perf_cores() {
#ifdef __APPLE__
    int n = 0;
    std::size_t sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0)
        return n;
#endif
    return 0;
}

static int default_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    if (int p = perf_cores(); p > 0) return p;
    long m = sysconf(_SC_NPROCESSORS_ONLN);
    return m > 0 ? static_cast<int>(m) : 1;
#endif
}

static void init_grid(ArrayView<double> A, ArrayView<double> B) {
    assert(A.nx() == B.nx() && A.ny() == B.ny());
    const std::size_t nx = A.nx(), ny = A.ny();
    const double denom = nx + ny;   // size_t -> double
    for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; ++x) {
            double v = (x + y) / denom;
            A(x, y) = v;
            B(x, y) = v;   // keep B's halo equal to A's so the boundary stays fixed
        }
}

// ---------------------------------------------------------------------------
//  Configuration
// ---------------------------------------------------------------------------
struct Config {
    std::size_t nx = 4096, ny = 0, lda = 0;
    Stencil stencil = Stencil::Jacobi;
    Backend backend = Backend::OmpFor;
    bool want_std = true, want_nt = true;
    int threads = -1, iters = 200, trials = 5;
    bool quiet = false;
};

static const char *name_of(Stencil s) { return s == Stencil::Jacobi ? "jacobi" : "nine"; }
static const char *name_of(Store st)  { return st == Store::NonTemporal ? "nt" : "std"; }
static const char *name_of(Backend b) { return b == Backend::OmpFor ? "omp-for" : "gcd"; }

// ---------------------------------------------------------------------------
//  Correctness check: backend+store output vs. a serial ordinary-store sweep.
// ---------------------------------------------------------------------------
static double verify_once(const Config &cfg, Store store) {
    const std::size_t nx = cfg.nx, ny = cfg.ny, lda = cfg.lda;
    AlignedVec A(lda * ny), ref(lda * ny), out(lda * ny);
    init_grid({A.data(), nx, ny, lda}, {ref.data(), nx, ny, lda});
    init_grid({A.data(), nx, ny, lda}, {out.data(), nx, ny, lda});

    select_kernel(cfg.stencil, Store::Standard)(A.data(), ref.data(), nx, ny, lda, 1, ny - 1);
    run_backend(cfg.backend, select_kernel(cfg.stencil, store), A.data(), out.data(), nx, ny, lda, cfg.threads);

    ArrayView<const double> R(ref.data(), nx, ny, lda), O(out.data(), nx, ny, lda);
    double maxd = 0.0;
    for (std::size_t y = 1; y < ny - 1; ++y)
        for (std::size_t x = 1; x < nx - 1; ++x)
            maxd = std::max(maxd, std::fabs(O(x, y) - R(x, y)));
    return maxd;
}

// ---------------------------------------------------------------------------
//  Timed run
// ---------------------------------------------------------------------------
struct Result {
    double time_us;    // per stencil sweep
    double eff_gbs;    // effective (algorithmic) bandwidth: 2*N moved / time
    double dram_gbs;   // modelled DRAM traffic: (std 3N | nt 2N) / time
    double mlups;      // million lattice-point updates per second
    int iters, trials;
};

static Result profile(const Config &cfg, Store store) {
    using clock = std::chrono::steady_clock;
    const std::size_t nx = cfg.nx, ny = cfg.ny, lda = cfg.lda;

    AlignedVec A(lda * ny), B(lda * ny);
    init_grid({A.data(), nx, ny, lda}, {B.data(), nx, ny, lda});
    KernelFn k = select_kernel(cfg.stencil, store);

    double *a = A.data();
    double *b = B.data();
    auto sweep = [&] {
        run_backend(cfg.backend, k, a, b, nx, ny, lda, cfg.threads);
        std::swap(a, b);
    };
    auto elapsed = [](clock::time_point t) {
        return std::chrono::duration<double>(clock::now() - t).count();
    };

    // Warm up, then probe one sweep to size the trial to a ~300 ms budget.
    for (int w = 0; w < 3; ++w) sweep();
    auto probe = clock::now();
    sweep();
    double one = elapsed(probe);
    if (one <= 0.0) one = 1e-9;
    int iters = std::clamp(int(0.3 / one), 3, cfg.iters);

    // Report the minimum over trials: least perturbed by OS jitter.
    double best = std::numeric_limits<double>::max();
    for (int tr = 0; tr < cfg.trials; ++tr) {
        auto t0 = clock::now();
        for (int it = 0; it < iters; ++it) sweep();
        best = std::min(best, elapsed(t0));
    }

    const double per     = best / iters;
    const double Nbyte   = double(nx) * double(ny) * sizeof(double);
    const double updates = double(nx - 2) * double(ny - 2);
    return {per * 1e6,
            2.0 * Nbyte / per / 1e9,
            (store == Store::NonTemporal ? 2.0 : 3.0) * Nbyte / per / 1e9,
            updates / per / 1e6,
            iters, cfg.trials};
}

static void run_one(const Config &cfg, Store store) {
    double err = verify_once(cfg, store);
    Result r = profile(cfg, store);

    if (!cfg.quiet)
        std::printf("  %-3s | verify max|err|=%.2e %s | %9.2f us/step | "
                    "eff %7.2f GB/s | dram~%7.2f GB/s | %8.2f MLUP/s\n",
                    name_of(store), err, err <= 1e-9 ? "OK " : "!! ",
                    r.time_us, r.eff_gbs, r.dram_gbs, r.mlups);

    std::printf("DATA,%s,%s,%s,%zu,%zu,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                name_of(cfg.stencil), name_of(cfg.backend), name_of(store),
                cfg.nx, cfg.ny, cfg.threads, r.iters, r.trials,
                r.time_us, r.eff_gbs, r.dram_gbs, r.mlups);
}

// ---------------------------------------------------------------------------
//  CLI
// ---------------------------------------------------------------------------
// Read the value following `arg` (parsed as T), else return `fallback`.
template <class T>
static T get_argval(char **begin, char **end, const std::string &arg, T fallback) {
    char **it = std::find(begin, end, arg);
    if (it != end && ++it != end) {
        std::istringstream in(*it);
        in >> fallback;
    }
    return fallback;
}
static bool get_arg(char **begin, char **end, const std::string &arg) {
    return std::find(begin, end, arg) != end;
}

static void usage(const char *prog) {
    std::printf(
    "Usage: %s [options]\n\n"
    "  -n N            square grid, nx = ny = N          (default 4096)\n"
    "  --nx N / --ny N rectangular grid\n"
    "  -s NAME         stencil: jacobi | nine            (default jacobi)\n"
    "  -b NAME         backend: omp-for | gcd            (default omp-for)\n"
    "  --store WHICH   std | nt | both                   (default both)\n"
    "  -t N            worker threads / GCD bands         (default P-cores)\n"
    "  -i N            max timed sweeps per trial         (default 200)\n"
    "  --trials N      timed trials, minimum reported     (default 5)\n"
    "  --lda N         row stride (default nx -> mult of 16)\n"
    "  -q              emit only the CSV DATA line(s)\n"
    "  -h              this help\n",
    prog);
}

int main(int argc, char **argv) {
    char **b = argv, **e = argv + argc;
    if (get_arg(b, e, "-h") || get_arg(b, e, "--help")) { usage(argv[0]); return 0; }

    Config cfg;
    cfg.nx      = get_argval(b, e, "-n", cfg.nx);
    cfg.nx      = get_argval(b, e, "--nx", cfg.nx);
    cfg.ny      = get_argval(b, e, "--ny", cfg.ny);   // 0 => square, set below
    cfg.lda     = get_argval(b, e, "--lda", cfg.lda);
    cfg.threads = get_argval(b, e, "-t", cfg.threads);
    cfg.iters   = get_argval(b, e, "-i", cfg.iters);
    cfg.trials  = get_argval(b, e, "--trials", cfg.trials);
    cfg.quiet   = get_arg(b, e, "-q") || get_arg(b, e, "--quiet");

    std::string stencil = get_argval<std::string>(b, e, "-s", "jacobi");
    std::string backend = get_argval<std::string>(b, e, "-b", "omp-for");
    std::string store   = get_argval<std::string>(b, e, "--store", "both");

    if      (stencil == "jacobi") cfg.stencil = Stencil::Jacobi;
    else if (stencil == "nine")   cfg.stencil = Stencil::Nine;
    else { std::fprintf(stderr, "error: unknown stencil '%s'\n", stencil.c_str()); return EXIT_FAILURE; }

    if      (backend == "omp-for") cfg.backend = Backend::OmpFor;
    else if (backend == "gcd")     cfg.backend = Backend::Gcd;
    else { std::fprintf(stderr, "error: unknown backend '%s'\n", backend.c_str()); return EXIT_FAILURE; }

    if      (store == "std")  { cfg.want_std = true;  cfg.want_nt = false; }
    else if (store == "nt")   { cfg.want_std = false; cfg.want_nt = true; }
    else if (store == "both") { cfg.want_std = true;  cfg.want_nt = true; }
    else { std::fprintf(stderr, "error: unknown store '%s'\n", store.c_str()); return EXIT_FAILURE; }

    if (cfg.ny == 0) cfg.ny = cfg.nx;
    if (cfg.nx < 3 || cfg.ny < 3) { std::fprintf(stderr, "error: grid must be at least 3x3\n"); return EXIT_FAILURE; }
    if (cfg.lda == 0) cfg.lda = (cfg.nx + 15) & ~std::size_t(15);   // multiple of 16 doubles (128 B)
    if (cfg.lda < cfg.nx) { std::fprintf(stderr, "error: lda (%zu) < nx (%zu)\n", cfg.lda, cfg.nx); return EXIT_FAILURE; }
    if (cfg.threads <= 0)
        cfg.threads = (cfg.backend == Backend::Gcd && perf_cores() > 0) ? perf_cores() : default_threads();

    // Warn when a selected feature isn't compiled in, so a serial / ordinary-
    // store fallback isn't mistaken for the real path.
#ifndef _OPENMP
    if (cfg.backend != Backend::Gcd) std::fprintf(stderr, "warning: no OpenMP; backend runs serially\n");
#endif
#ifndef HAVE_GCD
    if (cfg.backend == Backend::Gcd) std::fprintf(stderr, "warning: no GCD (build with -DUSE_GCD); gcd runs serially\n");
#endif
#ifndef HAVE_NT_STORE
    if (cfg.want_nt) std::fprintf(stderr, "warning: no __builtin_nontemporal_store; nt == std\n");
#endif

    if (!cfg.quiet)
        std::printf("grid %zu x %zu  lda %zu  (%.1f MB/array)  stencil=%s backend=%s threads=%d\n",
                    cfg.nx, cfg.ny, cfg.lda,
                    double(cfg.lda * cfg.ny * sizeof(double)) / (1024.0 * 1024.0),
                    name_of(cfg.stencil), name_of(cfg.backend), cfg.threads);

    if (cfg.want_std) run_one(cfg, Store::Standard);
    if (cfg.want_nt)  run_one(cfg, Store::NonTemporal);
    return 0;
}
