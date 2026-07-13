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
//  Backends split the strided (y) dimension into contiguous bands so each
//  thread streams over one large slab -- ideal for full-cache-line NT writes.
//  Run with --help for options.
// ===========================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <string_view>
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

// Clang exposes the LLVM builtin; on AArch64 it becomes STNP. The token must be
// preprocessor-guarded (not just behind `if constexpr`) so gcc still compiles.
#if defined(__clang__) && defined(__has_builtin)
#  if __has_builtin(__builtin_nontemporal_store)
#    define HAVE_NT_STORE 1
#  endif
#endif

// STNP is a *pair* instruction, so clang keeps the non-temporal hint only when
// it can pair two adjacent vector stores. Under register pressure (the 9-point
// kernel) the default heuristic fails to pair and silently downgrades to an
// ordinary STP -- so "nt" would measure "std". An explicit interleave hint
// makes STNP reliable for both kernels; confirm with `make verify-asm`.
#define DO_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#  define STENCIL_LOOP_HINT DO_PRAGMA(clang loop interleave_count(4))
#else
#  define STENCIL_LOOP_HINT
#endif

// ---------------------------------------------------------------------------
//  Storage: std::vector with a 128-byte-aligned allocator so each padded row
//  (lda a multiple of 16 doubles) starts on a cache-line boundary.
// ---------------------------------------------------------------------------
template <class T>
struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t alignment = 128;   // Apple Silicon cache line

    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

    T *allocate(std::size_t n) {
        // std::aligned_alloc (C++17) requires the size to be a multiple of the
        // alignment, so round up.
        std::size_t bytes = (n * sizeof(T) + alignment - 1) / alignment * alignment;
        if (void *p = std::aligned_alloc(alignment, bytes)) return static_cast<T *>(p);
        throw std::bad_alloc();
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }
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
    if constexpr (S == Store::NonTemporal) {
#ifdef HAVE_NT_STORE
        __builtin_nontemporal_store(v, p);
#else
        *p = v;
#endif
    } else {
        *p = v;
    }
}

// One kernel per stencil, formula written inline. always_inline so the loop
// (and its STNP) lands in the C-ABI entry point below, where verify-asm looks.
template <Store S>
[[gnu::always_inline]] inline void jacobi_band(
    const double *__restrict in, double *__restrict out,
    std::size_t nx, std::size_t ny, std::size_t lda, std::size_t y0, std::size_t y1) noexcept {
    ArrayView<const double> A(in, nx, ny, lda);
    ArrayView<double> Anew(out, nx, ny, lda);
    for (std::size_t y = y0; y < y1; ++y) {
        _Pragma("omp simd") STENCIL_LOOP_HINT
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
        _Pragma("omp simd") STENCIL_LOOP_HINT
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
enum class Backend { OmpFor, OmpSpmd, Gcd };

struct Band { std::size_t begin, end; };

// Block i of [0, n) split into `parts` contiguous, near-equal pieces.
[[maybe_unused]] static Band split(std::size_t n, int parts, int i) noexcept {
    std::size_t q = n / static_cast<std::size_t>(parts);
    std::size_t r = n % static_cast<std::size_t>(parts);
    std::size_t ii = static_cast<std::size_t>(i);
    std::size_t begin = ii * q + (ii < r ? ii : r);
    return {begin, begin + q + (ii < r ? 1 : 0)};
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

// SPMD: explicit parallel region, one contiguous band of rows per thread.
static void run_omp_spmd(KernelFn k, const double *in, double *out, std::size_t nx,
                         std::size_t ny, std::size_t lda, [[maybe_unused]] int nthreads) {
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
    {
        auto [ys, ye] = split(ny - 2, omp_get_num_threads(), omp_get_thread_num());
        k(in, out, nx, ny, lda, ys + 1, ye + 1);   // shift past the y=0 halo row
    }
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
    std::size_t nx, ny, lda;
    int nbands;
};
extern "C" void gcd_band(void *ctx, std::size_t b) {
    auto &t = *static_cast<GcdTask *>(ctx);
    auto [ys, ye] = split(t.ny - 2, t.nbands, static_cast<int>(b));
    t.k(t.in, t.out, t.nx, t.ny, t.lda, ys + 1, ye + 1);
}
#endif

static void run_gcd(KernelFn k, const double *in, double *out, std::size_t nx,
                    std::size_t ny, std::size_t lda, [[maybe_unused]] int nbands) {
#ifdef HAVE_GCD
    GcdTask t{k, in, out, nx, ny, lda, nbands};
    dispatch_apply_f(static_cast<std::size_t>(nbands),
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &t, gcd_band);
#else
    k(in, out, nx, ny, lda, 1, ny - 1);   // GCD not compiled in -> serial fallback
#endif
}

static void run_backend(Backend b, KernelFn k, const double *in, double *out,
                        std::size_t nx, std::size_t ny, std::size_t lda, int nthreads) {
    switch (b) {
        case Backend::OmpFor:  run_omp_for(k, in, out, nx, ny, lda, nthreads);  break;
        case Backend::OmpSpmd: run_omp_spmd(k, in, out, nx, ny, lda, nthreads); break;
        case Backend::Gcd:     run_gcd(k, in, out, nx, ny, lda, nthreads);      break;
    }
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
    const std::size_t nx = A.nx(), ny = A.ny();
    for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; ++x) {
            double v = static_cast<double>(x + y) / static_cast<double>(nx + ny);
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
static const char *name_of(Backend b) {
    return b == Backend::OmpFor ? "omp-for" : b == Backend::OmpSpmd ? "omp-spmd" : "gcd";
}

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

    // Warm up, then probe one sweep to size the trial to a ~50 ms budget.
    for (int w = 0; w < 3; ++w) sweep();
    auto probe = clock::now();
    sweep();
    double one = elapsed(probe);
    if (one <= 0.0) one = 1e-9;
    int iters = std::clamp(static_cast<int>(0.05 / one), 3, cfg.iters);

    // Report the minimum over trials: least perturbed by OS jitter.
    double best = std::numeric_limits<double>::max();
    for (int tr = 0; tr < cfg.trials; ++tr) {
        auto t0 = clock::now();
        for (int it = 0; it < iters; ++it) sweep();
        best = std::min(best, elapsed(t0));
    }

    const double per     = best / iters;
    const double Nbyte   = static_cast<double>(nx) * static_cast<double>(ny) * sizeof(double);
    const double updates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2);
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
static void usage(const char *prog) {
    std::printf(
    "Usage: %s [options]\n\n"
    "  -n, --size N        square grid, nx = ny = N        (default 4096)\n"
    "      --nx N / --ny N rectangular grid\n"
    "  -s, --stencil NAME  jacobi | nine                   (default jacobi)\n"
    "  -b, --backend NAME  omp-for | omp-spmd | gcd         (default omp-for)\n"
    "      --store WHICH   std | nt | both                 (default both)\n"
    "  -t, --threads N     worker threads / GCD bands       (default P-cores)\n"
    "  -i, --iters N       max timed sweeps per trial       (default 200)\n"
    "      --trials N      timed trials, minimum reported   (default 5)\n"
    "      --lda N         row stride (default nx -> mult of 16)\n"
    "  -q, --quiet         emit only the CSV DATA line(s)\n"
    "  -h, --help\n",
    prog);
}

int main(int argc, char **argv) {
    Config cfg;
    std::vector<std::string_view> args(argv + 1, argv + argc);

    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view a = args[i], key = a, inl;
        if (auto eq = a.find('='); eq != std::string_view::npos) {
            key = a.substr(0, eq);
            inl = a.substr(eq + 1);
        }
        auto value = [&]() -> std::string_view {
            if (!inl.empty()) return inl;
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "error: %.*s requires a value\n", (int)key.size(), key.data());
                std::exit(EXIT_FAILURE);
            }
            return args[++i];
        };
        auto as_int  = [&] { return std::atoi(std::string(value()).c_str()); };
        auto as_size = [&] { return static_cast<std::size_t>(std::strtoull(std::string(value()).c_str(), nullptr, 10)); };
        auto bad = [&](const char *what, std::string_view v) {
            std::fprintf(stderr, "error: unknown %s '%.*s'\n", what, (int)v.size(), v.data());
            std::exit(EXIT_FAILURE);
        };

        if (key == "-h" || key == "--help") { usage(argv[0]); return 0; }
        else if (key == "-n" || key == "--size") cfg.nx = cfg.ny = as_size();
        else if (key == "--nx") cfg.nx = as_size();
        else if (key == "--ny") cfg.ny = as_size();
        else if (key == "--lda") cfg.lda = as_size();
        else if (key == "-t" || key == "--threads") cfg.threads = as_int();
        else if (key == "-i" || key == "--iters") cfg.iters = as_int();
        else if (key == "--trials") cfg.trials = as_int();
        else if (key == "-q" || key == "--quiet") cfg.quiet = true;
        else if (key == "-s" || key == "--stencil") {
            auto v = value();
            if (v == "jacobi") cfg.stencil = Stencil::Jacobi;
            else if (v == "nine") cfg.stencil = Stencil::Nine;
            else bad("stencil", v);
        }
        else if (key == "-b" || key == "--backend") {
            auto v = value();
            if (v == "omp-for") cfg.backend = Backend::OmpFor;
            else if (v == "omp-spmd") cfg.backend = Backend::OmpSpmd;
            else if (v == "gcd") cfg.backend = Backend::Gcd;
            else bad("backend", v);
        }
        else if (key == "--store") {
            auto v = value();
            if (v == "std") { cfg.want_std = true; cfg.want_nt = false; }
            else if (v == "nt") { cfg.want_std = false; cfg.want_nt = true; }
            else if (v == "both") { cfg.want_std = cfg.want_nt = true; }
            else bad("store", v);
        }
        else bad("option", a);
    }

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
                    static_cast<double>(cfg.lda * cfg.ny * sizeof(double)) / (1024.0 * 1024.0),
                    name_of(cfg.stencil), name_of(cfg.backend), cfg.threads);

    if (cfg.want_std) run_one(cfg, Store::Standard);
    if (cfg.want_nt)  run_one(cfg, Store::NonTemporal);
    return 0;
}
