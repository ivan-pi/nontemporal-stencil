// ===========================================================================
//  stencil.cpp  --  Non-temporal store micro-benchmark for 2-D stencils
// ===========================================================================
//
//  Measures the effect of non-temporal (streaming) stores on the effective
//  memory bandwidth of two 2-D stencils, as a function of problem size, on
//  Apple Silicon (and portably elsewhere for functional testing).
//
//  Stencils (both average their neighbourhood, so the field stays bounded over
//  arbitrarily many iterations -- no overflow during long timing runs):
//    * jacobi : 5-point   B = 1/4 (N + S + E + W)
//    * nine   : 9-point   isotropic binomial smoother,
//                 B = 1/16 ( 4*C + 2*(N+S+E+W) + (NE+NW+SE+SW) )
//
//  Store variants (compile-time policy, chosen with `if constexpr`):
//    * std : ordinary stores (write-allocate: the line is read from DRAM
//            before being overwritten, then evicted -> ~3 array passes).
//    * nt  : non-temporal stores via __builtin_nontemporal_store, which skip
//            the write-allocate read -> ~2 array passes. On AArch64 this
//            lowers to STNP; elsewhere / on non-clang it degrades to an
//            ordinary store (see have_nt_store) so results stay correct.
//
//  Why NT can help: a stencil reads its input once (neighbours stay in cache)
//  and writes a distinct output array with no temporal reuse. For a large grid
//  the ordinary store's write-allocate read is pure waste; NT removes it,
//  cutting modelled DRAM traffic 3N -> 2N, i.e. up to a 1.5x speedup once the
//  working set spills the last-level cache. For small grids that fit in cache,
//  NT can be *slower* because it defeats reuse -- the crossover is what the
//  bandwidth-vs-size plot reveals.
//
//  Thread dispatch backends (see run_backend):
//    * omp-for  : #pragma omp parallel for over rows, schedule(runtime).
//                 The recommended, idiomatic form. Set OMP_SCHEDULE=static
//                 (the default here) for one contiguous band per thread.
//    * omp-spmd : explicit parallel region + manual band decomposition.
//    * gcd      : Apple Grand Central Dispatch dispatch_apply_f over bands.
//
//  Every backend calls the SAME named kernel through a function pointer, so
//  the arithmetic and the store instruction are identical across them -- the
//  numbers are directly comparable and share one correctness check.
//
//  Storage is a std::vector (128-byte aligned) addressed through a thin
//  ArrayView with x the unit-stride index and y strided by lda: a(x, y).
//  Work is split across the strided (y) dimension so every thread owns a large
//  contiguous slab of memory -- ideal for streaming stores, which want to
//  write whole 128-byte cache lines back to back.
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
#include <unistd.h>   // sysconf, _SC_NPROCESSORS_ONLN

#ifdef _OPENMP
#include <omp.h>
inline constexpr bool have_omp = true;
#else
inline constexpr bool have_omp = false;
#endif

#ifdef USE_GCD
#include <dispatch/dispatch.h>
#define HAVE_GCD 1
inline constexpr bool have_gcd = true;
#else
inline constexpr bool have_gcd = false;
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// ---------------------------------------------------------------------------
//  Non-temporal store support
// ---------------------------------------------------------------------------
// Clang exposes the LLVM builtin; on AArch64 it becomes STNP. The token itself
// must be guarded by the preprocessor (not just `if constexpr`) so the file
// still compiles with gcc, which does not declare it.
#if defined(__clang__) && defined(__has_builtin)
#  if __has_builtin(__builtin_nontemporal_store)
#    define HAVE_NT_STORE 1
#  endif
#endif

#ifdef HAVE_NT_STORE
inline constexpr bool have_nt_store = true;
#else
inline constexpr bool have_nt_store = false;
#endif

// AArch64's non-temporal store (STNP) is a *pair* instruction, so clang keeps
// the hint only when it can pair two adjacent vector stores from an unrolled
// loop. Under higher register pressure -- notably the 9-point kernel -- the
// default heuristic fails to pair and SILENTLY downgrades to an ordinary STP,
// so "nt" ends up measuring "std". An explicit interleave hint makes STNP
// emission reliable for both stencils (confirm with `make verify-asm`). gcc
// ignores the pragma and has no NT store anyway.
#define DO_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#  define STENCIL_LOOP_HINT DO_PRAGMA(clang loop interleave_count(4))
#else
#  define STENCIL_LOOP_HINT
#endif

// ---------------------------------------------------------------------------
//  Aligned storage: std::vector with a 128-byte-aligned allocator so each row
//  (lda padded to a multiple of 16 doubles) starts on a cache-line boundary.
// ---------------------------------------------------------------------------
template <class T, std::size_t Align = 128>
struct AlignedAllocator {
    using value_type = T;
    // Required because of the extra (non-type) Align parameter: the default
    // allocator_traits rebind cannot deduce it on its own.
    template <class U>
    struct rebind { using other = AlignedAllocator<U, Align>; };

    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

    T *allocate(std::size_t n) {
        void *p = nullptr;
        if (posix_memalign(&p, Align, n * sizeof(T)) != 0) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }

    template <class U>
    bool operator==(const AlignedAllocator<U, Align> &) const noexcept { return true; }
    template <class U>
    bool operator!=(const AlignedAllocator<U, Align> &) const noexcept { return false; }
};

using AlignedVec = std::vector<double, AlignedAllocator<double>>;

// ---------------------------------------------------------------------------
//  ArrayView: non-owning, stride-aware 2-D indexing. The __restrict on the
//  data pointer is what lets the compiler assume the input and output views do
//  not alias -- without it the non-temporal vectorisation (STNP) is dropped.
// ---------------------------------------------------------------------------
template <class T>
class ArrayView {
public:
    ArrayView(T *__restrict data, std::size_t nx, std::size_t ny, std::size_t lda) noexcept
        : data_(data), nx_(nx), ny_(ny), lda_(lda) {}

    T &operator()(std::size_t x, std::size_t y) const noexcept { return data_[y * lda_ + x]; }
    std::size_t nx() const noexcept { return nx_; }
    std::size_t ny() const noexcept { return ny_; }
    std::size_t lda() const noexcept { return lda_; }

private:
    T *__restrict data_;
    std::size_t nx_, ny_, lda_;
};

// ---------------------------------------------------------------------------
//  Store policy + stencils
// ---------------------------------------------------------------------------
enum class Store { Standard, NonTemporal };

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

enum class Stencil { Jacobi, Nine };

// Stateless generic lambdas: `a` is any ArrayView, `(x, y)` the centre point.
inline constexpr auto stencil_jacobi = [](const auto &a, std::size_t x, std::size_t y) {
    return 0.25 * (a(x, y - 1) + a(x, y + 1) + a(x - 1, y) + a(x + 1, y));
};
inline constexpr auto stencil_nine = [](const auto &a, std::size_t x, std::size_t y) {
    return (1.0 / 16.0) * (4.0 * a(x, y)
        + 2.0 * (a(x, y - 1) + a(x, y + 1) + a(x - 1, y) + a(x + 1, y))
        +       (a(x - 1, y - 1) + a(x + 1, y - 1) + a(x - 1, y + 1) + a(x + 1, y + 1)));
};

// ---------------------------------------------------------------------------
//  The kernel engine: update the row band [y0, y1) of `out` from `in`. Store
//  policy resolves at compile time; the stencil lambda inlines into the inner
//  unit-stride loop, which stays whole inside one thread so NT stores can
//  cover complete cache lines.
// ---------------------------------------------------------------------------
template <Store S, class Stencil>
inline void apply_band(const ArrayView<const double> &in, const ArrayView<double> &out,
                       std::size_t y0, std::size_t y1, Stencil stencil) noexcept {
    const std::size_t nx = in.nx();
    for (std::size_t y = y0; y < y1; ++y) {
        _Pragma("omp simd") STENCIL_LOOP_HINT
        for (std::size_t x = 1; x < nx - 1; ++x)
            store_at<S>(&out(x, y), stencil(in, x, y));
    }
}

// Named instantiations. `extern "C"` keeps the symbols unmangled so a single
// function-pointer type serves every backend and `make verify-asm` can grep
// STNP counts per kernel. Each is a one-line binding of engine + stencil + store.
using KernelFn = void (*)(const ArrayView<const double> &, const ArrayView<double> &,
                          std::size_t, std::size_t) noexcept;

extern "C" {
void k_jacobi_std(const ArrayView<const double> &in, const ArrayView<double> &out,
                  std::size_t y0, std::size_t y1) noexcept {
    apply_band<Store::Standard>(in, out, y0, y1, stencil_jacobi);
}
void k_jacobi_nt(const ArrayView<const double> &in, const ArrayView<double> &out,
                 std::size_t y0, std::size_t y1) noexcept {
    apply_band<Store::NonTemporal>(in, out, y0, y1, stencil_jacobi);
}
void k_nine_std(const ArrayView<const double> &in, const ArrayView<double> &out,
                std::size_t y0, std::size_t y1) noexcept {
    apply_band<Store::Standard>(in, out, y0, y1, stencil_nine);
}
void k_nine_nt(const ArrayView<const double> &in, const ArrayView<double> &out,
               std::size_t y0, std::size_t y1) noexcept {
    apply_band<Store::NonTemporal>(in, out, y0, y1, stencil_nine);
}
}

static KernelFn select_kernel(Stencil s, Store st) {
    if (s == Stencil::Jacobi) return st == Store::NonTemporal ? k_jacobi_nt : k_jacobi_std;
    else                      return st == Store::NonTemporal ? k_nine_nt   : k_nine_std;
}

// ---------------------------------------------------------------------------
//  Backends
// ---------------------------------------------------------------------------
enum class Backend { OmpFor, OmpSpmd, Gcd };

struct Band { std::size_t begin, end; };

// Split [0, n) into `parts` contiguous, near-equal blocks; return block i.
// ([[maybe_unused]]: only the omp-spmd and gcd backends reference it.)
[[maybe_unused]] static Band band_bounds(std::size_t n, int parts, int i) noexcept {
    std::size_t q = n / static_cast<std::size_t>(parts);
    std::size_t r = n % static_cast<std::size_t>(parts);
    std::size_t ii = static_cast<std::size_t>(i);
    std::size_t begin = ii * q + (ii < r ? ii : r);
    return {begin, begin + q + (ii < r ? 1 : 0)};
}

// Worksharing loop over rows. schedule(runtime) lets you compare policies with
// OMP_SCHEDULE at no code cost: `static` gives each thread one contiguous band
// of rows (recommended); `dynamic` scatters rows and is measurably slower here.
static void run_omp_for(KernelFn k, const ArrayView<const double> &in,
                        const ArrayView<double> &out, int nthreads) {
#ifdef _OPENMP
    const std::size_t ny = in.ny();
    #pragma omp parallel for schedule(runtime) num_threads(nthreads)
    for (std::size_t y = 1; y < ny - 1; ++y)
        k(in, out, y, y + 1);
#else
    (void)nthreads;
    k(in, out, 1, in.ny() - 1);
#endif
}

// SPMD: explicit parallel region, one contiguous band of rows per thread.
static void run_omp_spmd(KernelFn k, const ArrayView<const double> &in,
                         const ArrayView<double> &out, int nthreads) {
#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
    {
        int P = omp_get_num_threads();
        int t = omp_get_thread_num();
        auto [ys, ye] = band_bounds(in.ny() - 2, P, t);
        k(in, out, ys + 1, ye + 1);   // shift past the y=0 halo row
    }
#else
    (void)nthreads;
    k(in, out, 1, in.ny() - 1);
#endif
}

// Grand Central Dispatch: dispatch_apply_f runs gcd_band() `nbands` times on a
// concurrent queue, one contiguous band per invocation. A high-QoS global
// queue keeps the work on performance cores; DISPATCH_APPLY_AUTO would instead
// let libdispatch pick the width/QoS from the calling context. (The block form
// dispatch_apply(^(size_t){...}) is cleaner but this mirrors the C API asked
// about; dispatch_apply_f needs a plain function pointer, hence the context.)
#ifdef HAVE_GCD
struct GcdCtx {
    KernelFn k;
    const ArrayView<const double> *in;
    const ArrayView<double> *out;
    int nbands;
};
extern "C" void gcd_band(void *ctx, std::size_t b) {
    auto &g = *static_cast<GcdCtx *>(ctx);
    auto [ys, ye] = band_bounds(g.in->ny() - 2, g.nbands, static_cast<int>(b));
    g.k(*g.in, *g.out, ys + 1, ye + 1);
}
#endif

static void run_gcd(KernelFn k, const ArrayView<const double> &in,
                    const ArrayView<double> &out, int nbands) {
#ifdef HAVE_GCD
    GcdCtx ctx{k, &in, &out, nbands};
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    dispatch_apply_f(static_cast<std::size_t>(nbands), q, &ctx, gcd_band);
#else
    (void)nbands;
    k(in, out, 1, in.ny() - 1);   // GCD not compiled in -> serial fallback
#endif
}

static void run_backend(Backend b, KernelFn k, const ArrayView<const double> &in,
                        const ArrayView<double> &out, int nthreads) {
    switch (b) {
        case Backend::OmpFor:  run_omp_for(k, in, out, nthreads);  break;
        case Backend::OmpSpmd: run_omp_spmd(k, in, out, nthreads); break;
        case Backend::Gcd:     run_gcd(k, in, out, nthreads);      break;
    }
}

// ---------------------------------------------------------------------------
//  Topology + grid helpers
// ---------------------------------------------------------------------------
// Count of performance ("P") cores, or 0 if unknown -- the right default worker
// count for a bandwidth benchmark: E-cores have far less bandwidth and, with an
// equal static split, just become stragglers.
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

static void init_grid(std::size_t nx, std::size_t ny, std::size_t lda,
                      AlignedVec &A, AlignedVec &B) {
    // AlignedVec zero-initialises on construction, so the padding is already 0.
    for (std::size_t y = 0; y < ny; ++y)
        for (std::size_t x = 0; x < nx; ++x) {
            double v = static_cast<double>(x + y) / static_cast<double>(nx + ny);
            A[y * lda + x] = v;
            B[y * lda + x] = v;   // keep B's halo equal to A's so it stays fixed
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
//  Returns the maximum absolute difference over the interior (expect ~0).
// ---------------------------------------------------------------------------
static double verify_once(const Config &cfg, Store store) {
    const std::size_t nx = cfg.nx, ny = cfg.ny, lda = cfg.lda;
    AlignedVec A(lda * ny), ref(lda * ny), out(lda * ny);
    init_grid(nx, ny, lda, A, ref);
    init_grid(nx, ny, lda, A, out);

    ArrayView<const double> in(A.data(), nx, ny, lda);
    ArrayView<double> refv(ref.data(), nx, ny, lda);
    ArrayView<double> outv(out.data(), nx, ny, lda);

    select_kernel(cfg.stencil, Store::Standard)(in, refv, 1, ny - 1);   // serial reference
    run_backend(cfg.backend, select_kernel(cfg.stencil, store), in, outv, cfg.threads);

    double maxd = 0.0;
    for (std::size_t y = 1; y < ny - 1; ++y)
        for (std::size_t x = 1; x < nx - 1; ++x)
            maxd = std::max(maxd, std::fabs(outv(x, y) - refv(x, y)));
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
    init_grid(nx, ny, lda, A, B);
    KernelFn k = select_kernel(cfg.stencil, store);

    double *a = A.data();
    double *b = B.data();
    auto sweep = [&] {
        ArrayView<const double> in(a, nx, ny, lda);
        ArrayView<double> out(b, nx, ny, lda);
        run_backend(cfg.backend, k, in, out, cfg.threads);
        std::swap(a, b);
    };
    auto elapsed = [](clock::time_point t0) {
        return std::chrono::duration<double>(clock::now() - t0).count();
    };

    // Warm caches / TLB, then probe one sweep to size the iteration count so
    // each timed trial runs for roughly a fixed wall-time budget (~50 ms).
    for (int w = 0; w < 3; ++w) sweep();
    auto p0 = clock::now();
    sweep();
    double one = elapsed(p0);
    if (one <= 0.0) one = 1e-9;
    int iters = std::clamp(static_cast<int>(0.05 / one), 3, cfg.iters);

    // Report the minimum over trials: least perturbed by OS jitter / migration.
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

    // Machine-readable line for the sweep script / gnuplot.
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
    "Options:\n"
    "  -n, --size N        square grid, nx = ny = N        (default 4096)\n"
    "      --nx N / --ny N rectangular grid\n"
    "  -s, --stencil NAME  jacobi | nine                   (default jacobi)\n"
    "  -b, --backend NAME  omp-for | omp-spmd | gcd         (default omp-for)\n"
    "      --store WHICH   std | nt | both                 (default both)\n"
    "  -t, --threads N     worker threads / GCD bands       (default P-cores)\n"
    "  -i, --iters N       max timed sweeps per trial       (default 200)\n"
    "      --trials N      timed trials, minimum reported   (default 5)\n"
    "      --lda N         row stride (default nx -> mult of 16, 128 B aligned)\n"
    "  -q, --quiet         emit only the CSV DATA line(s)\n"
    "  -h, --help          this help\n\n"
    "CSV columns (prefixed 'DATA,'):\n"
    "  stencil,backend,store,nx,ny,threads,iters,trials,time_us,eff_gbs,dram_gbs,mlups\n",
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
        // Fetch the value for a flag that expects one (inline `=v` or next arg).
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
    if (cfg.threads <= 0) {
        cfg.threads = (cfg.backend == Backend::Gcd && perf_cores() > 0) ? perf_cores() : default_threads();
    }

    // Warn when a selected feature isn't compiled in, so a serial / ordinary-
    // store fallback isn't mistaken for the real path.
    if (cfg.backend == Backend::Gcd) {
        if constexpr (!have_gcd) std::fprintf(stderr, "warning: built without GCD (-DUSE_GCD); 'gcd' runs SERIALLY.\n");
    } else {
        if constexpr (!have_omp) std::fprintf(stderr, "warning: built without OpenMP; backend runs SERIALLY.\n");
    }
    if (cfg.want_nt && !have_nt_store)
        std::fprintf(stderr, "warning: compiler lacks __builtin_nontemporal_store; 'nt' == 'std'.\n");

    if (!cfg.quiet) {
        std::printf("grid %zu x %zu  (lda %zu, %.1f MB/array)  stencil=%s backend=%s threads=%d\n",
                    cfg.nx, cfg.ny, cfg.lda,
                    static_cast<double>(cfg.lda * cfg.ny * sizeof(double)) / (1024.0 * 1024.0),
                    name_of(cfg.stencil), name_of(cfg.backend), cfg.threads);
#ifdef _OPENMP
        if (cfg.backend != Backend::Gcd) {
            omp_sched_t sk; int chunk; omp_get_schedule(&sk, &chunk);
            int base = static_cast<int>(sk) & 0xf;   // strip monotonic/nonmonotonic modifier bits
            const char *sname = base == 1 ? "static" : base == 2 ? "dynamic"
                              : base == 3 ? "guided" : base == 4 ? "auto" : "?";
            std::printf("schedule(runtime) -> %s chunk=%d  (set OMP_SCHEDULE=static for one band/thread)\n",
                        sname, chunk);
        }
#endif
    }

    if (cfg.want_std) run_one(cfg, Store::Standard);
    if (cfg.want_nt)  run_one(cfg, Store::NonTemporal);
    return 0;
}
