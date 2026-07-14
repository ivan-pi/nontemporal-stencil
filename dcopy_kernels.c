/*
 * dcopy_kernels.c
 * ---------------------------------------------------------------------------
 * Fortran-callable, BLAS-style DCOPY kernels used to compare ordinary
 * ("temporal") stores against streaming ("non-temporal") stores.
 *
 * Every routine performs the unit-stride Level-1 BLAS operation
 *
 *         y(1:n) = x(1:n)          (i.e. DCOPY with incx = incy = 1)
 *
 * The routines deliberately use the classic Fortran name-mangling ABI
 * (lower-case name + a single trailing underscore, all arguments passed by
 * reference) instead of iso_c_binding, so the Fortran driver can call them
 * exactly the way it would call a legacy BLAS library:
 *
 *         call dcopy_store (n, x, y)   ! standard, cacheable stores
 *         call dcopy_memcpy(n, x, y)   ! hand it to the C library
 *         call dcopy_nt    (n, x, y)   ! non-temporal / streaming stores
 *
 * dcopy_store_ and dcopy_nt_ share an identical structure; the ONLY
 * difference is the store instruction (plain vs. streaming), which isolates
 * the effect being measured.  Streaming stores bypass the caches and skip
 * the read-for-ownership traffic a normal store incurs, so they win once the
 * destination no longer fits in cache.
 * ---------------------------------------------------------------------------
 */

#include <string.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>   /* SSE2 / AVX intrinsics + _mm_sfence() */
#endif

/* ------------------------------------------------------------------------- */
/* 1. Standard stores.                                                       */
/*                                                                           */
/*    On x86 this mirrors dcopy_nt_ exactly but uses ordinary aligned vector */
/*    stores, so the comparison is apples-to-apples.  Using intrinsics also  */
/*    stops the compiler from quietly rewriting the loop into a memcpy call, */
/*    which would erase the distinction from dcopy_memcpy_.                   */
/* ------------------------------------------------------------------------- */
void dcopy_store_(const int *n, const double *restrict x, double *restrict y)
{
    const long len = (long)(*n);
    long i = 0;

#if defined(__AVX__)
    /* Peel until the destination is 32-byte aligned. */
    while (i < len && ((uintptr_t)(y + i) & 31u) != 0) { y[i] = x[i]; ++i; }
    for (; i + 4 <= len; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        _mm256_store_pd(y + i, v);            /* ordinary (temporal) store */
    }
#elif defined(__SSE2__)
    while (i < len && ((uintptr_t)(y + i) & 15u) != 0) { y[i] = x[i]; ++i; }
    for (; i + 2 <= len; i += 2) {
        __m128d v = _mm_loadu_pd(x + i);
        _mm_store_pd(y + i, v);               /* ordinary (temporal) store */
    }
#endif
    for (; i < len; ++i) y[i] = x[i];         /* scalar tail (and fallback) */
}

/* ------------------------------------------------------------------------- */
/* 2. memcpy: let the C library pick an architecture-tuned implementation    */
/*    (glibc itself switches to non-temporal stores for large copies).       */
/* ------------------------------------------------------------------------- */
void dcopy_memcpy_(const int *n, const double *restrict x, double *restrict y)
{
    memcpy(y, x, (size_t)(*n) * sizeof(double));
}

/* ------------------------------------------------------------------------- */
/* 3. Non-temporal / streaming stores.                                       */
/*                                                                           */
/*    Streaming stores require an aligned destination, so a scalar prologue  */
/*    peels elements until y is vector-aligned and a scalar epilogue copies  */
/*    the tail.  A trailing store fence makes the weakly-ordered results     */
/*    visible before the caller reads them back.                             */
/* ------------------------------------------------------------------------- */
void dcopy_nt_(const int *n, const double *restrict x, double *restrict y)
{
    const long len = (long)(*n);
    long i = 0;

#if defined(__AVX__)
    while (i < len && ((uintptr_t)(y + i) & 31u) != 0) { y[i] = x[i]; ++i; }
    for (; i + 4 <= len; i += 4) {
        __m256d v = _mm256_loadu_pd(x + i);
        _mm256_stream_pd(y + i, v);           /* non-temporal store */
    }
#elif defined(__SSE2__)
    while (i < len && ((uintptr_t)(y + i) & 15u) != 0) { y[i] = x[i]; ++i; }
    for (; i + 2 <= len; i += 2) {
        __m128d v = _mm_loadu_pd(x + i);
        _mm_stream_pd(y + i, v);              /* non-temporal store */
    }
#elif defined(__clang__)
    /* Portable fallback: Clang lowers this builtin to the target's native
       non-temporal store (e.g. stnp on AArch64 / Apple silicon). */
    for (; i < len; ++i) __builtin_nontemporal_store(x[i], y + i);
#endif
    for (; i < len; ++i) y[i] = x[i];         /* scalar tail (and fallback) */

#if defined(__x86_64__) || defined(__i386__)
    _mm_sfence();                             /* order the streaming stores */
#endif
}
