#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ---------------------------------------------------------
// Compiler-Specific Non-Temporal Store Macro
// ---------------------------------------------------------
#if defined(__clang__)
    // Clang supports the LLVM builtin for non-temporal stores
    #define STREAM_STORE(val, addr) __builtin_nontemporal_store((val), (addr))
#else
    // Fallback for GCC and other compilers
    #define STREAM_STORE(val, addr) (*(addr) = (val))
#endif

// ---------------------------------------------------------
// Helpers & Topology
// ---------------------------------------------------------

void print_help(const char *prog_name) {
    printf("Usage: %s [nx] [ny] [iterations] [px] [py] [lda]\n\n", prog_name);
    printf("Arguments:\n");
    printf("  nx          Grid size in the X (contiguous/stride-1) dimension (default: 8192)\n");
    printf("  ny          Grid size in the Y dimension (default: 8192)\n");
    printf("  iterations  Number of Jacobi steps (default: 100)\n");
    printf("  px          Thread grid topology in X (default: 1)\n");
    printf("  py          Thread grid topology in Y (default: max_threads)\n");
    printf("  lda         Leading dimension for padding (default: nx padded to mult of 16)\n");
    printf("\nNote: For Apple M2 Pro non-temporal stores, lda should be a multiple of 16.\n");
}

void cart_coords(int rank, int px, int py, int *tx, int *ty) {
    *tx = rank % px;
    *ty = rank / px;
}

void block_decompose(size_t total_elements, int total_ranks, int my_rank, size_t *start, size_t *end) {
    size_t work = total_elements / total_ranks;
    size_t rem = total_elements % total_ranks;
    *start = my_rank * work + (my_rank < rem ? my_rank : rem);
    *end = *start + work + (my_rank < rem ? 1 : 0);
}

void swap_pointers(double **a, double **b) {
    double *tmp = *a;
    *a = *b;
    *b = tmp;
}

// ---------------------------------------------------------
// Memory & Generic Launcher
// ---------------------------------------------------------

void allocate_grid(double **grid, size_t lda, size_t ny) {
    if (posix_memalign((void**)grid, 128, lda * ny * sizeof(double)) != 0) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
}

typedef void (*BlockKernel)(size_t x_start, size_t x_end, size_t y_start, size_t y_end, void *ctx);

void launch_2d_grid(int px, int py, size_t total_x, size_t total_y, size_t offset_x, size_t offset_y, BlockKernel kernel, void *ctx) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        if (tid < px * py) {
            int tx, ty;
            cart_coords(tid, px, py, &tx, &ty);

            size_t x_start, x_end, y_start, y_end;
            block_decompose(total_x, px, tx, &x_start, &x_end);
            block_decompose(total_y, py, ty, &y_start, &y_end);

            kernel(x_start + offset_x, x_end + offset_x, y_start + offset_y, y_end + offset_y, ctx);
        }
    }
}

// ---------------------------------------------------------
// Jacobi Kernels (Column-Major Format)
// ---------------------------------------------------------

typedef struct {
    size_t lda;
    const double *restrict A;
    double *restrict Anew;
} JacobiContext;

void kernel_jacobi_standard(size_t x_start, size_t x_end, size_t y_start, size_t y_end, void *ctx) {
    JacobiContext *jctx = (JacobiContext *)ctx;
    size_t lda = jctx->lda;
    const double *restrict A = jctx->A;
    double *restrict Anew = jctx->Anew;

    for (size_t y = y_start; y < y_end; ++y) {
        #pragma omp simd
        for (size_t x = x_start; x < x_end; ++x) {
            Anew[y * lda + x] = 0.25 * (A[(y - 1) * lda + x] +
                                        A[(y + 1) * lda + x] +
                                        A[y * lda + (x - 1)] +
                                        A[y * lda + (x + 1)]);
        }
    }
}

void kernel_jacobi_nt(size_t x_start, size_t x_end, size_t y_start, size_t y_end, void *ctx) {
    JacobiContext *jctx = (JacobiContext *)ctx;
    size_t lda = jctx->lda;
    const double *restrict A = jctx->A;
    double *restrict Anew = jctx->Anew;

    for (size_t y = y_start; y < y_end; ++y) {
        #pragma omp simd
        for (size_t x = x_start; x < x_end; ++x) {
            double val = 0.25 * (A[(y - 1) * lda + x] +
                                 A[(y + 1) * lda + x] +
                                 A[y * lda + (x - 1)] +
                                 A[y * lda + (x + 1)]);
            // Use the macro to branch safely between Compilers
            STREAM_STORE(val, &Anew[y * lda + x]);
        }
    }
}

void init_grid(size_t nx, size_t ny, size_t lda, double *A, double *Anew) {
    memset(A, 0, lda * ny * sizeof(double));
    memset(Anew, 0, lda * ny * sizeof(double));

    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            double val = (double)(x + y) / (nx + ny);
            A[y * lda + x] = val;
            Anew[y * lda + x] = val; 
        }
    }
}

void profile_kernel(const char *name, BlockKernel kernel, size_t nx, size_t ny, size_t lda, int px, int py, int iterations) {
    double *A, *Anew;
    allocate_grid(&A, lda, ny);
    allocate_grid(&Anew, lda, ny);
    init_grid(nx, ny, lda, A, Anew);

    size_t active_x = nx - 2;
    size_t active_y = ny - 2;
    size_t offset_x = 1;
    size_t offset_y = 1;

    JacobiContext ctx = { .lda = lda, .A = A, .Anew = Anew };

    // Warmup
    for (int iter = 0; iter < 5; ++iter) {
        ctx.A = A; ctx.Anew = Anew;
        launch_2d_grid(px, py, active_x, active_y, offset_x, offset_y, kernel, &ctx);
        swap_pointers(&A, &Anew);
    }

    double start_time = omp_get_wtime();
    int actual_iters = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        ctx.A = A; ctx.Anew = Anew;
        launch_2d_grid(px, py, active_x, active_y, offset_x, offset_y, kernel, &ctx);
        swap_pointers(&A, &Anew);
        actual_iters++;

        // Adaptive early exit: Stop if it has taken > 500ms and at least 3 iterations have run
        if (actual_iters >= 3 && (omp_get_wtime() - start_time) >= 0.5) {
            break;
        }
    }
    double end_time = omp_get_wtime();

    // Crucial: Calculate averages using actual_iters, not the requested iterations
    double avg_time = (end_time - start_time) / actual_iters;
    double time_us = avg_time * 1e6;

    double data_gb = (2.0 * nx * ny * sizeof(double)) / 1e9;
    double bandwidth = data_gb / avg_time;

    // Human readable logs
    printf("%-15s | %5d iters | %10.2f us/step | %8.2f GB/s\n", name, actual_iters, time_us, bandwidth);

    // Machine readable prefix for awk (Added actual_iters as the 5th column)
    printf("DATA,%s,%zu,%zu,%d,%.2f,%.2f\n", name, nx, ny, actual_iters, time_us, bandwidth);

    free(A);
    free(Anew);
}

int main(int argc, char **argv) {
    if (argc > 1 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        print_help(argv[0]);
        return 0;
    }

    size_t nx = 8192;
    size_t ny = 8192;
    int iterations = 100;
    int px = 1;
    int py = omp_get_max_threads(); 
    size_t lda = 0;

    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) iterations = atoi(argv[3]);
    if (argc > 4) px = atoi(argv[4]);
    if (argc > 5) py = atoi(argv[5]);
    if (argc > 6) {
        lda = atoi(argv[6]);
    } else {
        // Default lda: pad nx to the next multiple of 16 (128 bytes)
        lda = (nx + 15) & ~15;
//        lda = nx;
    }

    if (lda < nx) {
        fprintf(stderr, "Error: lda (%zu) cannot be smaller than nx (%zu).\n", lda, nx);
        return EXIT_FAILURE;
    }

    if (px * py > omp_get_max_threads()) {
        fprintf(stderr, "Error: Requested thread grid %dx%d (%d) exceeds available threads (%d).\n", 
                px, py, px * py, omp_get_max_threads());
        return EXIT_FAILURE;
    }

    printf("Logical Grid: %zu (X) x %zu (Y)\n", nx, ny);
    printf("Memory Layout: lda = %zu | Padded Size: %zu MB\n", lda, (lda * ny * sizeof(double)) / (1024 * 1024));
    printf("Iterations: %d\n", iterations);
    printf("Thread Topology: %d (X) x %d (Y) = %d active threads\n", px, py, px * py);
    printf("------------------------------------------------------\n");

    profile_kernel("Standard", kernel_jacobi_standard, nx, ny, lda, px, py, iterations);
    profile_kernel("Non-Temporal", kernel_jacobi_nt, nx, ny, lda, px, py, iterations);

    return 0;
}
