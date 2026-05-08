#include "filters.h"
#include "image_io.h"
#include "bench.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    const char* name;
    void (*run)(const image_t*, image_t*);
} filter_entry_t;

static void run_blur(const image_t* s, image_t* d)   { gaussian_blur_5x5_mpi(s, d); }
static void run_sobel(const image_t* s, image_t* d)  { sobel_edges_mpi(s, d); }
static void run_sharp(const image_t* s, image_t* d)  { unsharp_mask_mpi(s, d, 1.0f); }
static void run_bc(const image_t* s, image_t* d)     { brightness_contrast_mpi(s, d, 20.0f, 1.2f); }
static void run_histeq(const image_t* s, image_t* d) { histogram_equalize_mpi(s, d); }

static const filter_entry_t FILTERS[] = {
    {"blur", run_blur}, {"sobel", run_sobel}, {"sharp", run_sharp},
    {"bc", run_bc}, {"histeq", run_histeq}, {NULL, NULL}
};

static double mpi_run_one_timed(const image_t* src, image_t* dst, void (*fn)(const image_t*, image_t*), int warmup, int timed, double* stddev_out, double* min_out, double* max_out) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < warmup; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        fn(src, dst);
    }

    double* samples = (double*)malloc(timed * sizeof(double));
    double sum = 0.0, mn = 1e18, mx = 0.0;
    for (int i = 0; i < timed; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        fn(src, dst);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        double dt = t1 - t0;
        samples[i] = dt;
        sum += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;
    }
    double mean = sum / timed;
    double var = 0.0;
    for (int i = 0; i < timed; i++) { double d = samples[i] - mean; var += d * d; }
    var /= timed;
    free(samples);
    if (stddev_out) *stddev_out = sqrt(var);
    if (min_out) *min_out = mn;
    if (max_out) *max_out = mx;
    (void)rank;
    return mean;
}

static void make_output_path(char* out, size_t n, const char* prefix, const char* filter, int nranks) {
    const char* dot = strrchr(prefix, '.');
    if (dot) {
        int base_len = (int)(dot - prefix);
        snprintf(out, n, "%.*s_%s_p%d%s", base_len, prefix, filter, nranks, dot);
    } else {
        snprintf(out, n, "%s_%s_p%d.png", prefix, filter, nranks);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (argc < 4) {
        if (rank == 0) fprintf(stderr, "Usage: mpirun -n N %s <input> <filter|all> <output_prefix> [warmup] [timed] [csv]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    const char* in_path = argv[1];
    const char* fname   = argv[2];
    const char* prefix  = argv[3];
    int warmup = (argc > 4) ? atoi(argv[4]) : 2;
    int timed  = (argc > 5) ? atoi(argv[5]) : 5;
    const char* csv = (argc > 6 && argv[6][0]) ? argv[6] : NULL;

    if (rank == 0) bench_print_hardware();
    if (rank == 0) printf("MPI ranks: %d\n", nranks);

    image_t* src = NULL;
    int W = 0, H = 0;
    if (rank == 0) {
        src = image_load_grey(in_path);
        if (!src) { MPI_Abort(MPI_COMM_WORLD, 2); }
        W = src->width; H = src->height;
        printf("Loaded %s: %dx%d\n", in_path, W, H);
    }
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);

    image_t shell_src = { W, H, 1, NULL };
    image_t shell_dst = { W, H, 1, NULL };
    image_t* use_src = (rank == 0) ? src : &shell_src;
    image_t* dst = NULL;
    if (rank == 0) dst = image_alloc(W, H, 1);
    image_t* use_dst = (rank == 0) ? dst : &shell_dst;

    FILE* csvf = NULL;
    if (rank == 0 && csv) {
        FILE* probe = fopen(csv, "r");
        int new_file = (probe == NULL);
        if (probe) fclose(probe);
        csvf = fopen(csv, "a");
        if (csvf && new_file) bench_emit_csv_header(csvf);
    }

    int run_all = (strcmp(fname, "all") == 0);
    for (int i = 0; FILTERS[i].name; i++) {
        if (!run_all && strcmp(FILTERS[i].name, fname) != 0) continue;

        double stddev = 0.0, mn = 0.0, mx = 0.0;
        double mean = mpi_run_one_timed(use_src, use_dst, FILTERS[i].run, warmup, timed, &stddev, &mn, &mx);

        if (rank == 0) {
            char label[96];
            snprintf(label, sizeof(label), "mpi/%s/p%d", FILTERS[i].name, nranks);
            bench_result_t r = { warmup, timed, mean, stddev, mn, mx };
            bench_print_result(label, &r);

            char out_path[1024];
            make_output_path(out_path, sizeof(out_path), prefix, FILTERS[i].name, nranks);
            image_save_png(out_path, dst);
            printf("  -> %s\n", out_path);

            if (csvf) bench_emit_csv_row(csvf, "mpi", FILTERS[i].name, nranks, W, H, &r);
        }
    }

    if (rank == 0) {
        if (csvf) fclose(csvf);
        if (src) image_free(src);
        if (dst) image_free(dst);
    }

    MPI_Finalize();
    return 0;
}
