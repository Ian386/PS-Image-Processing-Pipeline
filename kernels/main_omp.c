#include "filters.h"
#include "image_io.h"
#include "bench.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* name;
    void (*run)(const image_t*, image_t*, int, const char*);
} filter_entry_t;

static void run_blur(const image_t* s, image_t* d, int t, const char* sc) { gaussian_blur_5x5_omp(s, d, t, sc); }
static void run_sobel(const image_t* s, image_t* d, int t, const char* sc) { sobel_edges_omp(s, d, t, sc); }
static void run_sharp(const image_t* s, image_t* d, int t, const char* sc) { unsharp_mask_omp(s, d, 1.0f, t, sc); }
static void run_bc(const image_t* s, image_t* d, int t, const char* sc) { brightness_contrast_omp(s, d, 20.0f, 1.2f, t, sc); }
static void run_histeq(const image_t* s, image_t* d, int t, const char* sc) { histogram_equalize_omp(s, d, t, sc); }

static const filter_entry_t FILTERS[] = {
    {"blur",   run_blur},
    {"sobel",  run_sobel},
    {"sharp",  run_sharp},
    {"bc",     run_bc},
    {"histeq", run_histeq},
    {NULL, NULL}
};

typedef struct {
    const image_t* src;
    image_t* dst;
    void (*fn)(const image_t*, image_t*, int, const char*);
    int threads;
    const char* schedule;
} ctx_t;

static void work_fn(void* p) {
    ctx_t* c = (ctx_t*)p;
    c->fn(c->src, c->dst, c->threads, c->schedule);
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <input> <filter|all> <output_prefix> [warmup] [timed] [csv] [threads|sweep] [schedule]\n"
        "  filter   : blur | sobel | sharp | bc | histeq | all\n"
        "  warmup   : default 2\n"
        "  timed    : default 5\n"
        "  csv      : optional path to append results\n"
        "  threads  : positive int OR 'sweep' for full thread*schedule matrix\n"
        "  schedule : static | dynamic | guided   (ignored if threads==sweep)\n", prog);
}

static void make_output_path(char* out, size_t n, const char* prefix, const char* filter, int threads, const char* sched) {
    const char* dot = strrchr(prefix, '.');
    if (dot) {
        int base_len = (int)(dot - prefix);
        snprintf(out, n, "%.*s_%s_t%d_%s%s", base_len, prefix, filter, threads, sched, dot);
    } else {
        snprintf(out, n, "%s_%s_t%d_%s.png", prefix, filter, threads, sched);
    }
}

static int run_one(const image_t* src, image_t* dst, const filter_entry_t* fe, int threads, const char* sched, int warmup, int timed, const char* prefix, FILE* csvf) {
    ctx_t ctx = { src, dst, fe->run, threads, sched };
    bench_result_t r;
    bench_run(work_fn, &ctx, warmup, timed, &r);

    char label[96];
    snprintf(label, sizeof(label), "omp/%s/t%d/%s", fe->name, threads, sched);
    bench_print_result(label, &r);

    char out_path[1024];
    make_output_path(out_path, sizeof(out_path), prefix, fe->name, threads, sched);
    image_save_png(out_path, dst);
    printf("  -> %s\n", out_path);

    if (csvf) {
        char paradigm[32];
        snprintf(paradigm, sizeof(paradigm), "omp_%s", sched);
        bench_emit_csv_row(csvf, paradigm, fe->name, threads, src->width, src->height, &r);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }

    const char* in_path  = argv[1];
    const char* fname    = argv[2];
    const char* prefix   = argv[3];
    int warmup           = (argc > 4) ? atoi(argv[4]) : 2;
    int timed            = (argc > 5) ? atoi(argv[5]) : 5;
    const char* csv      = (argc > 6 && argv[6][0]) ? argv[6] : NULL;
    const char* threads_arg = (argc > 7) ? argv[7] : "4";
    const char* sched    = (argc > 8) ? argv[8] : "static";

    int sweep = (strcmp(threads_arg, "sweep") == 0);
    int single_threads = sweep ? 0 : atoi(threads_arg);
    if (!sweep && single_threads < 1) single_threads = 1;

    bench_print_hardware();
    int max_threads = omp_get_max_threads();
    printf("OpenMP max threads available: %d\n", max_threads);

    image_t* src = image_load_grey(in_path);
    if (!src) return 2;
    printf("Loaded %s: %dx%d\n", in_path, src->width, src->height);
    image_t* dst = image_alloc(src->width, src->height, src->channels);

    FILE* csvf = NULL;
    if (csv) {
        FILE* probe = fopen(csv, "r");
        int new_file = (probe == NULL);
        if (probe) fclose(probe);
        csvf = fopen(csv, "a");
        if (csvf && new_file) bench_emit_csv_header(csvf);
    }

    int run_all = (strcmp(fname, "all") == 0);

    int tcounts[] = {1, 2, 4, 8};
    const char* schedules[] = {"static", "dynamic", "guided"};

    for (int i = 0; FILTERS[i].name; i++) {
        if (!run_all && strcmp(FILTERS[i].name, fname) != 0) continue;

        if (sweep) {
            for (size_t s = 0; s < sizeof(schedules)/sizeof(schedules[0]); s++) {
                for (size_t t = 0; t < sizeof(tcounts)/sizeof(tcounts[0]); t++) {
                    if (tcounts[t] > max_threads) continue;
                    run_one(src, dst, &FILTERS[i], tcounts[t], schedules[s], warmup, timed, prefix, csvf);
                }
            }
        } else {
            run_one(src, dst, &FILTERS[i], single_threads, sched, warmup, timed, prefix, csvf);
        }
    }

    if (csvf) fclose(csvf);
    image_free(src);
    image_free(dst);
    return 0;
}
