#include "filters.h"
#include "image_io.h"
#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* name;
    void (*run)(const image_t* src, image_t* dst);
} filter_entry_t;

static void run_blur(const image_t* s, image_t* d)   { gaussian_blur_5x5_seq(s, d); }
static void run_sobel(const image_t* s, image_t* d)  { sobel_edges_seq(s, d); }
static void run_sharp(const image_t* s, image_t* d)  { unsharp_mask_seq(s, d, 1.0f); }
static void run_bc(const image_t* s, image_t* d)     { brightness_contrast_seq(s, d, 20.0f, 1.2f); }
static void run_histeq(const image_t* s, image_t* d) { histogram_equalize_seq(s, d); }

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
    void (*fn)(const image_t*, image_t*);
} ctx_t;

static void work_fn(void* p) {
    ctx_t* c = (ctx_t*)p;
    c->fn(c->src, c->dst);
}

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <input> <filter> <output> [warmup] [timed] [csv]\n"
        "  filter: blur | sobel | sharp | bc | histeq | all\n"
        "  warmup: default 2\n"
        "  timed : default 5\n"
        "  csv   : optional CSV path to append results\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 4) { usage(argv[0]); return 1; }
    const char* in_path = argv[1];
    const char* fname   = argv[2];
    const char* out_path = argv[3];
    int warmup = (argc > 4) ? atoi(argv[4]) : 2;
    int timed  = (argc > 5) ? atoi(argv[5]) : 5;
    const char* csv = (argc > 6) ? argv[6] : NULL;

    bench_print_hardware();

    image_t* src = image_load_grey(in_path);
    if (!src) return 2;
    printf("Loaded %s: %dx%d (%d channel)\n", in_path, src->width, src->height, src->channels);

    image_t* dst = image_alloc(src->width, src->height, src->channels);

    FILE* csvf = NULL;
    if (csv) {
        int new_file = 0;
        FILE* probe = fopen(csv, "r");
        if (!probe) new_file = 1; else fclose(probe);
        csvf = fopen(csv, "a");
        if (csvf && new_file) bench_emit_csv_header(csvf);
    }

    int run_all = (strcmp(fname, "all") == 0);

    for (int i = 0; FILTERS[i].name; i++) {
        if (!run_all && strcmp(FILTERS[i].name, fname) != 0) continue;

        ctx_t ctx = { src, dst, FILTERS[i].run };
        bench_result_t r;
        bench_run(work_fn, &ctx, warmup, timed, &r);

        char label[64];
        snprintf(label, sizeof(label), "seq/%s", FILTERS[i].name);
        bench_print_result(label, &r);

        char out_named[1024];
        if (run_all) {
            const char* dot = strrchr(out_path, '.');
            if (dot) {
                int base_len = (int)(dot - out_path);
                snprintf(out_named, sizeof(out_named), "%.*s_%s%s",
                         base_len, out_path, FILTERS[i].name, dot);
            } else {
                snprintf(out_named, sizeof(out_named), "%s_%s.png", out_path, FILTERS[i].name);
            }
        } else {
            snprintf(out_named, sizeof(out_named), "%s", out_path);
        }
        image_save_png(out_named, dst);
        printf("  -> %s\n", out_named);

        if (csvf) bench_emit_csv_row(csvf, "seq", FILTERS[i].name, 1, src->width, src->height, &r);
    }

    if (csvf) fclose(csvf);
    image_free(src);
    image_free(dst);
    return 0;
}
