#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>
#include <stdio.h>

double now_seconds(void);

typedef struct {
    int warmup_runs;
    int timed_runs;
    double mean_seconds;
    double stddev_seconds;
    double min_seconds;
    double max_seconds;
} bench_result_t;

void bench_run(void (*work)(void* ctx), void* ctx, int warmup, int timed, bench_result_t* out);
void bench_print_result(const char* label, const bench_result_t* r);
void bench_print_hardware(void);

void bench_emit_csv_header(FILE* f);
void bench_emit_csv_row(FILE* f, const char* paradigm, const char* filter, int threads, int width, int height, const bench_result_t* r);

#endif
