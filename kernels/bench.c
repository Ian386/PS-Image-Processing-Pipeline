#define _POSIX_C_SOURCE 199309L
#include "bench.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/utsname.h>
#endif

double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void bench_run(void (*work)(void* ctx), void* ctx, int warmup, int timed, bench_result_t* out) {
    for (int i = 0; i < warmup; i++) work(ctx);

    double* samples = (double*)malloc(sizeof(double) * timed);
    double sum = 0.0;
    double mn = 1e18, mx = 0.0;
    for (int i = 0; i < timed; i++) {
        double t0 = now_seconds();
        work(ctx);
        double t1 = now_seconds();
        double dt = t1 - t0;
        samples[i] = dt;
        sum += dt;
        if (dt < mn) mn = dt;
        if (dt > mx) mx = dt;
    }
    double mean = sum / timed;
    double var = 0.0;
    for (int i = 0; i < timed; i++) {
        double d = samples[i] - mean;
        var += d * d;
    }
    var /= timed;
    free(samples);

    out->warmup_runs = warmup;
    out->timed_runs = timed;
    out->mean_seconds = mean;
    out->stddev_seconds = sqrt(var);
    out->min_seconds = mn;
    out->max_seconds = mx;
}

void bench_print_result(const char* label, const bench_result_t* r) {
    printf("[%s] mean=%.4fs stddev=%.4fs min=%.4fs max=%.4fs (warmup=%d, timed=%d)\n",
        label, r->mean_seconds, r->stddev_seconds, r->min_seconds, r->max_seconds,
        r->warmup_runs, r->timed_runs);
}

static long read_meminfo_kb(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    long val = 0;
    size_t klen = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, klen) == 0) {
            sscanf(line + klen, "%ld", &val);
            break;
        }
    }
    fclose(f);
    return val;
}

static void read_cpu_model(char* out, size_t n) {
    FILE* f = fopen("/proc/cpuinfo", "r");
    out[0] = '\0';
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                colon += 2;
                size_t L = strlen(colon);
                if (L > 0 && colon[L-1] == '\n') colon[L-1] = '\0';
                strncpy(out, colon, n - 1);
                out[n - 1] = '\0';
            }
            break;
        }
    }
    fclose(f);
}

void bench_print_hardware(void) {
    char cpu[256];
    read_cpu_model(cpu, sizeof(cpu));
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    long ram_kb = read_meminfo_kb("MemTotal:");
    printf("=== Hardware ===\n");
    printf("CPU model     : %s\n", cpu[0] ? cpu : "unknown");
    printf("Logical cores : %ld\n", ncpu);
    printf("RAM total     : %.2f GB\n", ram_kb / (1024.0 * 1024.0));
#ifdef __linux__
    struct utsname u;
    if (uname(&u) == 0) {
        printf("OS            : %s %s (%s)\n", u.sysname, u.release, u.machine);
    }
#endif
    printf("================\n");
}

void bench_emit_csv_header(FILE* f) {
    fprintf(f, "paradigm,filter,threads,width,height,pixels,mean_s,stddev_s,min_s,max_s,warmup,timed\n");
}

void bench_emit_csv_row(FILE* f, const char* paradigm, const char* filter, int threads, int width, int height, const bench_result_t* r) {
    fprintf(f, "%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d\n",
        paradigm, filter, threads, width, height, width * height,
        r->mean_seconds, r->stddev_seconds, r->min_seconds, r->max_seconds,
        r->warmup_runs, r->timed_runs);
}
