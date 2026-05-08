#include "filters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static void apply_schedule(const char* kind) {
    omp_sched_t s = omp_sched_static;
    int chunk = 0;
    if (kind) {
        if (strcmp(kind, "dynamic") == 0)      s = omp_sched_dynamic;
        else if (strcmp(kind, "guided") == 0)  s = omp_sched_guided;
        else if (strcmp(kind, "auto") == 0)    s = omp_sched_auto;
        else                                    s = omp_sched_static;
    }
    omp_set_schedule(s, chunk);
}

static const float G5[5][5] = {
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 6.0f/256,24.0f/256, 36.0f/256,24.0f/256, 6.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256}
};

void gaussian_blur_5x5_omp(const image_t* src, image_t* dst, int num_threads, const char* schedule_kind) {
    const int W = src->width, H = src->height;
    omp_set_num_threads(num_threads);
    apply_schedule(schedule_kind);

    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                int yy = MIRROR(y + ky, H);
                for (int kx = -2; kx <= 2; kx++) {
                    int xx = MIRROR(x + kx, W);
                    acc += src->data[yy * W + xx] * G5[ky + 2][kx + 2];
                }
            }
            dst->data[y * W + x] = CLAMP_U8(acc + 0.5f);
        }
    }
}

void sobel_edges_omp(const image_t* src, image_t* dst, int num_threads, const char* schedule_kind) {
    const int W = src->width, H = src->height;
    static const int Kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int Ky[3][3] = {{ 1,2,1},{ 0,0,0},{-1,-2,-1}};
    float* mag = (float*)malloc((size_t)W * H * sizeof(float));
    float maxv = 0.0f;

    omp_set_num_threads(num_threads);
    apply_schedule(schedule_kind);

    #pragma omp parallel for schedule(runtime) reduction(max:maxv)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                int yy = MIRROR(y + ky, H);
                for (int kx = -1; kx <= 1; kx++) {
                    int xx = MIRROR(x + kx, W);
                    float v = src->data[yy * W + xx];
                    gx += v * Kx[ky + 1][kx + 1];
                    gy += v * Ky[ky + 1][kx + 1];
                }
            }
            float m = sqrtf(gx*gx + gy*gy);
            mag[y * W + x] = m;
            if (m > maxv) maxv = m;
        }
    }

    float scale = (maxv > 0.0f) ? (255.0f / maxv) : 1.0f;
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < W * H; i++) {
        dst->data[i] = CLAMP_U8(mag[i] * scale + 0.5f);
    }
    free(mag);
}

void unsharp_mask_omp(const image_t* src, image_t* dst, float amount, int num_threads, const char* schedule_kind) {
    image_t* blurred = image_alloc(src->width, src->height, src->channels);
    gaussian_blur_5x5_omp(src, blurred, num_threads, schedule_kind);

    const int N = src->width * src->height;
    omp_set_num_threads(num_threads);
    apply_schedule(schedule_kind);

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        int s = src->data[i];
        int b = blurred->data[i];
        float v = s + amount * (s - b);
        dst->data[i] = CLAMP_U8(v + 0.5f);
    }
    image_free(blurred);
}

void brightness_contrast_omp(const image_t* src, image_t* dst, float brightness, float contrast, int num_threads, const char* schedule_kind) {
    const int N = src->width * src->height;
    omp_set_num_threads(num_threads);
    apply_schedule(schedule_kind);

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        float v = (src->data[i] - 128.0f) * contrast + 128.0f + brightness;
        dst->data[i] = CLAMP_U8(v + 0.5f);
    }
}

void histogram_equalize_omp(const image_t* src, image_t* dst, int num_threads, const char* schedule_kind) {
    const int N = src->width * src->height;
    uint32_t hist[256] = {0};
    uint8_t lut[256];

    omp_set_num_threads(num_threads);
    apply_schedule(schedule_kind);

    #pragma omp parallel for schedule(runtime) reduction(+:hist[:256])
    for (int i = 0; i < N; i++) {
        hist[src->data[i]]++;
    }

    histogram_to_lut_seq(hist, N, lut);

    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < N; i++) {
        dst->data[i] = lut[src->data[i]];
    }
}
