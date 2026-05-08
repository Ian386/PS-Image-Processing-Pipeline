#include "filters.h"
#include "thread_pool.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    const image_t* src;
    image_t* dst;
    int y_start;
    int y_end;
    void* extra;
} strip_t;

static const float G5[5][5] = {
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 6.0f/256,24.0f/256, 36.0f/256,24.0f/256, 6.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256}
};

static void split_rows(int H, int n, int idx, int* y0, int* y1) {
    int chunk = H / n;
    int rem = H % n;
    *y0 = idx * chunk + (idx < rem ? idx : rem);
    *y1 = *y0 + chunk + (idx < rem ? 1 : 0);
}

static void task_blur(void* a, int wid) {
    (void)wid;
    strip_t* t = (strip_t*)a;
    const int W = t->src->width, H = t->src->height;
    for (int y = t->y_start; y < t->y_end; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                int yy = MIRROR(y + ky, H);
                for (int kx = -2; kx <= 2; kx++) {
                    int xx = MIRROR(x + kx, W);
                    acc += t->src->data[yy * W + xx] * G5[ky + 2][kx + 2];
                }
            }
            t->dst->data[y * W + x] = CLAMP_U8(acc + 0.5f);
        }
    }
}

void gaussian_blur_5x5_pthread(const image_t* src, image_t* dst, int num_threads) {
    thread_pool_t* p = pool_create(num_threads);
    int n = pool_num_workers(p);
    strip_t* strips = (strip_t*)calloc(n, sizeof(strip_t));
    for (int i = 0; i < n; i++) {
        strips[i].src = src;
        strips[i].dst = dst;
        split_rows(src->height, n, i, &strips[i].y_start, &strips[i].y_end);
        pool_submit(p, task_blur, &strips[i]);
    }
    pool_wait(p);
    pool_destroy(p);
    free(strips);
}

typedef struct {
    float* mag;
    float local_max;
} sobel_extra_t;

static void task_sobel_compute(void* a, int wid) {
    (void)wid;
    strip_t* t = (strip_t*)a;
    sobel_extra_t* ex = (sobel_extra_t*)t->extra;
    const int W = t->src->width, H = t->src->height;
    static const int Kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int Ky[3][3] = {{ 1,2,1},{ 0,0,0},{-1,-2,-1}};
    float mx = 0.0f;
    for (int y = t->y_start; y < t->y_end; y++) {
        for (int x = 0; x < W; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                int yy = MIRROR(y + ky, H);
                for (int kx = -1; kx <= 1; kx++) {
                    int xx = MIRROR(x + kx, W);
                    float v = t->src->data[yy * W + xx];
                    gx += v * Kx[ky + 1][kx + 1];
                    gy += v * Ky[ky + 1][kx + 1];
                }
            }
            float m = sqrtf(gx*gx + gy*gy);
            ex->mag[y * W + x] = m;
            if (m > mx) mx = m;
        }
    }
    ex->local_max = mx;
}

typedef struct {
    const float* mag;
    uint8_t* dst;
    int i_start;
    int i_end;
    float scale;
} norm_t;

static void task_sobel_normalize(void* a, int wid) {
    (void)wid;
    norm_t* t = (norm_t*)a;
    for (int i = t->i_start; i < t->i_end; i++) {
        float v = t->mag[i] * t->scale + 0.5f;
        t->dst[i] = CLAMP_U8(v);
    }
}

void sobel_edges_pthread(const image_t* src, image_t* dst, int num_threads) {
    const int W = src->width, H = src->height;
    float* mag = (float*)malloc((size_t)W * H * sizeof(float));

    thread_pool_t* p = pool_create(num_threads);
    int n = pool_num_workers(p);

    strip_t* strips = (strip_t*)calloc(n, sizeof(strip_t));
    sobel_extra_t* exs = (sobel_extra_t*)calloc(n, sizeof(sobel_extra_t));
    for (int i = 0; i < n; i++) {
        strips[i].src = src;
        strips[i].dst = dst;
        split_rows(H, n, i, &strips[i].y_start, &strips[i].y_end);
        exs[i].mag = mag;
        exs[i].local_max = 0.0f;
        strips[i].extra = &exs[i];
        pool_submit(p, task_sobel_compute, &strips[i]);
    }
    pool_wait(p);

    float maxv = 0.0f;
    for (int i = 0; i < n; i++) if (exs[i].local_max > maxv) maxv = exs[i].local_max;
    float scale = (maxv > 0.0f) ? (255.0f / maxv) : 1.0f;

    int N = W * H;
    norm_t* norms = (norm_t*)calloc(n, sizeof(norm_t));
    for (int i = 0; i < n; i++) {
        int chunk = N / n;
        int rem = N % n;
        norms[i].mag = mag;
        norms[i].dst = dst->data;
        norms[i].i_start = i * chunk + (i < rem ? i : rem);
        norms[i].i_end = norms[i].i_start + chunk + (i < rem ? 1 : 0);
        norms[i].scale = scale;
        pool_submit(p, task_sobel_normalize, &norms[i]);
    }
    pool_wait(p);

    pool_destroy(p);
    free(strips);
    free(exs);
    free(norms);
    free(mag);
}

typedef struct { const uint8_t* s; const uint8_t* b; uint8_t* d; int i0, i1; float amt; } usm_t;

static void task_unsharp(void* a, int wid) {
    (void)wid;
    usm_t* t = (usm_t*)a;
    for (int i = t->i0; i < t->i1; i++) {
        int s = t->s[i], b = t->b[i];
        float v = s + t->amt * (s - b);
        t->d[i] = CLAMP_U8(v + 0.5f);
    }
}

void unsharp_mask_pthread(const image_t* src, image_t* dst, float amount, int num_threads) {
    image_t* blurred = image_alloc(src->width, src->height, src->channels);
    gaussian_blur_5x5_pthread(src, blurred, num_threads);

    thread_pool_t* p = pool_create(num_threads);
    int n = pool_num_workers(p);
    int N = src->width * src->height;

    usm_t* tasks = (usm_t*)calloc(n, sizeof(usm_t));
    for (int i = 0; i < n; i++) {
        int chunk = N / n, rem = N % n;
        tasks[i].s = src->data;
        tasks[i].b = blurred->data;
        tasks[i].d = dst->data;
        tasks[i].i0 = i * chunk + (i < rem ? i : rem);
        tasks[i].i1 = tasks[i].i0 + chunk + (i < rem ? 1 : 0);
        tasks[i].amt = amount;
        pool_submit(p, task_unsharp, &tasks[i]);
    }
    pool_wait(p);
    pool_destroy(p);
    free(tasks);
    image_free(blurred);
}

typedef struct { const uint8_t* s; uint8_t* d; int i0, i1; float br; float ct; } bc_t;

static void task_bc(void* a, int wid) {
    (void)wid;
    bc_t* t = (bc_t*)a;
    for (int i = t->i0; i < t->i1; i++) {
        float v = (t->s[i] - 128.0f) * t->ct + 128.0f + t->br;
        t->d[i] = CLAMP_U8(v + 0.5f);
    }
}

void brightness_contrast_pthread(const image_t* src, image_t* dst, float brightness, float contrast, int num_threads) {
    thread_pool_t* p = pool_create(num_threads);
    int n = pool_num_workers(p);
    int N = src->width * src->height;
    bc_t* tasks = (bc_t*)calloc(n, sizeof(bc_t));
    for (int i = 0; i < n; i++) {
        int chunk = N / n, rem = N % n;
        tasks[i].s = src->data;
        tasks[i].d = dst->data;
        tasks[i].i0 = i * chunk + (i < rem ? i : rem);
        tasks[i].i1 = tasks[i].i0 + chunk + (i < rem ? 1 : 0);
        tasks[i].br = brightness;
        tasks[i].ct = contrast;
        pool_submit(p, task_bc, &tasks[i]);
    }
    pool_wait(p);
    pool_destroy(p);
    free(tasks);
}

typedef struct {
    const uint8_t* s;
    int i0;
    int i1;
    uint32_t (*local_hist)[256];
} hist_t;

static void task_hist(void* a, int wid) {
    hist_t* t = (hist_t*)a;
    uint32_t* h = t->local_hist[wid];
    for (int i = t->i0; i < t->i1; i++) h[t->s[i]]++;
}

typedef struct { const uint8_t* s; uint8_t* d; const uint8_t* lut; int i0, i1; } lut_t;

static void task_lut(void* a, int wid) {
    (void)wid;
    lut_t* t = (lut_t*)a;
    for (int i = t->i0; i < t->i1; i++) t->d[i] = t->lut[t->s[i]];
}

void histogram_equalize_pthread(const image_t* src, image_t* dst, int num_threads) {
    thread_pool_t* p = pool_create(num_threads);
    int n = pool_num_workers(p);
    int N = src->width * src->height;

    uint32_t (*local_hist)[256] = (uint32_t (*)[256])calloc(n, 256 * sizeof(uint32_t));
    hist_t* hts = (hist_t*)calloc(n, sizeof(hist_t));
    for (int i = 0; i < n; i++) {
        int chunk = N / n, rem = N % n;
        hts[i].s = src->data;
        hts[i].i0 = i * chunk + (i < rem ? i : rem);
        hts[i].i1 = hts[i].i0 + chunk + (i < rem ? 1 : 0);
        hts[i].local_hist = local_hist;
        pool_submit(p, task_hist, &hts[i]);
    }
    pool_wait(p);

    uint32_t hist[256] = {0};
    for (int w = 0; w < n; w++)
        for (int b = 0; b < 256; b++) hist[b] += local_hist[w][b];

    uint8_t lut[256];
    histogram_to_lut_seq(hist, N, lut);

    lut_t* lts = (lut_t*)calloc(n, sizeof(lut_t));
    for (int i = 0; i < n; i++) {
        int chunk = N / n, rem = N % n;
        lts[i].s = src->data;
        lts[i].d = dst->data;
        lts[i].lut = lut;
        lts[i].i0 = i * chunk + (i < rem ? i : rem);
        lts[i].i1 = lts[i].i0 + chunk + (i < rem ? 1 : 0);
        pool_submit(p, task_lut, &lts[i]);
    }
    pool_wait(p);
    pool_destroy(p);

    free(local_hist);
    free(hts);
    free(lts);
}
