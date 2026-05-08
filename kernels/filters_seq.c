#include "filters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

image_t* image_alloc(int width, int height, int channels) {
    image_t* img = (image_t*)malloc(sizeof(image_t));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t*)calloc((size_t)width * height * channels, sizeof(uint8_t));
    if (!img->data) { free(img); return NULL; }
    return img;
}

void image_free(image_t* img) {
    if (!img) return;
    free(img->data);
    free(img);
}

size_t image_bytes(const image_t* img) {
    return (size_t)img->width * img->height * img->channels;
}

image_t* image_clone(const image_t* src) {
    image_t* dst = image_alloc(src->width, src->height, src->channels);
    if (!dst) return NULL;
    memcpy(dst->data, src->data, image_bytes(src));
    return dst;
}

static const float G5[5][5] = {
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 6.0f/256,24.0f/256, 36.0f/256,24.0f/256, 6.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256}
};

void gaussian_blur_5x5_seq(const image_t* src, image_t* dst) {
    const int W = src->width, H = src->height;
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

void sobel_edges_seq(const image_t* src, image_t* dst) {
    const int W = src->width, H = src->height;
    static const int Kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int Ky[3][3] = {{ 1,2,1},{ 0,0,0},{-1,-2,-1}};
    float* mag = (float*)malloc((size_t)W * H * sizeof(float));
    float maxv = 0.0f;
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
    for (int i = 0; i < W * H; i++) {
        dst->data[i] = CLAMP_U8(mag[i] * scale + 0.5f);
    }
    free(mag);
}

void unsharp_mask_seq(const image_t* src, image_t* dst, float amount) {
    image_t* blurred = image_alloc(src->width, src->height, src->channels);
    gaussian_blur_5x5_seq(src, blurred);
    const int N = src->width * src->height;
    for (int i = 0; i < N; i++) {
        int s = src->data[i];
        int b = blurred->data[i];
        float v = s + amount * (s - b);
        dst->data[i] = CLAMP_U8(v + 0.5f);
    }
    image_free(blurred);
}

void brightness_contrast_seq(const image_t* src, image_t* dst, float brightness, float contrast) {
    const int N = src->width * src->height;
    for (int i = 0; i < N; i++) {
        float v = (src->data[i] - 128.0f) * contrast + 128.0f + brightness;
        dst->data[i] = CLAMP_U8(v + 0.5f);
    }
}

void compute_histogram_seq(const image_t* src, uint32_t hist[256]) {
    memset(hist, 0, 256 * sizeof(uint32_t));
    const int N = src->width * src->height;
    for (int i = 0; i < N; i++) {
        hist[src->data[i]]++;
    }
}

void histogram_to_lut_seq(const uint32_t hist[256], int total_pixels, uint8_t lut[256]) {
    uint32_t cdf[256];
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) cdf[i] = cdf[i-1] + hist[i];
    uint32_t cdf_min = 0;
    for (int i = 0; i < 256; i++) { if (cdf[i] != 0) { cdf_min = cdf[i]; break; } }
    uint32_t denom = (uint32_t)total_pixels - cdf_min;
    if (denom == 0) denom = 1;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] < cdf_min) lut[i] = 0;
        else lut[i] = (uint8_t)(((float)(cdf[i] - cdf_min) / denom) * 255.0f + 0.5f);
    }
}

void apply_lut_seq(const image_t* src, image_t* dst, const uint8_t lut[256]) {
    const int N = src->width * src->height;
    for (int i = 0; i < N; i++) dst->data[i] = lut[src->data[i]];
}

void histogram_equalize_seq(const image_t* src, image_t* dst) {
    uint32_t hist[256];
    uint8_t lut[256];
    compute_histogram_seq(src, hist);
    histogram_to_lut_seq(hist, src->width * src->height, lut);
    apply_lut_seq(src, dst, lut);
}
