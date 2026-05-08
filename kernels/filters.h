#ifndef FILTERS_H
#define FILTERS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int width;
    int height;
    int channels;
    uint8_t* data;
} image_t;

image_t* image_alloc(int width, int height, int channels);
void image_free(image_t* img);
image_t* image_clone(const image_t* src);
size_t image_bytes(const image_t* img);

#define CLAMP_U8(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (uint8_t)(x)))
#define MIRROR(i, n) ((i) < 0 ? -(i) : ((i) >= (n) ? (2*(n) - (i) - 2) : (i)))

void gaussian_blur_5x5_seq(const image_t* src, image_t* dst);
void sobel_edges_seq(const image_t* src, image_t* dst);
void unsharp_mask_seq(const image_t* src, image_t* dst, float amount);
void brightness_contrast_seq(const image_t* src, image_t* dst, float brightness, float contrast);
void histogram_equalize_seq(const image_t* src, image_t* dst);

void compute_histogram_seq(const image_t* src, uint32_t hist[256]);
void histogram_to_lut_seq(const uint32_t hist[256], int total_pixels, uint8_t lut[256]);
void apply_lut_seq(const image_t* src, image_t* dst, const uint8_t lut[256]);

#ifdef _OPENMP
void gaussian_blur_5x5_omp(const image_t* src, image_t* dst, int num_threads, const char* schedule_kind);
void sobel_edges_omp(const image_t* src, image_t* dst, int num_threads, const char* schedule_kind);
void unsharp_mask_omp(const image_t* src, image_t* dst, float amount, int num_threads, const char* schedule_kind);
void brightness_contrast_omp(const image_t* src, image_t* dst, float brightness, float contrast, int num_threads, const char* schedule_kind);
void histogram_equalize_omp(const image_t* src, image_t* dst, int num_threads, const char* schedule_kind);
#endif

void gaussian_blur_5x5_pthread(const image_t* src, image_t* dst, int num_threads);
void sobel_edges_pthread(const image_t* src, image_t* dst, int num_threads);
void unsharp_mask_pthread(const image_t* src, image_t* dst, float amount, int num_threads);
void brightness_contrast_pthread(const image_t* src, image_t* dst, float brightness, float contrast, int num_threads);
void histogram_equalize_pthread(const image_t* src, image_t* dst, int num_threads);

void gaussian_blur_5x5_mpi(const image_t* src, image_t* dst);
void sobel_edges_mpi(const image_t* src, image_t* dst);
void unsharp_mask_mpi(const image_t* src, image_t* dst, float amount);
void brightness_contrast_mpi(const image_t* src, image_t* dst, float brightness, float contrast);
void histogram_equalize_mpi(const image_t* src, image_t* dst);

#endif
