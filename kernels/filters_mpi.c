#include "filters.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int rank, nranks;
    int W, H;
    int halo;
    int local_h;        /* real rows on this rank */
    int local_y_start;  /* global index of this rank's first real row */
    int* sendcounts;    /* bytes per rank for Scatterv */
    int* displs;        /* byte offsets per rank for Scatterv */
    uint8_t* local_buf; /* (local_h + 2*halo) * W bytes; real rows at offset halo*W */
} mpi_strip_t;

static void strip_compute_layout(int H, int nranks, int rank, int* local_h, int* y_start) {
    int chunk = H / nranks;
    int rem = H % nranks;
    *y_start = rank * chunk + (rank < rem ? rank : rem);
    *local_h = chunk + (rank < rem ? 1 : 0);
}

static void strip_init(mpi_strip_t* s, int W, int H, int halo) {
    MPI_Comm_rank(MPI_COMM_WORLD, &s->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &s->nranks);
    s->W = W;
    s->H = H;
    s->halo = halo;
    strip_compute_layout(H, s->nranks, s->rank, &s->local_h, &s->local_y_start);

    s->sendcounts = (int*)calloc(s->nranks, sizeof(int));
    s->displs = (int*)calloc(s->nranks, sizeof(int));
    for (int r = 0; r < s->nranks; r++) {
        int lh, y0;
        strip_compute_layout(H, s->nranks, r, &lh, &y0);
        s->sendcounts[r] = lh * W;
        s->displs[r] = y0 * W;
    }
    s->local_buf = (uint8_t*)calloc((size_t)(s->local_h + 2 * halo) * W, 1);
}

static void strip_destroy(mpi_strip_t* s) {
    free(s->sendcounts);
    free(s->displs);
    free(s->local_buf);
}

static void strip_scatter(mpi_strip_t* s, const uint8_t* full_or_null) {
    MPI_Scatterv(full_or_null, s->sendcounts, s->displs, MPI_BYTE,
                 s->local_buf + s->halo * s->W, s->local_h * s->W, MPI_BYTE,
                 0, MPI_COMM_WORLD);
}

static void strip_gather(mpi_strip_t* s, uint8_t* full_or_null, const uint8_t* local_real_rows) {
    MPI_Gatherv(local_real_rows, s->local_h * s->W, MPI_BYTE,
                full_or_null, s->sendcounts, s->displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
}

/* Exchange halo rows with neighbours; for global boundaries, mirror own rows. */
static void strip_halo_exchange(mpi_strip_t* s) {
    const int W = s->W;
    const int halo = s->halo;
    const int local_h = s->local_h;
    uint8_t* buf = s->local_buf;

    int up   = (s->rank == 0) ? MPI_PROC_NULL : s->rank - 1;
    int down = (s->rank == s->nranks - 1) ? MPI_PROC_NULL : s->rank + 1;

    /* Send our top `halo` real rows up (becomes their bottom halo);
       Receive into our top halo (rows 0..halo-1).                  */
    uint8_t* my_top    = buf + halo * W;
    uint8_t* my_bottom = buf + (halo + local_h - halo) * W;
    uint8_t* top_halo    = buf + 0;
    uint8_t* bottom_halo = buf + (halo + local_h) * W;

    MPI_Sendrecv(my_top,    halo * W, MPI_BYTE, up,   0,
                 bottom_halo, halo * W, MPI_BYTE, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(my_bottom, halo * W, MPI_BYTE, down, 1,
                 top_halo,    halo * W, MPI_BYTE, up,   1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Mirror at global boundaries.
       Top halo: local row k (global row -(halo-k)) mirrors local row 2*halo - k
                 (global row halo - k). Matches seq MIRROR(i,n) = -i for i < 0.
       Bottom halo: local row (halo+local_h+k) (global row H+k) mirrors local row
                    halo + local_h - 2 - k (global row H-2-k). Matches MIRROR for i >= n. */
    if (s->rank == 0) {
        for (int k = 0; k < halo; k++) {
            int src_row = 2 * halo - k;
            memcpy(buf + k * W, buf + src_row * W, W);
        }
    }
    if (s->rank == s->nranks - 1) {
        for (int k = 0; k < halo; k++) {
            int src_row = halo + local_h - 2 - k;
            memcpy(buf + (halo + local_h + k) * W, buf + src_row * W, W);
        }
    }
}

/* ====================== Gaussian blur 5x5 ====================== */

static const float G5[5][5] = {
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 6.0f/256,24.0f/256, 36.0f/256,24.0f/256, 6.0f/256},
    { 4.0f/256,16.0f/256, 24.0f/256,16.0f/256, 4.0f/256},
    { 1.0f/256, 4.0f/256,  6.0f/256, 4.0f/256, 1.0f/256}
};

void gaussian_blur_5x5_mpi(const image_t* src, image_t* dst) {
    int W = src->width, H = src->height;
    mpi_strip_t s;
    strip_init(&s, W, H, 2);

    const uint8_t* full = (s.rank == 0) ? src->data : NULL;
    strip_scatter(&s, full);
    strip_halo_exchange(&s);

    uint8_t* local_dst = (uint8_t*)malloc((size_t)s.local_h * W);
    int halo = s.halo;
    for (int ly = 0; ly < s.local_h; ly++) {
        int by = ly + halo; /* row in local_buf coordinates */
        for (int x = 0; x < W; x++) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                int yy = by + ky;
                for (int kx = -2; kx <= 2; kx++) {
                    int xx = MIRROR(x + kx, W);
                    acc += s.local_buf[yy * W + xx] * G5[ky + 2][kx + 2];
                }
            }
            local_dst[ly * W + x] = CLAMP_U8(acc + 0.5f);
        }
    }

    uint8_t* full_dst = (s.rank == 0) ? dst->data : NULL;
    strip_gather(&s, full_dst, local_dst);

    free(local_dst);
    strip_destroy(&s);
}

/* ============================ Sobel ============================ */

void sobel_edges_mpi(const image_t* src, image_t* dst) {
    int W = src->width, H = src->height;
    mpi_strip_t s;
    strip_init(&s, W, H, 1);

    const uint8_t* full = (s.rank == 0) ? src->data : NULL;
    strip_scatter(&s, full);
    strip_halo_exchange(&s);

    static const int Kx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int Ky[3][3] = {{ 1,2,1},{ 0,0,0},{-1,-2,-1}};
    int halo = s.halo;

    float* mag = (float*)malloc((size_t)s.local_h * W * sizeof(float));
    float local_max = 0.0f;
    for (int ly = 0; ly < s.local_h; ly++) {
        int by = ly + halo;
        for (int x = 0; x < W; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                int yy = by + ky;
                for (int kx = -1; kx <= 1; kx++) {
                    int xx = MIRROR(x + kx, W);
                    float v = s.local_buf[yy * W + xx];
                    gx += v * Kx[ky + 1][kx + 1];
                    gy += v * Ky[ky + 1][kx + 1];
                }
            }
            float m = sqrtf(gx*gx + gy*gy);
            mag[ly * W + x] = m;
            if (m > local_max) local_max = m;
        }
    }

    float global_max = 0.0f;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    float scale = (global_max > 0.0f) ? (255.0f / global_max) : 1.0f;

    uint8_t* local_dst = (uint8_t*)malloc((size_t)s.local_h * W);
    for (int i = 0; i < s.local_h * W; i++) {
        local_dst[i] = CLAMP_U8(mag[i] * scale + 0.5f);
    }
    free(mag);

    uint8_t* full_dst = (s.rank == 0) ? dst->data : NULL;
    strip_gather(&s, full_dst, local_dst);

    free(local_dst);
    strip_destroy(&s);
}

/* ========================== Unsharp ============================ */

void unsharp_mask_mpi(const image_t* src, image_t* dst, float amount) {
    int W = src->width, H = src->height;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    image_t* blurred = NULL;
    if (rank == 0) blurred = image_alloc(W, H, 1);
    image_t blurred_meta = { W, H, 1, (rank == 0) ? blurred->data : NULL };
    gaussian_blur_5x5_mpi(src, &blurred_meta);

    /* Combine on rank 0 only (pointwise; cheap to do without scatter) */
    if (rank == 0) {
        int N = W * H;
        for (int i = 0; i < N; i++) {
            int sv = src->data[i];
            int bv = blurred->data[i];
            float v = sv + amount * (sv - bv);
            dst->data[i] = CLAMP_U8(v + 0.5f);
        }
        image_free(blurred);
    }
}

/* ====================== Brightness/contrast ===================== */

void brightness_contrast_mpi(const image_t* src, image_t* dst, float brightness, float contrast) {
    int W = src->width, H = src->height;
    mpi_strip_t s;
    strip_init(&s, W, H, 0);

    const uint8_t* full = (s.rank == 0) ? src->data : NULL;
    strip_scatter(&s, full);

    int N = s.local_h * W;
    uint8_t* local_dst = (uint8_t*)malloc(N);
    uint8_t* in_rows = s.local_buf; /* halo=0 so real rows are at offset 0 */
    for (int i = 0; i < N; i++) {
        float v = (in_rows[i] - 128.0f) * contrast + 128.0f + brightness;
        local_dst[i] = CLAMP_U8(v + 0.5f);
    }

    uint8_t* full_dst = (s.rank == 0) ? dst->data : NULL;
    strip_gather(&s, full_dst, local_dst);

    free(local_dst);
    strip_destroy(&s);
}

/* ===================== Histogram equalisation =================== */

void histogram_equalize_mpi(const image_t* src, image_t* dst) {
    int W = src->width, H = src->height;
    mpi_strip_t s;
    strip_init(&s, W, H, 0);

    const uint8_t* full = (s.rank == 0) ? src->data : NULL;
    strip_scatter(&s, full);

    int N = s.local_h * W;
    uint32_t local_hist[256] = {0};
    uint8_t* in_rows = s.local_buf;
    for (int i = 0; i < N; i++) local_hist[in_rows[i]]++;

    uint32_t global_hist[256];
    MPI_Allreduce(local_hist, global_hist, 256, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

    uint8_t lut[256];
    histogram_to_lut_seq(global_hist, W * H, lut);

    uint8_t* local_dst = (uint8_t*)malloc(N);
    for (int i = 0; i < N; i++) local_dst[i] = lut[in_rows[i]];

    uint8_t* full_dst = (s.rank == 0) ? dst->data : NULL;
    strip_gather(&s, full_dst, local_dst);

    free(local_dst);
    strip_destroy(&s);
}
