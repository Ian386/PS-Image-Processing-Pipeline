#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"
#include <stdio.h>
#include <stdlib.h>

image_t* image_load_grey(const char* path) {
    int w, h, n;
    uint8_t* pixels = stbi_load(path, &w, &h, &n, 1);
    if (!pixels) {
        fprintf(stderr, "image_load_grey: failed to load '%s' (%s)\n", path, stbi_failure_reason());
        return NULL;
    }
    image_t* img = (image_t*)malloc(sizeof(image_t));
    img->width = w;
    img->height = h;
    img->channels = 1;
    img->data = pixels;
    return img;
}

int image_save_png(const char* path, const image_t* img) {
    int rc = stbi_write_png(path, img->width, img->height, img->channels, img->data, img->width * img->channels);
    if (!rc) {
        fprintf(stderr, "image_save_png: failed to write '%s'\n", path);
    }
    return rc;
}
