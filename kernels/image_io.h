#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include "filters.h"

image_t* image_load_grey(const char* path);
int image_save_png(const char* path, const image_t* img);

#endif
