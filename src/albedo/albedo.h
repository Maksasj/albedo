#ifndef ALBEDO_H
#define ALBEDO_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

typedef enum GradientDirection {
    UP = 0,
    RIGHT = 1,
    DOWN = 2,
    LEFT = 3
} GradientDirection;

typedef struct GradientCell {
    float value[4];
} GradientCell;

typedef struct GradientLayer {
    unsigned int width;
    unsigned int height;

    GradientCell* cells;
} GradientLayer;

GradientLayer* create_gradient_layer(unsigned int width, unsigned int height) {
    GradientLayer* layer = (GradientLayer*) malloc(sizeof(GradientLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height;

    layer->cells = (GradientCell*) malloc(size * sizeof(GradientCell));

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            layer->cells[x + y*width].value[UP]     = (rand() % 255 - 128) / 128.0f;
            layer->cells[x + y*width].value[RIGHT]  = (rand() % 255 - 128) / 128.0f;
            layer->cells[x + y*width].value[DOWN]   = (rand() % 255 - 128) / 128.0f;
            layer->cells[x + y*width].value[LEFT]   = (rand() % 255 - 128) / 128.0f;
        }
    }

    return layer;
}

#endif