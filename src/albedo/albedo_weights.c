#include "albedo_weights.h"

AlbedoWeightsLayer* albedo_new_weights_layer_clamped(unsigned int width, unsigned int height, float min, float max) {
    AlbedoWeightsLayer* layer = (AlbedoWeightsLayer*) malloc(sizeof(AlbedoWeightsLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height;

    layer->weights = (AlbedoNeuronKernel*) malloc(size * sizeof(AlbedoNeuronKernel));

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < 3; ++w)
                for(int h = 0; h < 3; ++h)
                    layer->weights[x + y*width].kernel[w][h] = albedo_randf(min, max);
        }
    }

    return layer;
}

AlbedoWeightsLayer* albedo_new_weights_layer(unsigned int width, unsigned int height) {
    return albedo_new_weights_layer_clamped(width, height, -1.0, 1.0);
}

AlbedoWeightsLayer* albedo_copy_weights_layer(AlbedoWeightsLayer* src) {
    AlbedoWeightsLayer* layer = (AlbedoWeightsLayer*) malloc(sizeof(AlbedoWeightsLayer));

    layer->width = src->width;
    layer->height = src->height;

    unsigned int size = layer->width * layer->height * sizeof(AlbedoNeuronKernel);

    layer->weights = (AlbedoNeuronKernel*) malloc(size);
    memcpy(layer->weights, src->weights, size);

    return layer;
}

void albedo_free_weights_layer(AlbedoWeightsLayer* weights) {
    free(weights->weights);
    free(weights);
}

void albedo_weights_layer_add(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another) {
    unsigned int width = target->width;
    unsigned int height = target->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < 3; ++w) {
                for(int h = 0; h < 3; ++h) {
                    unsigned int index = x + y*width;

                    target->weights[index].kernel[w][h] += another->weights[index].kernel[w][h];
                }
            }
        }
    }
}

void albedo_weights_layer_subtract(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another) {
    unsigned int width = target->width;
    unsigned int height = target->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < 3; ++w) {
                for(int h = 0; h < 3; ++h) {
                    unsigned int index = x + y*width;

                    target->weights[index].kernel[w][h] -= another->weights[index].kernel[w][h];
                }
            }
        }
    }
}

void albedo_weights_layer_multiply(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another) {
    unsigned int width = target->width;
    unsigned int height = target->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < 3; ++w) {
                for(int h = 0; h < 3; ++h) {
                    unsigned int index = x + y*width;

                    target->weights[index].kernel[w][h] *= another->weights[index].kernel[w][h];
                }
            }
        }
    }
}

void albedo_weights_layer_clamp(AlbedoWeightsLayer* target, float min, float max) {
    unsigned int width = target->width;
    unsigned int height = target->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < 3; ++w) {
                for(int h = 0; h < 3; ++h) {
                    unsigned int index = x + y*width;

                    target->weights[index].kernel[w][h] = albedo_clampf(target->weights[index].kernel[w][h], min, max);
                }
            }
        }
    }
}

void albedo_tune_weights_layer(AlbedoWeightsLayer* weights, float error) {
    unsigned int width = weights->width;
    unsigned int height = weights->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            // How we edit our kernel
            for(int w = 0; w < 3; ++w) {
                for(int h = 0; h < 3; ++h) {
                    unsigned int index = x + y*width;
                    float value = weights->weights[index].kernel[w][h]; 

                    if(value == 0.0)
                        continue;

                    value += error * albedo_randf(-1.0f, 1.0f);
                    weights->weights[index].kernel[w][h] = albedo_clampf(value, -1.0f, 1.0f);
                }
            }
        }
    }
}

