#include "albedo_weights.h"

AlbedoWeightsLayer* albedo_new_weights_layer_clamped(unsigned int width, unsigned int height, float min, float max) {
    AlbedoWeightsLayer* layer = (AlbedoWeightsLayer*) malloc(sizeof(AlbedoWeightsLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height;

    layer->neurons = (AlbedoNeuronWeight*) malloc(size * sizeof(AlbedoNeuronWeight));

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w)
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h)
                    layer->neurons[x + y*width].mask[w][h] = albedo_randf(min, max);
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

    unsigned int size = layer->width * layer->height * sizeof(AlbedoNeuronWeight);

    layer->neurons = (AlbedoNeuronWeight*) malloc(size);
    memcpy(layer->neurons, src->neurons, size);

    return layer;
}

void albedo_free_weights_layer(AlbedoWeightsLayer* weights) {
    free(weights->neurons);
    free(weights);
}

void albedo_weights_layer_add(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another) {
    unsigned int width = target->width;
    unsigned int height = target->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                    unsigned int index = x + y*width;

                    target->neurons[index].mask[w][h] += another->neurons[index].mask[w][h];
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
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                    unsigned int index = x + y*width;

                    target->neurons[index].mask[w][h] -= another->neurons[index].mask[w][h];
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
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                    unsigned int index = x + y*width;

                    target->neurons[index].mask[w][h] *= another->neurons[index].mask[w][h];
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
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                    unsigned int index = x + y*width;

                    target->neurons[index].mask[w][h] = albedo_clampf(target->neurons[index].mask[w][h], min, max);
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
            // How we edit our mask
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                    unsigned int index = x + y*width;
                    float value = weights->neurons[index].mask[w][h]; 

                    if(value == 0.0)
                        continue;

                    value += error * albedo_randf(-1.0f, 1.0f);
                    weights->neurons[index].mask[w][h] = albedo_clampf(value, -1.0f, 1.0f);
                }
            }
        }
    }
}

