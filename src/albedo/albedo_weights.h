#ifndef ALBEDO_WEIGHTS_H
#define ALBEDO_WEIGHTS_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#define ALBEDO_NEURON_WEIGHT_MASK_WIDTH 3
#define ALBEDO_NEURON_WEIGHT_MASK_HEIGHT 3

typedef struct AlbedoNeuronWeight {
    float mask[ALBEDO_NEURON_WEIGHT_MASK_WIDTH][ALBEDO_NEURON_WEIGHT_MASK_HEIGHT];
} AlbedoNeuronWeight;

typedef struct AlbedoWeightsLayer {
    unsigned int width;
    unsigned int height;

    AlbedoNeuronWeight* neurons;
} AlbedoWeightsLayer;

AlbedoWeightsLayer* albedo_new_weights_layer(unsigned int width, unsigned int height) {
    AlbedoWeightsLayer* layer = (AlbedoWeightsLayer*) malloc(sizeof(AlbedoWeightsLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height;

    layer->neurons = (AlbedoNeuronWeight*) malloc(size * sizeof(AlbedoNeuronWeight));

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {

            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w)
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h)
                    layer->neurons[x + y*width].mask[w][h] = 0.0f;
            
            layer->neurons[x + y*width].mask[1][0] = albedo_randf(-1.0f, 1.0f);
            layer->neurons[x + y*width].mask[0][1] = albedo_randf(-1.0f, 1.0f);
            layer->neurons[x + y*width].mask[2][1] = albedo_randf(-1.0f, 1.0f);
            layer->neurons[x + y*width].mask[1][2] = albedo_randf(-1.0f, 1.0f);
        }
    }

    return layer;
}

void albedo_tune_weights_layer(AlbedoWeightsLayer* weights, float error) {
    unsigned int width = weights->width;
    unsigned int height = weights->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                    weights->neurons[x + y*width].mask[w][h] += error * albedo_randf(-1.0f, 1.0f);
                    albedo_clampf(weights->neurons[x + y*width].mask[w][h], -1.0, 1.0);
                }
            }
        }
    }
}

void albedo_free_weights_layer(AlbedoWeightsLayer* weights) {
    free(weights->neurons);
    free(weights);
}

#endif