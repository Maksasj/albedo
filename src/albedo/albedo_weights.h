#ifndef ALBEDO_WEIGHTS_H
#define ALBEDO_WEIGHTS_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "albedo_utils.h"

#define ALBEDO_NEURON_WEIGHT_MASK_WIDTH 5
#define ALBEDO_NEURON_WEIGHT_MASK_HEIGHT 5

typedef struct AlbedoNeuronWeight {
    float mask[ALBEDO_NEURON_WEIGHT_MASK_WIDTH][ALBEDO_NEURON_WEIGHT_MASK_HEIGHT];
} AlbedoNeuronWeight;

typedef struct AlbedoWeightsLayer {
    unsigned int width;
    unsigned int height;

    AlbedoNeuronWeight* neurons;
} AlbedoWeightsLayer;

AlbedoWeightsLayer* albedo_new_weights_layer(unsigned int width, unsigned int height);
AlbedoWeightsLayer* albedo_new_weights_layer_clamped(unsigned int width, unsigned int height, float min, float max);
AlbedoWeightsLayer* albedo_copy_weights_layer(AlbedoWeightsLayer* src);

void albedo_free_weights_layer(AlbedoWeightsLayer* weights);

void albedo_weights_layer_add(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another);
void albedo_weights_layer_multiply(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another);
void albedo_weights_layer_clamp(AlbedoWeightsLayer* target, float min, float max);

void albedo_tune_weights_layer(AlbedoWeightsLayer* weights, float error);

#endif