#ifndef ALBEDO_WEIGHTS_H
#define ALBEDO_WEIGHTS_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "../albedo_utils.h"

typedef struct AlbedoNeuronKernel {
    kiwi_fixed_t kernel[3][3];
} AlbedoNeuronKernel;

typedef struct AlbedoWeightsLayer {
    unsigned int width;
    unsigned int height;

    AlbedoNeuronKernel* weights;
} AlbedoWeightsLayer;

typedef struct AlbedoNeuronValue { 
    unsigned int x; 
    unsigned int y; 
    kiwi_fixed_t value;
} AlbedoNeuronValue;

AlbedoWeightsLayer* albedo_new_weights_layer(unsigned int width, unsigned int height);
AlbedoWeightsLayer* albedo_new_weights_layer_clamped(unsigned int width, unsigned int height, kiwi_fixed_t min, kiwi_fixed_t max);
AlbedoWeightsLayer* albedo_copy_weights_layer(AlbedoWeightsLayer* src);

void albedo_free_weights_layer(AlbedoWeightsLayer* weights);

void albedo_weights_layer_add(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another);
void albedo_weights_layer_subtract(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another);
void albedo_weights_layer_multiply(AlbedoWeightsLayer* target, AlbedoWeightsLayer* another);
void albedo_weights_layer_clamp(AlbedoWeightsLayer* target, kiwi_fixed_t min, kiwi_fixed_t max);

void albedo_tune_weights_layer(AlbedoWeightsLayer* weights, kiwi_fixed_t error);

#endif