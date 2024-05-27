#ifndef ALBEDO_NEURONS_H
#define ALBEDO_NEURONS_H

#include <stdlib.h>
#include <string.h>

#define PEACH_IMPLEMENTATION
#include "kiwi/kiwi.h"

typedef struct AlbedoNeuronLayer {
    unsigned int width;
    unsigned int height;

    kiwi_fixed_t* neurons;
} AlbedoNeuronLayer;

AlbedoNeuronLayer* albedo_new_neuron_layer(unsigned int width, unsigned int height);
AlbedoNeuronLayer* albedo_copy_neuron_layer(AlbedoNeuronLayer* src);

void albedo_set_neuron_layer_value(AlbedoNeuronLayer* layer, kiwi_fixed_t value);
void albedo_reset_neuron_layer_value(AlbedoNeuronLayer* layer);

void albedo_free_neuron_layer(AlbedoNeuronLayer* state);

#endif