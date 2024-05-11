#ifndef ALBEDO_NEURONS_H
#define ALBEDO_NEURONS_H

#include <stdlib.h>
#include <string.h>

typedef struct AlbedoNeuronLayer {
    unsigned int width;
    unsigned int height;

    float* neurons;
} AlbedoNeuronLayer;

AlbedoNeuronLayer* albedo_new_neuron_layer(unsigned int width, unsigned int height);
AlbedoNeuronLayer* albedo_copy_neuron_layer(AlbedoNeuronLayer* src);

void albedo_free_neuron_layer(AlbedoNeuronLayer* state);

#endif