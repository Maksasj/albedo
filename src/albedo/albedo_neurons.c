#include "albedo_neurons.h"

AlbedoNeuronLayer* albedo_new_neuron_layer(unsigned int width, unsigned int height) {
    AlbedoNeuronLayer* layer = (AlbedoNeuronLayer*) malloc(sizeof(AlbedoNeuronLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height * sizeof(float);

    layer->neurons = (float*) malloc(size);
    memset(layer->neurons, 0, size);

    return layer;
}

void albedo_free_neuron_layer(AlbedoNeuronLayer* state) {
    free(state->neurons);
    free(state);   
}