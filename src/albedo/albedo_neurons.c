#include "albedo_neurons.h"

AlbedoNeuronLayer* albedo_new_neuron_layer(unsigned int width, unsigned int height) {
    AlbedoNeuronLayer* layer = (AlbedoNeuronLayer*) malloc(sizeof(AlbedoNeuronLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height * sizeof(kiwi_fixed_t);

    layer->neurons = (kiwi_fixed_t*) malloc(size);
    memset(layer->neurons, 0, size);

    return layer;
}

AlbedoNeuronLayer* albedo_copy_neuron_layer(AlbedoNeuronLayer* src) {
    AlbedoNeuronLayer* layer = (AlbedoNeuronLayer*) malloc(sizeof(AlbedoNeuronLayer));

    layer->width = src->width;
    layer->height = src->height;

    unsigned int size = layer->width * layer->height * sizeof(kiwi_fixed_t);

    layer->neurons = (kiwi_fixed_t*) malloc(size);
    memcpy(layer->neurons, src->neurons, size);

    return layer;  
}

void albedo_set_neuron_layer_value(AlbedoNeuronLayer* layer, kiwi_fixed_t value) {
    unsigned int size = layer->width * layer->height;

    for(int i = 0; i < size; ++i)
        layer->neurons[i] = value;
}

void albedo_reset_neuron_layer_value(AlbedoNeuronLayer* layer) {
    unsigned int size = layer->width * layer->height * sizeof(kiwi_fixed_t);
    memset(layer->neurons, 0, size);
}

void albedo_free_neuron_layer(AlbedoNeuronLayer* state) {
    free(state->neurons);
    free(state);   
}