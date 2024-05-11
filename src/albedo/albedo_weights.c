#include "albedo_weights.h"

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
                    layer->neurons[x + y*width].mask[w][h] = albedo_randf(-1.0f, 1.0f);
            
        }
    }

    return layer;
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

void albedo_free_weights_layer(AlbedoWeightsLayer* weights) {
    free(weights->neurons);
    free(weights);
}