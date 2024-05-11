#include "albedo_model.h"

AlbedoModel* albedo_new_model(unsigned int width, unsigned int height) {
    AlbedoModel* model = (AlbedoModel*) malloc(sizeof(AlbedoModel));

    model->weights = albedo_new_weights_layer(width, height);

    model->state[0] = albedo_new_neuron_layer(width, height);
    model->state[1] = albedo_new_neuron_layer(width, height);

    model->iteration = 0;
    model->newIndex = 0;

    model->width = width;
    model->height = height;

    return model;
}

void albedo_free_model(AlbedoModel* model) {
    albedo_free_weights_layer(model->weights);

    albedo_free_neuron_layer(model->state[0]);
    albedo_free_neuron_layer(model->state[1]);

    free(model);
}

void calculate_new_state(AlbedoNeuronLayer* newState, AlbedoNeuronLayer* oldState, AlbedoWeightsLayer* weights) {
    unsigned int width = newState->width;
    unsigned int height = newState->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            int maskWidth = ALBEDO_NEURON_WEIGHT_MASK_WIDTH / 2;
            int maskHeight = ALBEDO_NEURON_WEIGHT_MASK_HEIGHT / 2;

            float value = 0.0f; 

            for(int w = -maskWidth; w <= maskWidth; ++w) {
                for(int h = -maskHeight; h <= maskHeight; ++h) {
                    if(((x + w) < 0) || ((x + w) >= width))
                        continue;
                    
                    if(((y + h) < 0) || ((y + h) >= height))
                        continue;

                    value += oldState->neurons[(x + w) + (y + h)*width] * weights->neurons[x + y*width].mask[w + maskWidth][h + maskHeight];
                }
            }

            newState->neurons[x + y*width] = albedo_clampf(value, 0.0, 1.0);
        }
    }
}

void albedo_simulate_model_step(AlbedoModel* model) {
    int old = model->iteration % 2;
    int new = (model->iteration + 1) % 2;

    model->newIndex = new;

    calculate_new_state(model->state[new], model->state[old], model->weights);
    ++model->iteration;
}

void albedo_simulate_model_steps(AlbedoModel* model, unsigned int steps) {
    for(int i = 0; i < steps; ++i) 
        albedo_simulate_model_step(model);
}

// =========================================================
void set_inputs_model(AlbedoModel* model, float input[]) {
    for(int i = 0; i < model->width; ++i) {
        model->state[0]->neurons[i] = input[i];
        model->state[1]->neurons[i] = input[i];
    }
}

float calculate_error_delta(AlbedoModel* model, float expectedOutput[]) {
    float error = 0.0;

    for(int i = 0; i < model->width; ++i)
        error += fabs(model->state[model->newIndex]->neurons[i + (model->height-1)*model->width] - expectedOutput[i]);

    return error;
}

float calculate_error(AlbedoModel* model, float expectedOutput[]) {
    return calculate_error_delta(model, expectedOutput) / (float) model->width;
}
// =========================================================
