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
AlbedoModel* albedo_copy_model(AlbedoModel* src) {
    AlbedoModel* model = (AlbedoModel*) malloc(sizeof(AlbedoModel));

    model->weights = albedo_copy_weights_layer(src->weights);
    model->state[0] = albedo_copy_neuron_layer(src->state[0]);
    model->state[1] = albedo_copy_neuron_layer(src->state[1]);

    model->iteration = src->iteration;
    model->newIndex = src->newIndex;

    model->width = src->width;
    model->height = src->height;

    return model;
}

void albedo_free_model(AlbedoModel* model) {
    albedo_free_weights_layer(model->weights);

    albedo_free_neuron_layer(model->state[0]);
    albedo_free_neuron_layer(model->state[1]);

    free(model);
}

void albedo_set_model_neurons_value(AlbedoModel* model, float value) {
    albedo_set_neuron_layer_value(model->state[0], value);
    albedo_set_neuron_layer_value(model->state[1], value);
}

void albedo_reset_model_neurons_value(AlbedoModel* model) {
    albedo_reset_neuron_layer_value(model->state[0]);
    albedo_reset_neuron_layer_value(model->state[1]);
}

void albedo_set_model_neurons_values(AlbedoModel* model, AlbedoNeuronValue* values, unsigned int count) {
    for(unsigned int i = 0; i < count; ++i) {
        AlbedoNeuronValue value = values[i];

        unsigned int index = value.x + value.y * model->width;

        model->state[0]->neurons[index] = value.value;
        model->state[1]->neurons[index] = value.value;
    }
}

// Todo improve this
float albedo_get_dif_model_neurons_values(AlbedoModel* model, AlbedoNeuronValue* values, unsigned int count) {
    float dif = 0.0f;

    for(unsigned int i = 0; i < count; ++i) {
        AlbedoNeuronValue value = values[i];

        unsigned int index = value.x + value.y * model->width;

        dif += fabs(model->state[model->newIndex]->neurons[index] - value.value);
    }

    return dif;
}

void calculate_new_state(AlbedoNeuronLayer* newState, AlbedoNeuronLayer* oldState, AlbedoWeightsLayer* weights) {
    unsigned int width = newState->width;
    unsigned int height = newState->height;

    unsigned int widthM = width - 1;
    unsigned int heightM = height - 1;

    // Corners
    {   // Top left
        float value = oldState->neurons[0]              * weights->weights[0].kernel[1][1];
        value += oldState->neurons[1]                   * weights->weights[0].kernel[2][1];
        value += oldState->neurons[width]                   * weights->weights[0].kernel[1][0];
        value += oldState->neurons[1 + width]                   * weights->weights[0].kernel[2][0];
        newState->neurons[0] = albedo_clampf(value, 0.0, 1.0);
    }

    {   // Top right
        float value = oldState->neurons[widthM]         * weights->weights[widthM].kernel[1][1];
        value += oldState->neurons[widthM - 1]              * weights->weights[widthM].kernel[0][1];
        value += oldState->neurons[widthM + width]              * weights->weights[widthM].kernel[1][0];
        value += oldState->neurons[widthM - 1 + width]              * weights->weights[widthM].kernel[0][0];
        newState->neurons[widthM] = albedo_clampf(value, 0.0, 1.0);
    }

    {   // Bottom left
        unsigned int index = heightM * width;
        float value = oldState->neurons[index]          * weights->weights[index].kernel[1][1];
        value += oldState->neurons[index - width]               * weights->weights[index].kernel[1][2];
        value += oldState->neurons[index + 1]               * weights->weights[index].kernel[2][1];
        value += oldState->neurons[index - width + 1]               * weights->weights[index].kernel[2][2];
        newState->neurons[index] = albedo_clampf(value, 0.0, 1.0);
    }

    {   // Bottom right
        unsigned int index = widthM + heightM * width;
        float value = oldState->neurons[index]          * weights->weights[index].kernel[1][1];
        value += oldState->neurons[index - 1]               * weights->weights[index].kernel[1][0];
        value += oldState->neurons[index - width]               * weights->weights[index].kernel[2][1];
        value += oldState->neurons[index - 1 - width]               * weights->weights[index].kernel[2][0];
        newState->neurons[index] = albedo_clampf(value, 0.0, 1.0);
    }

    // Borders
    // along x axis
    for(int x = 1; x < widthM; ++x) {
        {
            float value = oldState->neurons[x]  * weights->weights[x].kernel[1][1];
            value += oldState->neurons[x - 1]       * weights->weights[x].kernel[0][1];
            value += oldState->neurons[x + 1]       * weights->weights[x].kernel[2][1];
            value += oldState->neurons[x + width - 1]       * weights->weights[x].kernel[0][0];
            value += oldState->neurons[x + width]       * weights->weights[x].kernel[1][0];
            value += oldState->neurons[x + width + 1]       * weights->weights[x].kernel[2][0];
            newState->neurons[x] = albedo_clampf(value, 0.0, 1.0);
        }

        {
            unsigned int index = x + heightM * width;
            float value = oldState->neurons[index]  * weights->weights[index].kernel[1][1];
            value += oldState->neurons[index - 1]       * weights->weights[index].kernel[0][1];
            value += oldState->neurons[index + 1]       * weights->weights[index].kernel[2][1];
            value += oldState->neurons[index - width -1]       * weights->weights[index].kernel[0][2];
            value += oldState->neurons[index - width]       * weights->weights[index].kernel[1][2];
            value += oldState->neurons[index - width + 1]       * weights->weights[index].kernel[2][2];
            newState->neurons[index] = albedo_clampf(value, 0.0, 1.0);
        }
    }

    // along y axis
    for(int y = 1; y < widthM; ++y) {
        {
            unsigned int index = y * width;
            float value = oldState->neurons[index]  * weights->weights[index].kernel[1][1];
            value += oldState->neurons[index - width]       * weights->weights[index].kernel[1][2];
            value += oldState->neurons[index + width]       * weights->weights[index].kernel[1][0];
            value += oldState->neurons[index - width + 1]       * weights->weights[index].kernel[2][2];
            value += oldState->neurons[index + 1]       * weights->weights[index].kernel[2][1];
            value += oldState->neurons[index + width + 1]       * weights->weights[index].kernel[2][0];
            newState->neurons[index] = albedo_clampf(value, 0.0, 1.0);
        }

        {
            unsigned int index = widthM + y * width;
            float value = oldState->neurons[index]  * weights->weights[index].kernel[1][1];
            value += oldState->neurons[index - width]       * weights->weights[index].kernel[1][2];
            value += oldState->neurons[index + width]       * weights->weights[index].kernel[1][0];
            value += oldState->neurons[index - width - 1]       * weights->weights[index].kernel[0][2];
            value += oldState->neurons[index - 1]       * weights->weights[index].kernel[0][1];
            value += oldState->neurons[index + width - 1]       * weights->weights[index].kernel[0][0];
            newState->neurons[index] = albedo_clampf(value, 0.0, 1.0);
        }
    }

    // Center square
    for(int x = 1; x < widthM; ++x) {
        for(int y = 1; y < heightM; ++y) {
            unsigned int index = x + y*width;

            float value = oldState->neurons[(x - 1)  + (y + 1)*width]   * weights->weights[index].kernel[0][2];
            value += oldState->neurons[(x)      + (y + 1)*width]        * weights->weights[index].kernel[1][2];
            value += oldState->neurons[(x + 1)  + (y + 1)*width]        * weights->weights[index].kernel[2][2];

            value += oldState->neurons[(x - 1)  + (y)*width]            * weights->weights[index].kernel[0][1];
            value += oldState->neurons[(x)      + (y)*width]            * weights->weights[index].kernel[1][1];
            value += oldState->neurons[(x + 1)  + (y)*width]            * weights->weights[index].kernel[2][1];

            value += oldState->neurons[(x - 1)  + (y - 1)*width]        * weights->weights[index].kernel[0][0];
            value += oldState->neurons[(x)      + (y - 1)*width]        * weights->weights[index].kernel[1][0];
            value += oldState->neurons[(x + 1)  + (y - 1)*width]        * weights->weights[index].kernel[2][0];

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
