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
kiwi_fixed_t albedo_get_dif_model_neurons_values(AlbedoModel* model, AlbedoNeuronValue* values, unsigned int count) {
    kiwi_fixed_t dif = 0.0f;

    for(unsigned int i = 0; i < count; ++i) {
        AlbedoNeuronValue value = values[i];

        unsigned int index = value.x + value.y * model->width;

        dif += kiwi_abs(model->state[model->newIndex]->neurons[index] - value.value);
    }

    return dif;
}

void calculate_new_state(AlbedoNeuronLayer* newState, AlbedoNeuronLayer* oldState, AlbedoWeightsLayer* weights) {
    const unsigned int width = newState->width;
    const unsigned int height = newState->height;

    const unsigned int widthM = width - 1;
    const unsigned int heightM = height - 1;

    const kiwi_fixed_t fzero = kiwi_float_to_fixed(0.0f);
    const kiwi_fixed_t fone = kiwi_float_to_fixed(1.0f);

    // Corners
    {   // Top left
        const AlbedoNeuronKernel kl = weights->weights[0];

        kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[0], kl.kernel[1][1]);
        value += kiwi_fixed_mul(oldState->neurons[1], kl.kernel[2][1]);
        value += kiwi_fixed_mul(oldState->neurons[width], kl.kernel[1][0]);
        value += kiwi_fixed_mul(oldState->neurons[1 + width], kl.kernel[2][0]);
        newState->neurons[0] = albedo_clamp_fixed(value, fzero, fone);
    }

    {   // Top right
        const AlbedoNeuronKernel kl = weights->weights[widthM];

        kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[widthM], kl.kernel[1][1]);
        value += kiwi_fixed_mul(oldState->neurons[widthM - 1], kl.kernel[0][1]);
        value += kiwi_fixed_mul(oldState->neurons[widthM + width], kl.kernel[1][0]);
        value += kiwi_fixed_mul(oldState->neurons[widthM - 1 + width], kl.kernel[0][0]);
        newState->neurons[widthM] = albedo_clamp_fixed(value, fzero, fone);
    }

    {   // Bottom left
        const unsigned int index = heightM * width;
        const AlbedoNeuronKernel kl = weights->weights[index];

        kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[index], kl.kernel[1][1]);
        value += kiwi_fixed_mul(oldState->neurons[index - width], kl.kernel[1][2]);
        value += kiwi_fixed_mul(oldState->neurons[index + 1], kl.kernel[2][1]);
        value += kiwi_fixed_mul(oldState->neurons[index - width + 1], kl.kernel[2][2]);
        newState->neurons[index] = albedo_clamp_fixed(value, fzero, fone);
    }

    {   // Bottom right
        const unsigned int index = widthM + heightM * width;
        const AlbedoNeuronKernel kl = weights->weights[index];

        kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[index], kl.kernel[1][1]);
        value += kiwi_fixed_mul(oldState->neurons[index - 1], kl.kernel[1][0]);
        value += kiwi_fixed_mul(oldState->neurons[index - width], kl.kernel[2][1]);
        value += kiwi_fixed_mul(oldState->neurons[index - 1 - width], kl.kernel[2][0]);
        newState->neurons[index] = albedo_clamp_fixed(value, fzero, fone);
    }

    // Borders
    // along x axis
    for(int x = 1; x < widthM; ++x) {
        {
            const AlbedoNeuronKernel kl = weights->weights[x];

            kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[x], kl.kernel[1][1]);
            value += kiwi_fixed_mul(oldState->neurons[x - 1], kl.kernel[0][1]);
            value += kiwi_fixed_mul(oldState->neurons[x + 1], kl.kernel[2][1]);
            value += kiwi_fixed_mul(oldState->neurons[x + width - 1], kl.kernel[0][0]);
            value += kiwi_fixed_mul(oldState->neurons[x + width], kl.kernel[1][0]);
            value += kiwi_fixed_mul(oldState->neurons[x + width + 1], kl.kernel[2][0]);
            newState->neurons[x] = albedo_clamp_fixed(value, fzero, fone);
        }

        {
            const unsigned int index = x + heightM * width;
            const AlbedoNeuronKernel kl = weights->weights[index];

            kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[index], kl.kernel[1][1]);
            value += kiwi_fixed_mul(oldState->neurons[index - 1], kl.kernel[0][1]);
            value += kiwi_fixed_mul(oldState->neurons[index + 1], kl.kernel[2][1]);
            value += kiwi_fixed_mul(oldState->neurons[index - width -1], kl.kernel[0][2]);
            value += kiwi_fixed_mul(oldState->neurons[index - width], kl.kernel[1][2]);
            value += kiwi_fixed_mul(oldState->neurons[index - width + 1], kl.kernel[2][2]);
            newState->neurons[index] = albedo_clamp_fixed(value, fzero, fone);
        }
    }

    // along y axis
    for(int y = 1; y < widthM; ++y) {
        {
            const unsigned int index = y * width;
            const AlbedoNeuronKernel kl = weights->weights[index];

            kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[index], kl.kernel[1][1]);
            value += kiwi_fixed_mul(oldState->neurons[index - width], kl.kernel[1][2]);
            value += kiwi_fixed_mul(oldState->neurons[index + width], kl.kernel[1][0]);
            value += kiwi_fixed_mul(oldState->neurons[index - width + 1], kl.kernel[2][2]);
            value += kiwi_fixed_mul(oldState->neurons[index + 1], kl.kernel[2][1]);
            value += kiwi_fixed_mul(oldState->neurons[index + width + 1], kl.kernel[2][0]);
            newState->neurons[index] = albedo_clamp_fixed(value, fzero, fone);
        }

        {
            const unsigned int index = widthM + y * width;
            const AlbedoNeuronKernel kl = weights->weights[index];

            kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[index], kl.kernel[1][1]);
            value += kiwi_fixed_mul(oldState->neurons[index - width], kl.kernel[1][2]);
            value += kiwi_fixed_mul(oldState->neurons[index + width], kl.kernel[1][0]);
            value += kiwi_fixed_mul(oldState->neurons[index - width - 1], kl.kernel[0][2]);
            value += kiwi_fixed_mul(oldState->neurons[index - 1], kl.kernel[0][1]);
            value += kiwi_fixed_mul(oldState->neurons[index + width - 1], kl.kernel[0][0]);
            newState->neurons[index] = albedo_clamp_fixed(value, fzero, fone);
        }
    }

    // Center square
    for(int x = 1; x < widthM; ++x) {
        for(int y = 1; y < heightM; ++y) {
            const unsigned int index = x + y*width;
            const AlbedoNeuronKernel kl = weights->weights[index];

            kiwi_fixed_t value = kiwi_fixed_mul(oldState->neurons[(x - 1)  + (y + 1)*width], kl.kernel[0][2]);
            value += kiwi_fixed_mul(oldState->neurons[(x)      + (y + 1)*width], kl.kernel[1][2]);
            value += kiwi_fixed_mul(oldState->neurons[(x + 1)  + (y + 1)*width], kl.kernel[2][2]);

            value += kiwi_fixed_mul(oldState->neurons[(x - 1)  + (y)*width], kl.kernel[0][1]);
            value += kiwi_fixed_mul(oldState->neurons[(x)      + (y)*width], kl.kernel[1][1]);
            value += kiwi_fixed_mul(oldState->neurons[(x + 1)  + (y)*width], kl.kernel[2][1]);

            value += kiwi_fixed_mul(oldState->neurons[(x - 1)  + (y - 1)*width], kl.kernel[0][0]);
            value += kiwi_fixed_mul(oldState->neurons[(x)      + (y - 1)*width], kl.kernel[1][0]);
            value += kiwi_fixed_mul(oldState->neurons[(x + 1)  + (y - 1)*width], kl.kernel[2][0]);

            newState->neurons[x + y*width] = albedo_clamp_fixed(value, fzero, fone);
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
