#ifndef BLUEBERRY_H
#define BLUEBERRY_H

#ifdef BLUEBERRY_IMPLEMENTATION
    #define PEACH_IMPLEMENTATION
#endif

#include "peach.h"

#if !defined(BLUEB_MALLOC) && !defined(BLUEB_FREE)
    #include <stdlib.h>

    #define BLUEB_MALLOC malloc
    #define BLUEB_FREE free
#endif

#ifndef BLUEB_PRINTF
    #include <stdio.h>
    
    #define BLUEB_PRINTF printf
#endif

typedef struct BlueBerryModel {
    peach_matrix_t** weights;
    peach_matrix_t** biases;
    peach_matrix_t** neurons;

    peach_size_t count;
} BlueBerryModel;

BlueBerryModel* blueb_new_model(int* arch, peach_size_t archSize);
void blueb_free_model(BlueBerryModel* model);

BlueBerryModel* blueb_copy_model(BlueBerryModel* src);

void blueb_fill_model(BlueBerryModel* model, peach_float_t value);
void blueb_rand_model(BlueBerryModel* model, peach_float_t min, peach_float_t max);

void blueb_feed(BlueBerryModel* model, peach_matrix_t* input);
void blueb_forward(BlueBerryModel* model);
void blueb_feedforward(BlueBerryModel* model, peach_matrix_t* input);

peach_float_t blueb_mse_cost(
    BlueBerryModel* model, peach_matrix_t** inputs, peach_matrix_t** outputs, peach_size_t count);

void blueb_finite_difference(
    BlueBerryModel* model, 
    BlueBerryModel* gradient, 
    peach_matrix_t** inputs, 
    peach_matrix_t** outputs, 
    peach_size_t count
);

void blueb_learn_gradient(
    BlueBerryModel* model, BlueBerryModel* gradient, peach_float_t learningRate);

void blueb_train_gradient_descent(
    BlueBerryModel* model,
    peach_matrix_t** inputs, 
    peach_matrix_t** outputs, 
    peach_size_t count,
    peach_size_t epochs,
    peach_float_t learningRate
);

void blueb_print_model(BlueBerryModel* model);

#ifdef BLUEBERRY_IMPLEMENTATION

BlueBerryModel* blueb_new_model(int* arch, peach_size_t archSize) {
    BlueBerryModel* model = (BlueBerryModel*) BLUEB_MALLOC(sizeof(BlueBerryModel));

    model->count = archSize;

    model->weights = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * (archSize - 1));
    model->biases = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * (archSize - 1));
    model->neurons = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * archSize);

    model->neurons[0] = paech_new_matrix(1, arch[0]);

    for(peach_size_t i = 1; i < archSize; ++i) {
        model->weights[i - 1] = paech_new_matrix(arch[i - 1], arch[i]);
        model->biases[i - 1] = paech_new_matrix(1, arch[i]);
        model->neurons[i] = paech_new_matrix(1, arch[i]);
    }

    return model;
}

BlueBerryModel* blueb_copy_model(BlueBerryModel* src) {
    const peach_size_t archSize = src->count;

    BlueBerryModel* model = (BlueBerryModel*) BLUEB_MALLOC(sizeof(BlueBerryModel));

    model->weights = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * (archSize - 1));
    model->biases =  (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * (archSize - 1));
    model->neurons = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * archSize);

    model->neurons[0] = paech_copy_matrix(src->neurons[0]);

    for(peach_size_t i = 1; i < archSize; ++i) {
        model->weights[i - 1] = paech_copy_matrix(src->weights[i - 1]);
        model->biases[i - 1] = paech_copy_matrix(src->biases[i - 1]);
        model->neurons[i] = paech_copy_matrix(src->neurons[i]);
    }

    return model;
}

BlueBerryModel* blueb_copy_model_arc(BlueBerryModel* src) {
    const peach_size_t archSize = src->count;

    BlueBerryModel* model = (BlueBerryModel*) BLUEB_MALLOC(sizeof(BlueBerryModel));

    model->weights = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * (archSize - 1));
    model->biases =  (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * (archSize - 1));
    model->neurons = (peach_matrix_t**) BLUEB_MALLOC(sizeof(peach_matrix_t*) * archSize);

    model->neurons[0] = paech_copy_matrix_empty(src->neurons[0]);

    for(peach_size_t i = 1; i < archSize; ++i) {
        model->weights[i - 1] = paech_copy_matrix_empty(src->weights[i - 1]);
        model->biases[i - 1] = paech_copy_matrix_empty(src->biases[i - 1]);
        model->neurons[i] = paech_copy_matrix_empty(src->neurons[i]);
    }

    return model;
}

void blueb_fill_model(BlueBerryModel* model, peach_float_t value) {
    const peach_size_t size = model->count;

    for(peach_size_t i = 1; i < size; ++i) {
        peach_matrix_fill(model->biases[i - 1], value);
        peach_matrix_fill(model->weights[i - 1], value);
    }
}

void blueb_rand_model(BlueBerryModel* model, peach_float_t min, peach_float_t max) {
    const peach_size_t size = model->count;

    for(peach_size_t i = 1; i < size; ++i) {
        peach_matrix_rand(model->biases[i - 1], min, max);
        peach_matrix_rand(model->weights[i - 1], min, max);
    }
}

void blueb_feed(BlueBerryModel* model, peach_matrix_t* input) {
    peach_matrix_copy_content_target(model->neurons[0], input);
}

void blueb_forward(BlueBerryModel* model) {
    const peach_size_t size = model->count - 1;
    
    for(peach_size_t i = 0; i < size; ++i) {
        peach_matrix_dot_target(model->neurons[i + 1], model->neurons[i], model->weights[i]);

        peach_matrix_sum_target(model->neurons[i + 1], model->biases[i]);
        peach_matrix_apply_sigmoid(model->neurons[i + 1]);
    }
}

void blueb_feedforward(BlueBerryModel* model, peach_matrix_t* input) {
    blueb_feed(model, input);
    blueb_forward(model);
}

peach_float_t blueb_mse_cost(BlueBerryModel* model, peach_matrix_t** inputs, peach_matrix_t** outputs, peach_size_t count) {
    peach_matrix_t* output = model->neurons[model->count - 1]; 

    assert(output->rows == outputs[0]->rows);
    assert(output->cols == outputs[0]->cols);

    peach_float_t cost = 0.0f;

    for(peach_size_t i = 0; i < count; ++i) {
        blueb_feedforward(model, inputs[i]);

        peach_size_t size = output->rows * output->cols;

        for(peach_size_t j = 0; j < size; ++j) {
            peach_float_t d = output->value[j] - outputs[i]->value[j];
            cost += d * d;
        }
    }

    return cost;
}

void blueb_finite_difference(
    BlueBerryModel* model, 
    BlueBerryModel* gradient, 
    peach_matrix_t** inputs, 
    peach_matrix_t** outputs, 
    peach_size_t count
) {
    blueb_fill_model(gradient, 0.0f);
    const peach_size_t layers = model->count - 1;

    peach_float_t cost = blueb_mse_cost(model, inputs, outputs, count);

    static peach_float_t epsilon = 0.05f;

    for(peach_size_t t = 0; t < count; ++t) {
        // Weights
        for(peach_size_t w = 0; w < layers; ++w) {
            peach_matrix_t* weights = model->weights[w];
            peach_matrix_t* gWeights = gradient->weights[w];

            const peach_size_t size = weights->rows * weights->cols;
            
            for(peach_size_t i = 0; i < size; ++i) {
                peach_float_t saved = weights->value[i];

                weights->value[i] += epsilon;

                peach_float_t dcost = blueb_mse_cost(model, inputs, outputs, count);
                gWeights->value[i] += (dcost - cost) / epsilon;

                weights->value[i] = saved;
            }
        }

        // Biases
        for(peach_size_t b = 0; b < layers; ++b) {
            peach_matrix_t* biases = model->biases[b];
            peach_matrix_t* gBiases = gradient->weights[b];

            const peach_size_t size = biases->rows * biases->cols;
            
            for(peach_size_t i = 0; i < size; ++i) {
                peach_float_t saved = biases->value[i];

                biases->value[i] += epsilon;

                peach_float_t dcost = blueb_mse_cost(model, inputs, outputs, count);
                gBiases->value[i] += (dcost - cost) / epsilon;

                biases->value[i] = saved;
            }
        }
    }
}

void blueb_learn_gradient(BlueBerryModel* model, BlueBerryModel* gradient, peach_float_t learningRate) {
    const peach_size_t layers = model->count - 1;

    for(peach_size_t w = 0; w < layers; ++w) {
        peach_matrix_t* weights = model->weights[w];
        peach_matrix_t* gWeights = gradient->weights[w];

        peach_matrix_scale(gWeights, learningRate);
        peach_matrix_sub_target(weights, gWeights);
    }

    for(peach_size_t b = 0; b < layers; ++b) {
        peach_matrix_t* biases = model->biases[b];
        peach_matrix_t* gBiases = gradient->weights[b];

        peach_matrix_scale(gBiases, learningRate);
        peach_matrix_sub_target(biases, gBiases);
    }
}

void blueb_free_model(BlueBerryModel* model) {
    const peach_size_t size = model->count;

    for(peach_size_t i = 0; i < size; ++i)
        peach_free_matrix(model->neurons[i]);

    for(peach_size_t i = 1; i < size; ++i) {
        peach_free_matrix(model->weights[i - 1]);
        peach_free_matrix(model->biases[i - 1]);
    }

    BLUEB_FREE(model->biases);
    BLUEB_FREE(model->weights);
    BLUEB_FREE(model->neurons);

    BLUEB_FREE(model);
}

void blueb_train_gradient_descent(
    BlueBerryModel* model,
    peach_matrix_t** inputs, 
    peach_matrix_t** outputs, 
    peach_size_t count,
    peach_size_t epochs,
    peach_float_t learningRate
) {
    BlueBerryModel* gradient = blueb_copy_model_arc(model);

    for(peach_size_t e = 0; e < epochs; ++e) {
        blueb_finite_difference(model, gradient, inputs, outputs, count);
        blueb_learn_gradient(model, gradient, learningRate);
    }

    blueb_free_model(gradient);
}

void blueb_print_model(BlueBerryModel* model) {
    const peach_size_t size = model->count;

    for(peach_size_t i = 0; i < size; ++i) {
        if(i > 0) {
            BLUEB_PRINTF("weights[%lld] = ", i - 1);
            peach_matrix_print(model->weights[i - 1]);

            BLUEB_PRINTF("biases[%lld] = ", i - 1);
            peach_matrix_print(model->biases[i - 1]);
        }

        BLUEB_PRINTF("neurons[%lld] = ", i);
        peach_matrix_print(model->neurons[i]);
    }
}

#endif
 
#endif