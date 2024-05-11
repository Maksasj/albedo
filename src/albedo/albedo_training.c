#include "albedo_training.h"

float albedo_model_calculate_error(
    AlbedoModel* model,
    AlbedoNeuronValue* inputs,
    AlbedoNeuronValue* outputs, 
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int steps
) {
    float error = 0.0f;

    for(int s = 0; s < steps; ++s) {
        albedo_set_model_neurons_values(model, inputs, inputCount);
        albedo_simulate_model_step(model);
    }

    error += albedo_get_dif_model_neurons_values(model, outputs, outputCount);

    return (float) error / (float) outputCount;
}

float albedo_model_calculate_error_from_tests(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int steps
) {
    float error = 0.0f;

    for(int t = 0; t < testCases; ++t) {
        albedo_reset_model_neurons_value(model);
        error += albedo_model_calculate_error(model, inputs[t], outputs[t], inputCount, outputCount, steps);
    }

    error /= (float) (testCases);

    return error;
}


void albedo_genetic_algorithm_training_internal(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps,
    AlbedoTrainingSnapshotCallback *snapshotCallback
) {
    unsigned int width = model->width;
    unsigned int height = model->height;

    AlbedoModel* models[SAMPLE_MODELS] = { NULL };

    float error = 1.0f;
    for(int e = 0; e < ALBEDO_MAX_EPOCHS; ++e) {
        int bestIndex = -1;

        float errorStep = error*error;

        // Populate models
        for(int i = 0; i < SAMPLE_MODELS; ++i) {
            models[i] = albedo_copy_model(model);

            AlbedoWeightsLayer* dif = albedo_new_weights_layer_clamped(width, height, -errorStep, errorStep);
            albedo_weights_layer_add(models[i]->weights, dif);
            albedo_free_weights_layer(dif);
        }

        // Simulate models
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            float localError = albedo_model_calculate_error_from_tests(
                models[m], 
                inputs, 
                outputs, 
                testCases, 
                inputCount, 
                outputCount, 
                desiredSteps
            );
            
            if(localError < error) {
                error = localError;
                bestIndex = m;
            }
        }

        // If there is model better, tune model weights
        if(bestIndex >= 0) {
            albedo_free_weights_layer(model->weights);
            model->weights = albedo_copy_weights_layer(models[bestIndex]->weights);
        }
        
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            albedo_free_model(models[m]);
            models[m] = NULL;
        }

        // Callback
        if(snapshotCallback != NULL) {
            AlbedoTrainingSnapshot snapshot;

            snapshot.model = model;
            snapshot.inputs = inputs;
            snapshot.outputs = outputs; 
            snapshot.testCases = testCases;
            snapshot.inputCount = inputCount;
            snapshot.outputCount = outputCount;
            snapshot.desiredError = desiredError;
            snapshot.desiredSteps = desiredSteps;
            snapshot.epoch = e;
            snapshot.currentError = error;

            (*snapshotCallback)(&snapshot);
        }

        if(error < desiredError)
            break;
    }
}

void albedo_genetic_algorithm_training(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps
) {
    albedo_genetic_algorithm_training_internal(
        model, 
        inputs, 
        outputs, 
        testCases, 
        inputCount, 
        outputCount, 
        desiredError, 
        desiredSteps,
        NULL);
}

void albedo_finite_difference_training_internal(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps,
    float epsilon,
    float learningRate,
    AlbedoTrainingSnapshotCallback *snapshotCallback
) {
    unsigned int width = model->width;
    unsigned int height = model->height;

    for(int e = 0; e < ALBEDO_MAX_EPOCHS; ++e) {
        float error = albedo_model_calculate_error_from_tests(model, inputs, outputs, testCases, inputCount, outputCount, desiredSteps);

        // Callback
        if(snapshotCallback != NULL) {
            AlbedoTrainingSnapshot snapshot;

            snapshot.model = model;
            snapshot.inputs = inputs;
            snapshot.outputs = outputs; 
            snapshot.testCases = testCases;
            snapshot.inputCount = inputCount;
            snapshot.outputCount = outputCount;
            snapshot.desiredError = desiredError;
            snapshot.desiredSteps = desiredSteps;
            snapshot.epoch = e;
            snapshot.currentError = error;

            (*snapshotCallback)(&snapshot);
        }

        if(error <= desiredError)
            break;

        AlbedoWeightsLayer* gradient = albedo_copy_weights_layer(model->weights);

        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                for(int w = 0; w < ALBEDO_NEURON_WEIGHT_MASK_WIDTH; ++w) {
                    for(int h = 0; h < ALBEDO_NEURON_WEIGHT_MASK_HEIGHT; ++h) {
                        unsigned int index = x + y*width;

                        model->weights->neurons[index].mask[w][h] += epsilon;
                        float dcost = albedo_model_calculate_error_from_tests(model, inputs, outputs, testCases, inputCount, outputCount, desiredSteps);
                        model->weights->neurons[index].mask[w][h] -= epsilon;

                        gradient->neurons[index].mask[w][h] += learningRate*dcost;
                    }
                }
            }
        }

        albedo_weights_layer_subtract(model->weights, gradient);
        albedo_free_weights_layer(gradient);
    }
}

void albedo_finite_difference_training(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps,
    float epsilon,
    float learningRate
) {
    albedo_finite_difference_training_internal(
        model,
        inputs,
        outputs,
        testCases,
        inputCount,
        outputCount,
        desiredError,
        desiredSteps,
        epsilon,
        learningRate,
        NULL
    );
}   