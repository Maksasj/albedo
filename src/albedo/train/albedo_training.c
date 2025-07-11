#include "albedo_training.h"

void albedo_genetic_algorithm_training_internal(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    kiwi_fixed_t epsilon,
    kiwi_fixed_t desiredCost,
    unsigned int desiredSteps,
    AlbedoCostFunction* costFunction,
    AlbedoTrainingSnapshotCallback *snapshotCallback
) {
    unsigned int width = model->width;
    unsigned int height = model->height;

    AlbedoModel* models[SAMPLE_MODELS] = { NULL };

    int generation_count = 1;
    kiwi_fixed_t cost = (*costFunction)(model, inputs, outputs, testCases, inputCount, outputCount, desiredSteps);
    for(int e = 0; e < ALBEDO_MAX_EPOCHS; ++e) {
        int bestIndex = -1;

        // Populate models
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            models[m] = albedo_copy_model(model);

            const float current_mutation_strength = 1.0 / sqrtf((float) generation_count);
            AlbedoWeightsLayer* dif = albedo_new_weights_layer_clamped(width, height, kiwi_float_to_fixed(-current_mutation_strength), kiwi_float_to_fixed(current_mutation_strength)); // Todo
            albedo_weights_layer_add(models[m]->weights, dif);
            albedo_free_weights_layer(dif);
        }

        // Simulate models
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            kiwi_fixed_t localCost = (*costFunction)(models[m], inputs, outputs, testCases, inputCount, outputCount, desiredSteps);
            // printf("%f\n", kiwi_fixed_to_float(localCost));

            if(localCost < cost) {
                cost = localCost;
                bestIndex = m;
            }
        }
        
        
        // If there is model better, tune model weights
        if(bestIndex >= 0) {
            albedo_free_weights_layer(model->weights);
            model->weights = albedo_copy_weights_layer(models[bestIndex]->weights);

            generation_count += 1;
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
            snapshot.desiredCost = desiredCost;
            snapshot.desiredSteps = desiredSteps;
            snapshot.epoch = e;
            snapshot.currentCost = cost;

            (*snapshotCallback)(&snapshot);
        }

        if(cost < desiredCost)
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
    kiwi_fixed_t epsilon,
    kiwi_fixed_t desiredCost,
    unsigned int desiredSteps,
    AlbedoCostFunction* costFunction
) {
    albedo_genetic_algorithm_training_internal(
        model, 
        inputs, 
        outputs, 
        testCases, 
        inputCount, 
        outputCount, 
        epsilon,
        desiredCost, 
        desiredSteps,
        costFunction,
        NULL);
}

void albedo_finite_difference_training_internal(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    kiwi_fixed_t desiredCost,
    unsigned int desiredSteps,
    kiwi_fixed_t epsilon,
    kiwi_fixed_t learningRate,
    AlbedoCostFunction* costFunction,
    AlbedoTrainingSnapshotCallback *snapshotCallback
) {
    unsigned int width = model->width;
    unsigned int height = model->height;

    AlbedoWeightsLayer* gradient = albedo_new_weights_layer(width, height);

    for(int e = 0; e < ALBEDO_MAX_EPOCHS; ++e) {
        kiwi_fixed_t cost = (*costFunction)(model, inputs, outputs, testCases, inputCount, outputCount, desiredSteps);

        // Callback
        if(snapshotCallback != NULL) {
            AlbedoTrainingSnapshot snapshot;

            snapshot.model = model;
            snapshot.inputs = inputs;
            snapshot.outputs = outputs; 
            snapshot.testCases = testCases;
            snapshot.inputCount = inputCount;
            snapshot.outputCount = outputCount;
            snapshot.desiredCost = desiredCost;
            snapshot.desiredSteps = desiredSteps;
            snapshot.epoch = e;
            snapshot.currentCost = cost;

            (*snapshotCallback)(&snapshot);
        }

        if(cost <= desiredCost)
            break;

        for(int x = 0; x < width; ++x) {
            for(int y = 0; y < height; ++y) {
                for(int w = 0; w < 3; ++w) {
                    for(int h = 0; h < 3; ++h) {
                        unsigned int index = x + y*width;

                        kiwi_fixed_t saved = model->weights->weights[index].kernel[w][h];
                        model->weights->weights[index].kernel[w][h] += epsilon;

                        kiwi_fixed_t eCost = (*costFunction)(model, inputs, outputs, testCases, inputCount, outputCount, desiredSteps);
                        kiwi_fixed_t dcost = (eCost - cost) / epsilon;

                        model->weights->weights[index].kernel[w][h] = saved;

                        gradient->weights[index].kernel[w][h] = learningRate*dcost;
                    }
                }
            }
        }

        albedo_weights_layer_subtract(model->weights, gradient);
    }

    albedo_free_weights_layer(gradient);
}

void albedo_finite_difference_training(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    kiwi_fixed_t desiredCost,
    unsigned int desiredSteps,
    kiwi_fixed_t epsilon,
    kiwi_fixed_t learningRate,
    AlbedoCostFunction* costFunction
) {
    albedo_finite_difference_training_internal(
        model,
        inputs,
        outputs,
        testCases,
        inputCount,
        outputCount,
        desiredCost,
        desiredSteps,
        epsilon,
        learningRate,
        costFunction,
        NULL
    );
}   