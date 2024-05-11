#include "albedo_training.h"

void albedo_genetic_algorithm_training_internal(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps,
    void (*snapshotCallback)(AlbedoModel*, AlbedoNeuronValue**, unsigned int)
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
            float localError = 0.0;

            // Simulate each test case and collect error
            for(int t = 0; t < testCases; ++t) {
                albedo_reset_model_neurons_value(models[m]);

                for(int s = 0; s < desiredSteps; ++s) {
                    albedo_set_model_neurons_values(models[m], inputs[t], inputCount);
                    albedo_simulate_model_step(models[m]);
                }

                localError += albedo_get_dif_model_neurons_values(models[m], outputs[t], outputCount);
            }

            localError /= (float) (testCases * outputCount);
            
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

        printf("Simulated epoch %d, error %f, best index %d\n", e, error, bestIndex);

        // Callback
        if(snapshotCallback != NULL)
            (*snapshotCallback)(model, inputs, inputCount);

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
