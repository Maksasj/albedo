#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "albedo_visualization.h"
#include "albedo/albedo.h"

#define GRID_WIDTH 8
#define GRID_HEIGHT 8

#define SAMPLE_MODELS 100
#define STEPS 50

#define ALBEDO_EPSILON 0.05

void run_tests_on_model(AlbedoModel* bestModel, float** inputs, float** outputs, unsigned int testCases) {
    // Show case best model
    float totalModelError = 0.0;

    for(int t = 0; t < testCases; ++t) {
        printf("Test case %d\n", t);
        printf("Input date: ");
        for(int i = 0; i < GRID_WIDTH; ++i) {
            printf(" %1.f", inputs[t][i]);
        }
        printf("\n");

        printf("Expected output: ");
        for(int i = 0; i < GRID_WIDTH; ++i) {
            printf(" %1.f", outputs[t][i]);
        }
        printf("\n");

        memset(bestModel->state[0]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));
        memset(bestModel->state[1]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));

        for(int i = 0; i < STEPS; ++i) {
            set_inputs_model(bestModel, inputs[t]);
            albedo_simulate_model_step(bestModel);
            set_inputs_model(bestModel, inputs[t]);
        }

        printf("Gotten outputs: ");
        for(int i = 0; i < GRID_WIDTH; ++i) {   
            printf(" %1.f", bestModel->state[bestModel->newIndex]->neurons[i + 7*GRID_WIDTH]);
        }
        printf("\n");

        float error = calculate_error(bestModel, outputs[t]);
        totalModelError += error;

        printf("AlbedoModel error %f\n", error);

        export_gradient_layer_to_png(bestModel->weights, "weights.png");
        export_state_layer_to_png(bestModel->state[bestModel->newIndex], "state.png");
    }

    totalModelError /= (float) testCases;

    printf("Total modal error %f\n", totalModelError);
}

int main() {
    srand(time(0));

    unsigned int testCases = 4;

    float** inputs = (float**) malloc(sizeof(float*) * testCases);
    float** outputs = (float**) malloc(sizeof(float*) * testCases);
    
    for(int t = 0; t < testCases; ++t) {
        inputs[t] = (float*) malloc(sizeof(float) * GRID_WIDTH);
        outputs[t] = (float*) malloc(sizeof(float) * GRID_WIDTH);

        memset(inputs[t], 0, sizeof(float) * GRID_WIDTH);
        memset(outputs[t], 0, sizeof(float) * GRID_WIDTH);
    }

    inputs[0][0] = 0.0f;
    inputs[0][1] = 0.0f;
    inputs[0][2] = 1.0f; // Bias
        outputs[0][0] = 1.0f;

    inputs[1][0] = 0.0f;
    inputs[1][1] = 1.0f;
    inputs[1][2] = 1.0f; // Bias
        outputs[1][0] = 0.0f;

    inputs[2][0] = 0.0f;
    inputs[2][1] = 1.0f;
    inputs[2][2] = 1.0f; // Bias
        outputs[2][0] = 0.0f;
    
    inputs[3][0] = 1.0f;
    inputs[3][1] = 1.0f;
    inputs[3][2] = 1.0f; // Bias
        outputs[3][0] = 1.0f;

    printf("Have prepared test date \n");

    AlbedoModel* bestModel = NULL;

    AlbedoModel* models[SAMPLE_MODELS] = { NULL };
    for(int i = 0; i < SAMPLE_MODELS; ++i)
        models[i] = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);

    printf("Started training\n");
    for(int i = 0;;++i) {
        int bestIndex = -1;
        float bestError = FLT_MAX;

        // Run simulations across all models and get best
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            float error = 0.0;

            // Simulate each test case and collect error
            for(int t = 0; t < testCases; ++t) {
                memset(models[m]->state[0]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));
                memset(models[m]->state[1]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));

                for(int s = 0; s < STEPS; ++s) {
                    set_inputs_model(models[m], inputs[t]);
                    albedo_simulate_model_step(models[m]);
                    set_inputs_model(models[m], inputs[t]);
                }

                error += calculate_error(models[m], outputs[t]);
            }

            error /= (float) (testCases);
            
            if(error < bestError) {
                bestError = error;
                bestIndex = m;
            }
        }

        bestModel = models[bestIndex];

        printf("Simulated epoch %d, error %f, best index %d\n", i, bestError, bestIndex);

        if(bestError < 0.004) {
            printf("Best error is %f, stopping training\n", bestError);
            break;
        }

        // Delete all models
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            if(m == bestIndex) {
                models[m] = NULL;
                continue;
            }

            albedo_free_model(models[m]);
            models[m] = NULL;
        }

        // Populate best model
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            models[m] = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);
            memcpy(models[m]->weights->neurons, bestModel->weights->neurons, GRID_WIDTH*GRID_HEIGHT*sizeof(AlbedoNeuronWeight));

            AlbedoModel* model = models[m];
            albedo_tune_weights_layer(model->weights, bestError);
        }

        albedo_free_model(bestModel);
    }

    run_tests_on_model(bestModel, inputs, outputs, testCases);
    albedo_free_model(bestModel);

    return 0;
}