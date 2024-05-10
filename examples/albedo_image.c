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

void run_show_resulting_image(AlbedoModel* model, float** inputs) {
    unsigned int size = 8 * 8;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int x = 0; x < 8; ++x) {
        for(int y = 0; y < 8; ++y) {
            memset(model->state[0]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));
            memset(model->state[1]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));

            for(int s = 0; s < STEPS; ++s) {
                set_inputs_model(model, inputs[x + y*8]);
                albedo_simulate_model_step(model);
                set_inputs_model(model, inputs[x + y*8]);
            }

            float value = model->state[model->newIndex]->neurons[3 + (model->height-1)*model->width];

            unsigned char r = value * 255.0f;   

            grid[x + y*8] = (255 << 24) | (r << 16) | (r << 8) | (r);
        }
    }

    export_gradient_layer_to_png(model->weights, "weights.png");

    stbi_write_png("result.png", 8, 8, 4, grid, 8*4);
}

void run_tests_on_model(AlbedoModel* bestModel, float** inputs, float** outputs, unsigned int testCases) {
    // Show case best model
    float totalModelError = 0.0;

    for(int t = 0; t < testCases; ++t) {
        printf("Test case %d\n", t);
        printf("Input date: ");
        for(int i = 0; i < GRID_WIDTH; ++i) {
            printf(" %f", inputs[t][i]);
        }
        printf("\n");

        printf("Expected output: ");
        for(int i = 0; i < GRID_WIDTH; ++i) {
            printf(" %f", outputs[t][i]);
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
            printf(" %f", bestModel->state[bestModel->newIndex]->neurons[i + 7*GRID_WIDTH]);
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

    int width, height, channels;
    void* bytes = stbi_load("number.png", &width, &height, &channels, 0);

    printf("%d %d %d\n", width, height, channels);

    unsigned int testCases = width*height;

    float** inputs = (float**) malloc(sizeof(float*) * testCases);
    float** outputs = (float**) malloc(sizeof(float*) * testCases);
    
    for(int t = 0; t < testCases; ++t) {
        inputs[t] = (float*) malloc(sizeof(float) * GRID_WIDTH);
        outputs[t] = (float*) malloc(sizeof(float) * GRID_WIDTH);

        memset(inputs[t], 0, sizeof(float) * GRID_WIDTH);
        memset(outputs[t], 1.0f, sizeof(float) * GRID_WIDTH);
    }

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            unsigned char pixel = ((unsigned char*) bytes)[1 + (x + y*width) * channels];
            float value = (float) pixel / (float) 256.0f;

            inputs[x + y*width][0] = (float) x / (float) width; 
            inputs[x + y*width][7] = (float) y / (float) height; 
            
            outputs[x + y*width][7] = value; 
        }
    }

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

            printf("Error %f\n", error);
            
            if(error < bestError) {
                bestError = error;
                bestIndex = m;
            }
        }

        bestModel = models[bestIndex];
        run_show_resulting_image(bestModel, inputs);

        printf("Simulated epoch %d, error %f, best index %d\n", i, bestError, bestIndex);

        if(bestError < 0.0001) {
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

        // albedo_free_model(bestModel);
    }

    run_tests_on_model(bestModel, inputs, outputs, testCases);
    albedo_free_model(bestModel);

    return 0;
}