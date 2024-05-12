#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "albedo_visualization.h"
#include "albedo/albedo.h"

#define GRID_WIDTH 8
#define GRID_HEIGHT 8

#define TEST_CASES 2
#define INPUT_COUNT 2
#define OUTPUT_COUNT 1
#define STEPS 50

int main() {
    srand(time(0));

    // Last is a bias
    AlbedoNeuronValue rawInputs[TEST_CASES][INPUT_COUNT] = {
        {{0, 0, 0.0f}, {0, 7, 1.0f}},
        {{0, 0, 1.0f}, {0, 7, 1.0f}},
    };

    AlbedoNeuronValue rawOutputs[TEST_CASES][OUTPUT_COUNT] = {
        {{7, 7, 1.0f}},
        {{7, 7, 0.0f}},
    };

    AlbedoNeuronValue** inputs = malloc(sizeof(AlbedoNeuronValue*) * TEST_CASES);
    AlbedoNeuronValue** outputs = malloc(sizeof(AlbedoNeuronValue*) * TEST_CASES);

    for(int t = 0; t < TEST_CASES; ++t) {
        inputs[t] = malloc(sizeof(AlbedoNeuronValue) * INPUT_COUNT);
        outputs[t] = malloc(sizeof(AlbedoNeuronValue) * OUTPUT_COUNT);

        memcpy(inputs[t], rawInputs[t], sizeof(AlbedoNeuronValue) * INPUT_COUNT);
        memcpy(outputs[t], rawOutputs[t], sizeof(AlbedoNeuronValue) * OUTPUT_COUNT);
    }

    AlbedoModel* model = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);

    printf("Started training\n");
    albedo_genetic_algorithm_training(model, inputs, outputs, TEST_CASES, INPUT_COUNT, OUTPUT_COUNT, 0.05, 0.004, STEPS, &albedo_calculate_fixed_step_result_cost);
    printf("Training done\n");
    
    albedo_sumup_testing(model, inputs, outputs, TEST_CASES, INPUT_COUNT, OUTPUT_COUNT, STEPS);

    for(int t = 0; t < TEST_CASES; ++t) {
        free(inputs[t]);
        free(outputs[t]);
    }

    free(inputs);
    free(outputs);

    albedo_free_model(model);

    return 0;
}