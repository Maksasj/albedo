#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "albedo_visualization.h"
#include "albedo/albedo.h"

#define GRID_WIDTH 8
#define GRID_HEIGHT 8
#define STEPS 50
#define INPUT_COUNT 2
#define OUTPUT_COUNT 1

void run_show_resulting_image(AlbedoModel* model, AlbedoNeuronValue** inputs, unsigned int inputCount) {
    int width = 8;
    int height = 8;

    unsigned int size = width * height;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            albedo_reset_model_neurons_value(model);

            for(int s = 0; s < STEPS; ++s) {
                albedo_set_model_neurons_values(model, inputs[x + y*width], inputCount);
                albedo_simulate_model_step(model);
            }

            float value = model->state[model->newIndex]->neurons[0 + 7*model->width];

            HSL hsl;
            hsl.H = (1.0 - value) * 240;
            hsl.L = 0.5f;
            hsl.S = 1.0f;

            RGB rgb = hsl_to_rgb(hsl);

            grid[y + x*height] = (255 << 24) | (rgb.B << 16) | (rgb.G << 8) | (rgb.R);
        }
    }

    export_gradient_layer_to_png(model->weights, "weights.png");

    static int frame = 0;

    char fileName[50] = { '\0' };  
    sprintf(fileName, "result/frame_%d.png", frame);
    stbi_write_png(fileName, width, height, 4, grid, width*4);
    stbi_write_png("result.png", width, height, 4, grid, width*4);

    free(grid);

    ++frame;
}

void intermediate_result(AlbedoTrainingSnapshot* snapshot) {
    printf("Simulated epoch %d, error %f\n", snapshot->epoch, snapshot->currentCost);

    run_show_resulting_image(snapshot->model, snapshot->inputs, snapshot->inputCount);
}

int main() {
    srand(time(0));
    
    int width, height, channels;
    void* bytes = stbi_load("examples/number.png", &width, &height, &channels, 0);

    printf("%d %d %d\n", width, height, channels);

    unsigned int TEST_CASES = width*height;

    AlbedoNeuronValue** inputs = malloc(sizeof(AlbedoNeuronValue*) * TEST_CASES);
    AlbedoNeuronValue** outputs = malloc(sizeof(AlbedoNeuronValue*) * TEST_CASES);

    for(int t = 0; t < TEST_CASES; ++t) {
        inputs[t] = malloc(sizeof(AlbedoNeuronValue) * INPUT_COUNT);
        outputs[t] = malloc(sizeof(AlbedoNeuronValue) * OUTPUT_COUNT);
    }
    
    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            unsigned char pixel = ((unsigned char*) bytes)[1 + (y + x*height) * channels];
            float value = (float) pixel / (float) 256.0f;

            inputs[x + y*width][0].value = (float) x / (float) width;
            inputs[x + y*width][0].x = 0;
            inputs[x + y*width][0].y = 0;

            inputs[x + y*width][1].value = (float) y / (float) height; 
            inputs[x + y*width][1].x = 0;
            inputs[x + y*width][1].y = 7;

            outputs[x + y*width][0].value = value;
            outputs[x + y*width][0].x = 0;
            outputs[x + y*width][0].y = 7;
        }
    }

    printf("Have prepared test date \n");

    AlbedoModel* model = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);

    printf("Started training\n");

    albedo_finite_difference_training_internal(
        model, 
        inputs, 
        outputs, 
        TEST_CASES, 
        INPUT_COUNT, 
        OUTPUT_COUNT, 
        0.01, 
        STEPS,
        1e-3,
        1e-3,
        &albedo_calculate_fixed_step_result_cost,
        &intermediate_result
    );

    /*
    albedo_genetic_algorithm_training_internal(
        model, 
        inputs, 
        outputs, 
        TEST_CASES, 
        INPUT_COUNT, 
        OUTPUT_COUNT, 
        0.01, 
        STEPS,
        &intermediate_result);
    */

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