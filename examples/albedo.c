#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "albedo/albedo.h"

#include "stb_image_write.h"

#define GRID_WIDTH 8
#define GRID_HEIGHT 8

typedef struct RGB {
	unsigned char R;
	unsigned char G;
	unsigned char B;
} RGB;

typedef struct HSL {
	int H;
	float S;
	float L;
} HSL;

float hue_to_rgb(float v1, float v2, float vH) {
	if (vH < 0)
		vH += 1;

	if (vH > 1)
		vH -= 1;

	if ((6 * vH) < 1)
		return (v1 + (v2 - v1) * 6 * vH);

	if ((2 * vH) < 1)
		return v2;

	if ((3 * vH) < 2)
		return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

	return v1;
}

struct RGB hsl_to_rgb(struct HSL hsl) {
	RGB rgb;

	if (hsl.S == 0) {
		rgb.R = rgb.G = rgb.B = (unsigned char)(hsl.L * 255);
	} else {
		float v1, v2;
		float hue = (float)hsl.H / 360;

		v2 = (hsl.L < 0.5) ? (hsl.L * (1 + hsl.S)) : ((hsl.L + hsl.S) - (hsl.L * hsl.S));
		v1 = 2 * hsl.L - v2;

		rgb.R = (unsigned char)(255 * hue_to_rgb(v1, v2, hue + (1.0f / 3)));
		rgb.G = (unsigned char)(255 * hue_to_rgb(v1, v2, hue));
		rgb.B = (unsigned char)(255 * hue_to_rgb(v1, v2, hue - (1.0f / 3)));
	}

	return rgb;
}

void export_gradient_layer_to_png(AlbedoWeightsLayer* layer, char* fileName) {
    unsigned int size = layer->height * layer->width;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int i = 0; i < size; ++i) {
        unsigned char r = (layer->neurons[i].mask[1][0] + 0.5f) * 255.0f;   
        unsigned char g = (layer->neurons[i].mask[0][1] + 0.5f) * 255.0f;
        unsigned char b = (layer->neurons[i].mask[2][1] + 0.5f) * 255.0f;
        unsigned char a = (layer->neurons[i].mask[1][2] + 0.5f) * 255.0f;

        grid[i] = (a << 24) | (b << 16) | (g << 8) | (r);
    }   

    stbi_write_png(fileName, layer->width, layer->height, 4, grid, GRID_WIDTH*4);
    
    free(grid);
}

void export_state_layer_to_png(AlbedoNeuronLayer* layer, char* fileName) {
    unsigned int size = layer->height * layer->width;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int i = 0; i < size; ++i) {
        HSL hsl;
        hsl.H = (1.0 - layer->neurons[i]) * 240;
        hsl.L = 0.5f;
        hsl.S = 1.0f;

        RGB rgb = hsl_to_rgb(hsl);

        grid[i] = (255 << 24) | (rgb.B << 16) | (rgb.G << 8) | (rgb.R);
    }   

    stbi_write_png(fileName, layer->width, layer->height, 4, grid, GRID_WIDTH*4);
    
    free(grid);
}

#define SAMPLE_MODELS 100
#define TEST_CASES 4
#define STEPS 25

#define ALBEDO_EPSILON 0.05

float inputs[TEST_CASES][GRID_WIDTH] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
};

float outputs[TEST_CASES][GRID_WIDTH] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
};

void run_tests_on_model(AlbedoModel* bestModel) {
    // Show case best model
    float totalModelError = 0.0;

    for(int t = 0; t < TEST_CASES; ++t) {
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

    totalModelError /= (float) TEST_CASES;

    printf("Total modal error %f\n", totalModelError);
}

int main() {
    srand(time(0));

    AlbedoModel* bestModel = NULL;

    AlbedoModel* models[SAMPLE_MODELS] = { NULL };
    for(int i = 0; i < SAMPLE_MODELS; ++i)
        models[i] = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);

    for(int i = 0;;++i) {
        int bestIndex = -1;
        float bestError = FLT_MAX;

        // Run simulations across all models and get best
        for(int m = 0; m < SAMPLE_MODELS; ++m) {
            float error = 0.0;

            // Simulate each test case and collect error
            for(int t = 0; t < TEST_CASES; ++t) {
                memset(models[m]->state[0]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));
                memset(models[m]->state[1]->neurons, 0, GRID_WIDTH*GRID_HEIGHT*sizeof(float));

                for(int s = 0; s < STEPS; ++s) {
                    set_inputs_model(models[m], inputs[t]);
                    albedo_simulate_model_step(models[m]);
                    set_inputs_model(models[m], inputs[t]);
                }

                error += calculate_error(models[m], outputs[t]);
            }

            error /= (float) (TEST_CASES);
            
            if(error < bestError) {
                bestError = error;
                bestIndex = m;
            }
        }

        bestModel = models[bestIndex];

        printf("Simulated epoch %d, error %f, best index %d\n", i, bestError, bestIndex);

        if(bestError < 0.01) {
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
            albedo_tune_weights_layer(model->weights, 0.25);
        }

        albedo_free_model(bestModel);
    }

    run_tests_on_model(bestModel);
    albedo_free_model(bestModel);

    return 0;
}