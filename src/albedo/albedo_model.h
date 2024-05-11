#ifndef ALBEDO_MODEL_H
#define ALBEDO_MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "albedo_neurons.h"
#include "albedo_weights.h"

typedef struct AlbedoModel {
    AlbedoNeuronLayer* state[2];
    AlbedoWeightsLayer* weights;

    unsigned int iteration;
    unsigned char newIndex;    

    unsigned int width;
    unsigned int height;
} AlbedoModel;

AlbedoModel* albedo_new_model(unsigned int width, unsigned int height);
void albedo_free_model(AlbedoModel* model);

void calculate_new_state(AlbedoNeuronLayer* newState, AlbedoNeuronLayer* oldState, AlbedoWeightsLayer* weights);

void albedo_simulate_model_step(AlbedoModel* model);
void albedo_simulate_model_steps(AlbedoModel* model, unsigned int steps);

// =========================================================
void set_inputs_model(AlbedoModel* model, float input[]);

float calculate_error_delta(AlbedoModel* model, float expectedOutput[]);
float calculate_error(AlbedoModel* model, float expectedOutput[]);
// =========================================================

#endif