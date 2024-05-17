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
AlbedoModel* albedo_copy_model(AlbedoModel* src);
void albedo_free_model(AlbedoModel* model);

void albedo_set_model_neurons_value(AlbedoModel* model, float value);
void albedo_reset_model_neurons_value(AlbedoModel* model);

void albedo_set_model_neurons_values(AlbedoModel* model, AlbedoNeuronValue* values, unsigned int count);
kiwi_fixed_t albedo_get_dif_model_neurons_values(AlbedoModel* model, AlbedoNeuronValue* values, unsigned int count);

void calculate_new_state(AlbedoNeuronLayer* newState, AlbedoNeuronLayer* oldState, AlbedoWeightsLayer* weights);

void albedo_simulate_model_step(AlbedoModel* model);
void albedo_simulate_model_steps(AlbedoModel* model, unsigned int steps);

#endif