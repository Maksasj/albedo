#ifndef ALBEDO_COST_H
#define ALBEDO_COST_H

#include "albedo_model.h"

typedef float (AlbedoCostFunction)(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int desiredStep
);

float albedo_calculate_fixed_step_result_cost(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int desiredStep
);

float albedo_calculate_continuous_result_cost(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int maxStep
);

#endif