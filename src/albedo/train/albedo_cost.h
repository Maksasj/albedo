#ifndef ALBEDO_COST_H
#define ALBEDO_COST_H

#include "../arc/albedo_model.h"

typedef kiwi_fixed_t (AlbedoCostFunction)(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int desiredStep
);

kiwi_fixed_t albedo_calculate_fixed_step_result_cost(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int desiredStep
);

kiwi_fixed_t albedo_calculate_continuous_result_cost(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int maxStep
);

#endif