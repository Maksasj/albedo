#ifndef ALBEDO_TESTING_H
#define ALBEDO_TESTING_H

#include <stdlib.h>

#include "albedo_model.h"

void albedo_sumup_testing(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,   
    unsigned int desiredSteps
);

#endif