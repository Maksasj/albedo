#ifndef ALBEDO_TRAINING_H
#define ALBEDO_TRAINING_H

#include "albedo_model.h"

#define SAMPLE_MODELS 100
#define ALBEDO_MAX_EPOCHS 1000

void albedo_genetic_algorithm_training_internal(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps,
    void (*snapshotCallback)(AlbedoModel*, AlbedoNeuronValue**, unsigned int)
);

void albedo_genetic_algorithm_training(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps
);

#endif