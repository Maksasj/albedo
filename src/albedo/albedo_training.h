#ifndef ALBEDO_TRAINING_H
#define ALBEDO_TRAINING_H

#include "albedo_model.h"

#define SAMPLE_MODELS 100
#define ALBEDO_MAX_EPOCHS 10000

float albedo_model_calculate_error(
    AlbedoModel* model,
    AlbedoNeuronValue* inputs,
    AlbedoNeuronValue* outputs, 
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int steps
);

float albedo_model_calculate_error_from_tests(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int steps
);

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

void albedo_finite_difference_training(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredError,
    unsigned int desiredSteps,
    float epsilon,
    float learningRate
); 

#endif