#ifndef ALBEDO_TRAINING_H
#define ALBEDO_TRAINING_H

#include "albedo_cost.h"
#include "albedo_model.h"

#define SAMPLE_MODELS 100
#define ALBEDO_MAX_EPOCHS 10000

// Todo add some cost functions

typedef struct AlbedoTrainingSnapshot {
    // Common values
    AlbedoModel* model;
    AlbedoNeuronValue** inputs;
    AlbedoNeuronValue** outputs; 
    unsigned int testCases;
    unsigned int inputCount;
    unsigned int outputCount;
    float desiredCost;
    unsigned int desiredSteps;

    unsigned int epoch;
    float currentCost;
} AlbedoTrainingSnapshot;

typedef void (AlbedoTrainingSnapshotCallback)(AlbedoTrainingSnapshot*);

void albedo_genetic_algorithm_training_internal(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float epsilon,
    float desiredCost,
    unsigned int desiredSteps,
    AlbedoCostFunction* costFunction,
    AlbedoTrainingSnapshotCallback* snapshotCallback
);

void albedo_genetic_algorithm_training(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float epsilon,
    float desiredCost,
    unsigned int desiredSteps,
    AlbedoCostFunction* costFunction
);

void albedo_finite_difference_training_internal(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredCost,
    unsigned int desiredSteps,
    float epsilon,
    float learningRate,
    AlbedoCostFunction* costFunction,
    AlbedoTrainingSnapshotCallback *snapshotCallback
);

void albedo_finite_difference_training(
    AlbedoModel* model,
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    float desiredCost,
    unsigned int desiredSteps,
    float epsilon,
    float learningRate,
    AlbedoCostFunction* costFunction
); 

#endif