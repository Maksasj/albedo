#ifndef ALBEDO_TRAINING_H
#define ALBEDO_TRAINING_H

#include "albedo_cost.h"
#include "../arc/albedo_model.h"

#define SAMPLE_MODELS 100
#define ALBEDO_MAX_EPOCHS 1000000

// Todo add some cost functions

typedef struct AlbedoTrainingSnapshot {
    // Common values
    AlbedoModel* model;
    AlbedoNeuronValue** inputs;
    AlbedoNeuronValue** outputs; 
    unsigned int testCases;
    unsigned int inputCount;
    unsigned int outputCount;
    kiwi_fixed_t desiredCost;
    unsigned int desiredSteps;

    unsigned int epoch;
    kiwi_fixed_t currentCost;
} AlbedoTrainingSnapshot;

typedef void (AlbedoTrainingSnapshotCallback)(AlbedoTrainingSnapshot*);

void albedo_genetic_algorithm_training_internal(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    kiwi_fixed_t epsilon,
    kiwi_fixed_t desiredCost,
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
    kiwi_fixed_t epsilon,
    kiwi_fixed_t desiredCost,
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
    kiwi_fixed_t desiredCost,
    unsigned int desiredSteps,
    kiwi_fixed_t epsilon,
    kiwi_fixed_t learningRate,
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
    kiwi_fixed_t desiredCost,
    unsigned int desiredSteps,
    kiwi_fixed_t epsilon,
    kiwi_fixed_t learningRate,
    AlbedoCostFunction* costFunction
); 

#endif