#include "albedo.h"

float albedo_calculate_fixed_step_result_cost(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int desiredStep
) {
    float cost = 0.0f;

    for(int t = 0; t < testCases; ++t) {
        albedo_reset_model_neurons_value(model);

        for(int s = 0; s < desiredStep; ++s) {
            albedo_set_model_neurons_values(model, inputs[t], inputCount);
            albedo_simulate_model_step(model);
        }

        cost += albedo_get_dif_model_neurons_values(model, outputs[t], outputCount);
    }

    cost /= (float) (testCases * outputCount);

    return cost;
}

float albedo_calculate_continuous_result_cost(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int maxStep
) {
    return 10.0f;
}
