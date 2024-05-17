#include "albedo_testing.h"

void albedo_sumup_testing(
    AlbedoModel* model, 
    AlbedoNeuronValue** inputs,
    AlbedoNeuronValue** outputs, 
    unsigned int testCases,
    unsigned int inputCount,
    unsigned int outputCount,
    unsigned int desiredSteps
) {
    float error = 0.0;

    for(int t = 0; t < testCases; ++t) {
        albedo_reset_model_neurons_value(model);

        printf("Test case %d\n", t);

        printf("    Input data: ");
        for(int i = 0; i < inputCount; ++i)
            printf("(%d, %d, %f)", inputs[t][i].x, inputs[t][i].y, kiwi_fixed_to_float(inputs[t][i].value));
        printf("\n");

        printf("    Expected output: ");
        for(int i = 0; i < outputCount; ++i)
            printf("(%d, %d, %f)", outputs[t][i].x, outputs[t][i].y, kiwi_fixed_to_float(outputs[t][i].value));
        printf("\n");

        for(int i = 0; i < desiredSteps; ++i) {
            albedo_set_model_neurons_values(model, inputs[t], inputCount);
            albedo_simulate_model_step(model);
        }

        printf("    Gotten output: ");
        for(int i = 0; i < outputCount; ++i) {
            int x = outputs[t][i].x;
            int y = outputs[t][i].y;

            kiwi_fixed_t value = model->state[model->newIndex]->neurons[x + y*model->width];
            printf("(%d, %d, %f)", x, y, kiwi_fixed_to_float(value));
        }
        printf("\n");

        float localError = kiwi_fixed_to_float(albedo_get_dif_model_neurons_values(model, outputs[t], outputCount));
        error += localError;

        printf("    Test case error %f\n", error);
    }

    error /= (float) (testCases * outputCount);

    printf("Total modal error %f\n", error);
}
