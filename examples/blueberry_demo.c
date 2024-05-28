#include <time.h>

#include <math.h>
#include <stdlib.h>

#define PEACH_POWF powf
#define PEACH_MALLOC malloc
#define PEACH_FREE free

#define BLUEBERRY_IMPLEMENTATION
#define PEACH_IMPLEMENTATION
#include "blueberry/blueberry.h"

int main() {
    srand(time(0));

    peach_matrix_t* inputs[] = {
        paech_new_matrix(1, 2),
        paech_new_matrix(1, 2),
        paech_new_matrix(1, 2),
        paech_new_matrix(1, 2),
    };

    peach_matrix_t* outputs[] = {
        paech_new_matrix(1, 1),
        paech_new_matrix(1, 1),
        paech_new_matrix(1, 1),
        paech_new_matrix(1, 1),
    };

    for(int i = 0; i < 4; ++i) {
        peach_matrix_fill(inputs[i], 0.0f);
        peach_matrix_fill(outputs[i], 0.0f);
    }

    // Setup inputs
    PEACH_MATRIX_AT(inputs[1], 0, 0) = 1.0f;
    PEACH_MATRIX_AT(inputs[2], 0, 1) = 1.0f;
    
    PEACH_MATRIX_AT(inputs[3], 0, 0) = 1.0f;
    PEACH_MATRIX_AT(inputs[3], 0, 1) = 1.0f;

    // Setup outputs
    PEACH_MATRIX_AT(outputs[3], 0, 0) = 1.0f;

    int arc[] = { 2, 2, 1 };
    BlueBerryModel* model = blueb_new_model(arc, 3);
    blueb_rand_model(model, -1.0f, 1.0f);

    blueb_train_gradient_descent(model, inputs, outputs, 4, 10000, 0.05f);

    printf("Cost %f\n", blueb_mse_cost(model, inputs, outputs, 4));

    for(int i = 0; i < 4; ++i) {
        printf("Test case: %d\n", i);
        blueb_feedforward(model, inputs[i]);

        peach_matrix_print(model->neurons[model->count - 1]);
    }

    blueb_free_model(model);

    return 0; 
}