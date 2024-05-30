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

    peach_float_t itmp[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    peach_matrix_t* inputs = paech_new_matrix(4, 2);
    peach_matrix_fill_values(inputs, itmp);

    peach_float_t otmp[] = {
        0.0f,
        0.0f,
        0.0f,
        1.0f
    };

    peach_matrix_t* outputs = paech_new_matrix(4, 1);
    peach_matrix_fill_values(outputs, otmp);

    int arc[] = { 2, 2, 1 };
    BlueBerryModel* model = blueb_new_model(arc, 3);
    blueb_rand_model(model, -1.0f, 1.0f);

    blueb_train_gradient_descent(model, inputs, outputs, 4, 10000, 0.05f);

    printf("Cost %f\n", blueb_mse_cost(model, inputs, outputs, 4));

    for(int i = 0; i < 4; ++i) {
        printf("Test case: %d\n", i);
        blueb_feedforward_values(model, PEACH_MATRIX_ROW(inputs, i));

        peach_matrix_print(model->neurons[model->count - 1]);
    }

    blueb_free_model(model);

    return 0; 
}