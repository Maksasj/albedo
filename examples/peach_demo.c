#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "peach/peach.h"

float data[4][3] = {
    { 0.0f, 0.0f, 0.0f },
    { 0.0f, 1.0f, 0.0f },
    { 1.0f, 0.0f, 0.0f },
    { 1.0f, 1.0f, 1.0f }
};

typedef struct And {
    peach_matrix_t* inputs;
    peach_matrix_t* weights;
    peach_matrix_t* outputs;
} And;

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846
#define EULER_NUMBER_L 2.71828182845904523536

float sigmoidf(float n) {
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

void forward_pass(And* and, int inputs) {
    PEACH_MATRIX_AT(and->inputs, 0, 0) = data[inputs][0];
    PEACH_MATRIX_AT(and->inputs, 0, 1) = data[inputs][1];

    peach_free_matrix(and->outputs);
    and->outputs = peach_matrix_dot(and->inputs, and->weights);

    unsigned int rows = and->outputs->rows;
    unsigned int cols = and->outputs->cols;

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(and->outputs, i, j) = sigmoidf(PEACH_MATRIX_AT(and->outputs, i, j));
        }
    }
}

float cost(And* and) {
    float avg = 0.0f;

    for(int i = 0; i < 4; ++i) {
        forward_pass(and, i);
        float d = data[i][2] - and->outputs->value[0];
        avg += d * d;
    }

    return avg /= 4;
}

static float epsilon = 0.05f;

void train(And* and, And* gradient) {
    unsigned int rows = and->weights->rows;
    unsigned int cols = and->weights->cols;

    float initialCost = cost(and);

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            float saved = PEACH_MATRIX_AT(and->weights, i, j);
            PEACH_MATRIX_AT(and->weights, i, j) += epsilon;

            float newCost = cost(and);

            PEACH_MATRIX_AT(gradient->weights, i, j) = (newCost - initialCost) / epsilon;
            PEACH_MATRIX_AT(and->weights, i, j) = saved;
        }
    }
}

void apply_gradient(And* and, And* gradient) {
    peach_matrix_substract_target(and->weights, gradient->weights);
}

int main() {
    srand(time(0));

    And and;
    and.inputs = paech_new_matrix(1, 2);
    and.weights = paech_new_matrix_random(2, 1, -1.0f, 1.0f);
    and.outputs = paech_new_matrix(1, 1);

    And gradient;
    gradient.weights = paech_new_matrix(2, 1);

    for(int e = 0; e < 1000; ++e) {
        peach_matrix_fill(gradient.weights, 0.0f);

        train(&and, &gradient);

        peach_matrix_scale(gradient.weights, 0.1f);
        apply_gradient(&and, &gradient);

        float c = cost(&and);
        printf("Cost: %f\n", c);
    }

    for(int i = 0; i < 4; ++i) {
        forward_pass(&and, i);
        printf("(%f %f) -> %f\n", data[i][0], data[i][1], and.outputs[0]);
    }

    return 0;
}