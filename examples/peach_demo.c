#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "peach/peach.h"

float data[4][3] = {
    { 0, 0, 0 },
    { 0, 1, 0 },
    { 1, 0, 0 },
    { 1, 1, 1 }
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
    printf("Forward pass:\n");

    PEACH_MATRIX_AT(and->inputs, 0, 0) = data[inputs][0];
    PEACH_MATRIX_AT(and->inputs, 1, 0) = data[inputs][1];
    printf("Input\n");
    peach_print_matrix(and->inputs);

    and->outputs = peach_matrix_multiply(and->inputs, and->weights);
    printf("weights\n");
    peach_print_matrix(and->weights);
    printf("outputs\n");
    peach_print_matrix(and->outputs);

    unsigned int width = and->outputs->width;
    unsigned int height = and->outputs->height;

    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            PEACH_MATRIX_AT(and->outputs, i, j) = sigmoidf(PEACH_MATRIX_AT(and->outputs, i, j));
        }
    }
}

float cost(And* and, float expected) {
    float d = and->outputs->value[0] - expected;
    
    return d * d;
}

static float epsilon = 0.1f;

void train(And* and, And* gradient) {
    unsigned int width = and->weights->width;
    unsigned int height = and->weights->height;

    float initialCost = cost(and, data[3][2]);

    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            float saved = PEACH_MATRIX_AT(and->weights, i, j);
            
            PEACH_MATRIX_AT(and->weights, i, j) += epsilon;
            
            forward_pass(and, 3);
            float newCost = cost(and, data[3][2]);


            printf("Adjust: %f %f\n", initialCost, newCost);

            peach_print_matrix(and->weights);
            PEACH_MATRIX_AT(gradient->weights, i, j) = (newCost - newCost) / epsilon;

            PEACH_MATRIX_AT(and->weights, i, j) = saved;
        }
    }
}

void apply_gradient(And* and, And* gradient) {
    peach_matrix_t* res = peach_matrix_substract(and->weights, gradient->weights);

    peach_free_matrix(and->weights);

    and->weights = res;
}

int main() {
    srand(time(0));

    And and;

    and.inputs = paech_new_matrix_random(2, 1, -1.0f, 1.0f);
    and.weights = paech_new_matrix_random(1, 2, -1.0f, 1.0f);
    and.outputs = peach_matrix_multiply(and.inputs, and.weights);
    peach_print_matrix(and.weights);

    forward_pass(&and, 3);
    printf("Cost: %f\n", cost(&and, 1.0f));

    And gradient;
    gradient.weights = paech_new_matrix(1, 2);
    peach_print_matrix(gradient.weights);

    train(&and, &gradient);
    apply_gradient(&and, &gradient);
    
    peach_print_matrix(gradient.weights);
    peach_print_matrix(and.weights);

    // forward_pass(&and, 3);



    /*
    printf("Cost: %f\n", cost(&and, 1.0f));
*/
    // peach_free_matrix(gradient.inputs);
    // peach_free_matrix(gradient.weights);
    // peach_free_matrix(gradient.outputs);

    return 0;
}