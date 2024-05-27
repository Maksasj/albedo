#include <stdlib.h>
#include <time.h>
#include <math.h>

#define PEACH_IMPLEMENTATION
#include "peach/peach.h"

typedef struct Arc {
    peach_matrix_t* inputs;
    peach_matrix_t* weights;
    
    peach_matrix_t* bias;
    peach_matrix_t* outputs;
} Arc;

void init_arc(Arc* arc);
void print_arc(Arc* arc);
void feed_forward(Arc* arc, peach_matrix_t* input);
float arc_cost(Arc* arc, peach_matrix_t* inputs[], peach_matrix_t* outputs[], unsigned int count);

void calc_gradient(Arc* arc, peach_matrix_t* weightGradient, peach_matrix_t* biasGradient, peach_matrix_t* inputs[], peach_matrix_t* outputs[], unsigned int sampleCount) {
    peach_matrix_fill(weightGradient, 0.0f);
    peach_matrix_fill(biasGradient, 0.0f);

    float initialCost = arc_cost(arc, inputs, outputs, sampleCount);
    
    static float epsilon = 0.05f;

    for(int t = 0; t < sampleCount; ++t) {
        unsigned int weightSize = weightGradient->rows * weightGradient->cols;

        // Weights
        for(int i = 0; i < weightSize; ++i) {
            float saved = arc->weights->value[i];
            arc->weights->value[i] += epsilon;

            float cost = arc_cost(arc, inputs, outputs, sampleCount);
            weightGradient->value[i] += (cost - initialCost) / epsilon;    

            arc->weights->value[i] = saved;        
        } 

        unsigned int biasSize = biasGradient->rows * biasGradient->cols;

        // Bias
        for(int i = 0; i < biasSize; ++i) {
            float saved = arc->bias->value[i];
            arc->bias->value[i] += epsilon;

            float cost = arc_cost(arc, inputs, outputs, sampleCount);
            weightGradient->value[i] += (cost - initialCost) / epsilon;    

            arc->bias->value[i] = saved;        
        } 
    }
}

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

    Arc arc;
    init_arc(&arc);

    peach_matrix_t* weightGradient = paech_new_matrix(2, 1);
    peach_matrix_t* biasGradient = paech_new_matrix(2, 1);
    static float learningRate = 0.01f;

    for(int i = 0; i < 1000; ++i) {
        calc_gradient(&arc, weightGradient, biasGradient, inputs, outputs, 4);
        
        peach_matrix_scale(weightGradient, learningRate);
        peach_matrix_scale(biasGradient, learningRate);

        peach_matrix_sub_target(arc.weights, weightGradient);
        peach_matrix_sub_target(arc.bias, biasGradient);

        printf("Cost: %f\n", arc_cost(&arc, inputs, outputs, 4));
    }

    for(int i = 0; i < 4; ++i) {
        printf("Test case: %d\n", i);
        peach_matrix_print(inputs[i]);
        feed_forward(&arc, inputs[i]);
        peach_matrix_print(arc.outputs);
    }

    return 0;
}

void init_arc(Arc* arc) {
    arc->inputs = paech_new_matrix(1, 2);

    arc->weights = paech_new_matrix_random(2, 1, -1.0f, 1.0f);
    arc->bias = paech_new_matrix_random(1, 1, 0.0f, 1.0f);

    arc->outputs = peach_matrix_dot(arc->inputs, arc->weights);
}

void print_arc(Arc* arc) {
    peach_matrix_print(arc->inputs);

    peach_matrix_print(arc->weights);
    peach_matrix_print(arc->bias);
    
    peach_matrix_print(arc->outputs);
}

void feed_forward(Arc* arc, peach_matrix_t* input) {
    peach_matrix_copy_content_target(arc->inputs, input);

    peach_free_matrix(arc->outputs);

    arc->outputs = peach_matrix_dot(arc->inputs, arc->weights);
    peach_matrix_sum_target(arc->outputs, arc->bias);

    peach_matrix_apply_sigmoid(arc->outputs);
}

float arc_cost(Arc* arc, peach_matrix_t* inputs[], peach_matrix_t* outputs[], unsigned int count) {
    float cost = 0.0f;

    for(int t = 0; t < count; ++t) {
        peach_matrix_t* input = inputs[t];
        peach_matrix_t* output = outputs[t];
        
        feed_forward(arc, input);

        unsigned int rows = output->rows;
        unsigned int cols = output->cols;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float d = PEACH_MATRIX_AT(arc->outputs, i, j) - PEACH_MATRIX_AT(output, i, j);
                cost += d*d;
            }
        }
    }
    
    return cost / (float) count;
}