#include <stdlib.h>
#include <time.h>
#include <math.h>

#define PEACH_IMPLEMENTATION
#include "peach/peach.h"

typedef struct Arc {
    peach_matrix_t* inputs;
    peach_matrix_t* bias0;

    peach_matrix_t* weights;
    
    peach_matrix_t* outputs;
    peach_matrix_t* bias1;
} Arc;

void init_arc(Arc* arc);
void print_arc(Arc* arc);
void feed_forward(Arc* arc, peach_matrix_t* input);
float arc_cost(Arc* arc, peach_matrix_t* inputs[], peach_matrix_t* outputs[], unsigned int count);

void calc_gradient(Arc* arc, Arc* gradient, peach_matrix_t* inputs[], peach_matrix_t* outputs[], unsigned int sampleCount) {
    peach_matrix_fill(gradient->bias0, 0.0f);
    peach_matrix_fill(gradient->bias1, 0.0f);
    peach_matrix_fill(gradient->weights, 0.0f);

    float initialCost = arc_cost(arc, inputs, outputs, sampleCount);
    
    static float epsilon = 0.05f;

    for(int t = 0; t < sampleCount; ++t) {
        unsigned int weightSize = gradient->weights->rows * gradient->weights->cols;

        // Weights
        for(int i = 0; i < weightSize; ++i) {
            float saved = arc->weights->value[i];
            arc->weights->value[i] += epsilon;

            float cost = arc_cost(arc, inputs, outputs, sampleCount);
            gradient->weights->value[i] += (cost - initialCost) / epsilon;    

            arc->weights->value[i] = saved;        
        } 

        unsigned int bias0Size = gradient->bias0->rows * gradient->bias0->cols;

        // Bias0
        for(int i = 0; i < bias0Size; ++i) {
            float saved = arc->bias0->value[i];
            arc->bias0->value[i] += epsilon;

            float cost = arc_cost(arc, inputs, outputs, sampleCount);
            gradient->bias0->value[i] += (cost - initialCost) / epsilon;    

            arc->bias0->value[i] = saved;        
        } 

        unsigned int bias1Size = gradient->bias1->rows * gradient->bias1->cols;

        // Bias1
        for(int i = 0; i < bias1Size; ++i) {
            float saved = arc->bias1->value[i];
            arc->bias1->value[i] += epsilon;

            float cost = arc_cost(arc, inputs, outputs, sampleCount);
            gradient->bias1->value[i] += (cost - initialCost) / epsilon;    

            arc->bias1->value[i] = saved;        
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

    Arc gradient;
    init_arc(&gradient);

    static float learningRate = 0.05f;

    for(int i = 0; i < 1000000; ++i) {
        calc_gradient(&arc, &gradient, inputs, outputs, 4);
        
        peach_matrix_scale(gradient.weights, learningRate);
        peach_matrix_scale(gradient.bias0, learningRate);
        peach_matrix_scale(gradient.bias1, learningRate);

        peach_matrix_sub_target(arc.weights, gradient.weights);
        peach_matrix_sub_target(arc.bias0, gradient.bias0);
        peach_matrix_sub_target(arc.bias1, gradient.bias1);

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
    arc->bias0 = paech_new_matrix_random(1, 2, -1.0f, 1.0f);

    arc->weights = paech_new_matrix_random(2, 1, -1.0f, 1.0f);

    arc->outputs = peach_matrix_dot(arc->inputs, arc->weights);
    arc->bias1 = paech_new_matrix_random(1, 1, -1.0f, 1.0f);
}

void print_arc(Arc* arc) {
    peach_matrix_print(arc->inputs);
    peach_matrix_print(arc->bias0);

    peach_matrix_print(arc->weights);
    
    peach_matrix_print(arc->outputs);
    peach_matrix_print(arc->bias1);
}

void feed_forward(Arc* arc, peach_matrix_t* input) {
    peach_matrix_copy_content_target(arc->inputs, input);
    peach_matrix_sum_target(arc->inputs, arc->bias0);
    peach_matrix_apply_sigmoid(arc->inputs);

    peach_free_matrix(arc->outputs);

    arc->outputs = peach_matrix_dot(arc->inputs, arc->weights);
    peach_matrix_sum_target(arc->outputs, arc->bias1);
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