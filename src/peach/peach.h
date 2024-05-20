#ifndef PEACH_H
#define PEACH_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef float peach_float_t;
#define PEACH_MULTIPLY(A, B) (A) * (B)
#define PEACH_RANDOM_FLOAT (rand() / (float) RAND_MAX)

typedef struct peach_matrix_t {
    unsigned int width;
    unsigned int height;

    peach_float_t* value;
} peach_matrix_t;

#define PEACH_MATRIX_AT(M, X, Y) ((M)->value[(X) + (Y)*(M->width)])

peach_matrix_t* paech_new_matrix(unsigned int width, unsigned int height) {
    unsigned long long size = width * height * sizeof(peach_float_t);

    peach_matrix_t* matrix = (peach_matrix_t*) malloc(sizeof(peach_matrix_t));
    
    matrix->width = width;
    matrix->height = height;
    
    matrix->value = malloc(size);
    memset(matrix->value, 0, size);

    return matrix;
}

peach_matrix_t* paech_new_matrix_random(unsigned int width, unsigned int height, peach_float_t min, peach_float_t max) {
    peach_matrix_t* matrix = paech_new_matrix(width, height);

    for(int i = 0; i < width; ++i) {
        for(int j = 0; j < height; ++j) {
            PEACH_MATRIX_AT(matrix, i, j) = min + PEACH_MULTIPLY(PEACH_RANDOM_FLOAT, (max - min));
        }
    }

    return matrix;
}

void peach_free_matrix(peach_matrix_t* matrix) {
    free(matrix->value);
    free(matrix);
}

peach_matrix_t* peach_matrix_substract(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->width != b->width)
        return NULL;

    if(a->height != b->height)
        return NULL;

    unsigned int width = a->width;
    unsigned int height = a->height;

    peach_matrix_t* result = paech_new_matrix(width, height);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) - PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

peach_matrix_t* peach_matrix_sum(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->width != b->width)
        return NULL;

    if(a->height != b->height)
        return NULL;

    unsigned int width = a->width;
    unsigned int height = a->height;

    peach_matrix_t* result = paech_new_matrix(width, height);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) + PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

peach_matrix_t* peach_matrix_multiply(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->width != b->height)
        return NULL;

    unsigned int width = b->width;
    unsigned int height = a->height;

    peach_matrix_t* result = paech_new_matrix(width, height);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            PEACH_MATRIX_AT(result, j, i) = 0;

            for(int k = 0; k < a->width; ++k)
                PEACH_MATRIX_AT(result, j, i) += PEACH_MULTIPLY(PEACH_MATRIX_AT(a, k, i), PEACH_MATRIX_AT(b, j, k));
        }
    }

    return result;
}

void peach_print_matrix(peach_matrix_t* m) {
    if(m == NULL) {
        printf("NULL\n");
        return;
    }

    unsigned int width = m->width;
    unsigned int height = m->height;

    printf("(%d %d) = {\n", width, height);

    for(int j = 0; j < height; ++j) { 
        printf("    ");

        for(int i = 0; i < width; ++i) {
            printf("%f, ", PEACH_MATRIX_AT(m, j, i));
        }

        printf("\n");
    }

    printf("}\n");
}

#endif