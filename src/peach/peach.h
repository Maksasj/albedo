#ifndef PEACH_H
#define PEACH_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef float peach_float_t;
#define PEACH_MULTIPLY(A, B) (A) * (B)
#define PEACH_RANDOM_FLOAT (rand() / (float) RAND_MAX)

typedef struct peach_matrix_t {
    unsigned int rows;
    unsigned int cols;

    peach_float_t* value;
} peach_matrix_t;

#define PEACH_MATRIX_AT(M, ROW, COL) ((M)->value[(ROW) * M->cols + (COL)])

peach_matrix_t* paech_new_matrix(unsigned int rows, unsigned int cols) {
    unsigned long long size = rows * cols * sizeof(peach_float_t);

    peach_matrix_t* matrix = (peach_matrix_t*) malloc(sizeof(peach_matrix_t));
    
    matrix->rows = rows;
    matrix->cols = cols;
    
    matrix->value = malloc(size);
    memset(matrix->value, 0, size);

    return matrix;
}

peach_matrix_t* paech_new_matrix_random(unsigned int rows, unsigned int cols, peach_float_t min, peach_float_t max) {
    peach_matrix_t* matrix = paech_new_matrix(rows, cols);

    for(int j = 0; j < cols; ++j) {
        for(int i = 0; i < rows; ++i) {
            PEACH_MATRIX_AT(matrix, i, j) = min + PEACH_MULTIPLY(PEACH_RANDOM_FLOAT, (max - min));
        }
    }

    return matrix;
}

void peach_free_matrix(peach_matrix_t* matrix) {
    free(matrix->value);
    free(matrix);
}

void peach_matrix_substract_target(peach_matrix_t* a, peach_matrix_t* b) {
    /*
    if(a->rows != b->rows)
        return NULL;

    if(a->cols != b->cols)
        return NULL;
    */

    unsigned int rows = a->rows;
    unsigned int cols = a->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(a, i, j) -= PEACH_MATRIX_AT(b, i, j);
        }
    }
}

peach_matrix_t* peach_matrix_substract(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->rows != b->rows)
        return NULL;

    if(a->cols != b->cols)
        return NULL;

    unsigned int rows = a->rows;
    unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) - PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

peach_matrix_t* peach_matrix_sum(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->rows != b->rows)
        return NULL;

    if(a->cols != b->cols)
        return NULL;

    unsigned int rows = a->rows;
    unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) + PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

void peach_matrix_fill(peach_matrix_t* target, peach_float_t value) {
    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) = value;
        }
    }
}

peach_matrix_t* peach_matrix_multiply(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->rows != b->rows)
        return NULL;

    if(a->cols != b->cols)
        return NULL;

    unsigned int rows = a->rows;
    unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) * PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

void peach_matrix_scale(peach_matrix_t* target, peach_float_t value) {
    unsigned int rows = target->rows;
    unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) *= value;
        }
    }
}

peach_matrix_t* peach_matrix_dot(peach_matrix_t* a, peach_matrix_t* b) {
    if(a->cols != b->rows)
        return NULL;

    unsigned int n = a->cols;
    
    unsigned int rows = a->rows;
    unsigned int cols = b->cols;

    peach_matrix_t* mat = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(mat, i, j) = 0;

            for (int k = 0; k < n; ++k) {
                PEACH_MATRIX_AT(mat, i, j) += PEACH_MATRIX_AT(a, i, k) * PEACH_MATRIX_AT(b, k, j);
            }
        }
    }

    return mat;
}

void peach_print_matrix(peach_matrix_t* m) {
    if(m == NULL) {
        printf("NULL\n");
        return;
    }

    unsigned int rows = m->rows;
    unsigned int cols = m->cols;

    printf("(%d %d) = {\n", rows, cols);

    for(int i = 0; i < rows; ++i) { 
        printf("    ");

        for(int j = 0; j < cols; ++j) {
            printf("%f, ", PEACH_MATRIX_AT(m, i, j));
        }

        printf("\n");
    }

    printf("}\n");
}

#endif