#ifndef PEACH_H
#define PEACH_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef float peach_float_t;
#define PEACH_MULTIPLY(A, B) (A) * (B)
#define PEACH_RANDOM_FLOAT (rand() / (peach_float_t) RAND_MAX)

typedef struct peach_matrix_t {
    unsigned int rows;
    unsigned int cols;

    peach_float_t* value;
} peach_matrix_t;

#define PEACH_MATRIX_AT(M, ROW, COL) ((M)->value[(ROW) * M->cols + (COL)])

peach_matrix_t* paech_new_matrix(unsigned int rows, unsigned int cols);
peach_matrix_t* paech_new_matrix_square(unsigned int size); // Todo
peach_matrix_t* paech_new_matrix_random(unsigned int rows, unsigned int cols, peach_float_t min, peach_float_t max);

void peach_free_matrix(peach_matrix_t* matrix);

void peach_matrix_fill(peach_matrix_t* target, peach_float_t value);
void peach_matrix_copy_content_target(peach_matrix_t* dst, peach_matrix_t* src);

peach_matrix_t* peach_matrix_sum(peach_matrix_t* a, peach_matrix_t* b);
void peach_matrix_sum_target(peach_matrix_t* target, peach_matrix_t* b); // Todo

peach_matrix_t* peach_matrix_sub(peach_matrix_t* a, peach_matrix_t* b);
void peach_matrix_sub_target(peach_matrix_t* target, peach_matrix_t* b);

peach_matrix_t* peach_matrix_multiply(peach_matrix_t* a, peach_matrix_t* b);
peach_matrix_t* peach_matrix_dot(peach_matrix_t* a, peach_matrix_t* b);

void peach_matrix_scale(peach_matrix_t* target, peach_float_t value);

void peach_matrix_print(peach_matrix_t* m);

#endif