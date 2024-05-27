#ifndef PEACH_H
#define PEACH_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef float peach_float_t;

#define PEACH_INLINE static inline
#define PEACH_EULER_NUMBER 2.71828182846

#define PEACH_MULTIPLY(A, B) (A) * (B)
#define PEACH_RANDOM_FLOAT (rand() / (peach_float_t) RAND_MAX)
#define PEACH_MATRIX_AT(M, ROW, COL) ((M)->value[(ROW) * M->cols + (COL)])

#ifndef PEACH_ASSERT
    #include <assert.h>
    #define PEACH_ASSERT(EXP) assert((EXP))
#endif

typedef struct peach_matrix_t {
    unsigned int rows;
    unsigned int cols;

    peach_float_t* value;
} peach_matrix_t;

PEACH_INLINE peach_float_t peach_sigmoid(peach_float_t n);
PEACH_INLINE peach_float_t peach_relu(peach_float_t n);

PEACH_INLINE peach_matrix_t* paech_new_matrix(unsigned int rows, unsigned int cols);
PEACH_INLINE peach_matrix_t* paech_new_matrix_square(unsigned int size);
PEACH_INLINE peach_matrix_t* paech_new_matrix_random(unsigned int rows, unsigned int cols, peach_float_t min, peach_float_t max);

PEACH_INLINE peach_matrix_t* paech_copy_matrix(peach_matrix_t* src);

PEACH_INLINE void peach_free_matrix(peach_matrix_t* matrix);

PEACH_INLINE void peach_matrix_fill(peach_matrix_t* target, peach_float_t value);
PEACH_INLINE void peach_matrix_rand(peach_matrix_t* target, peach_float_t min, peach_float_t max);
PEACH_INLINE void peach_matrix_copy_content_target(peach_matrix_t* dst, peach_matrix_t* src);
PEACH_INLINE void peach_matrix_scale(peach_matrix_t* target, peach_float_t value);

PEACH_INLINE peach_matrix_t* peach_matrix_sum(peach_matrix_t* a, peach_matrix_t* b);
PEACH_INLINE peach_matrix_t* peach_matrix_sub(peach_matrix_t* a, peach_matrix_t* b);
PEACH_INLINE peach_matrix_t* peach_matrix_mul(peach_matrix_t* a, peach_matrix_t* b);
PEACH_INLINE peach_matrix_t* peach_matrix_div(peach_matrix_t* a, peach_matrix_t* b);

PEACH_INLINE peach_matrix_t* peach_matrix_dot(peach_matrix_t* a, peach_matrix_t* b);

PEACH_INLINE void peach_matrix_sum_target(peach_matrix_t* target, peach_matrix_t* b);
PEACH_INLINE void peach_matrix_sub_target(peach_matrix_t* target, peach_matrix_t* b);
PEACH_INLINE void peach_matrix_mul_target(peach_matrix_t* target, peach_matrix_t* b);
PEACH_INLINE void peach_matrix_div_target(peach_matrix_t* target, peach_matrix_t* b);

PEACH_INLINE void peach_matrix_dot_target(peach_matrix_t* target, peach_matrix_t* a, peach_matrix_t* b);

PEACH_INLINE void peach_matrix_apply_sigmoid(peach_matrix_t* target);
PEACH_INLINE void peach_matrix_apply_relu(peach_matrix_t* target);

PEACH_INLINE void peach_matrix_print(peach_matrix_t* m);

#ifdef PEACH_IMPLEMENTATION

PEACH_INLINE peach_float_t peach_sigmoid(peach_float_t n) {
    return (1.0f / (1 + powf(PEACH_EULER_NUMBER, -n)));
}

PEACH_INLINE peach_float_t peach_relu(peach_float_t n) {
    if(n < 0)
        return 0;

    return n;
}

PEACH_INLINE peach_matrix_t* paech_new_matrix(unsigned int rows, unsigned int cols) {
    const unsigned long long size = rows * cols * sizeof(peach_float_t);

    peach_matrix_t* matrix = (peach_matrix_t*) malloc(sizeof(peach_matrix_t));
    
    matrix->rows = rows;
    matrix->cols = cols;
    
    matrix->value = malloc(size);
    memset(matrix->value, 0, size);

    return matrix;
}

PEACH_INLINE peach_matrix_t* paech_new_matrix_square(unsigned int size) {
    return paech_new_matrix(size, size);
} 

PEACH_INLINE peach_matrix_t* paech_new_matrix_random(unsigned int rows, unsigned int cols, peach_float_t min, peach_float_t max) {
    peach_matrix_t* matrix = paech_new_matrix(rows, cols);

    peach_matrix_rand(matrix, min, max);
    return matrix;
}

PEACH_INLINE peach_matrix_t* paech_copy_matrix(peach_matrix_t* src) {
    const unsigned int rows = src->rows;
    const unsigned int cols = src->cols;

    peach_matrix_t* matrix = paech_new_matrix(rows, cols);
    peach_matrix_copy_content_target(matrix, src);

    return matrix;
}

PEACH_INLINE void peach_free_matrix(peach_matrix_t* matrix) {
    free(matrix->value);
    free(matrix);
}

PEACH_INLINE void peach_matrix_fill(peach_matrix_t* target, peach_float_t value) {
    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) = value;
        }
    }
}

PEACH_INLINE void peach_matrix_rand(peach_matrix_t* target, peach_float_t min, peach_float_t max) {
    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) = min + PEACH_MULTIPLY(PEACH_RANDOM_FLOAT, (max - min));;
        }
    }
}

PEACH_INLINE void peach_matrix_copy_content_target(peach_matrix_t* dst, peach_matrix_t* src) {
    PEACH_ASSERT(dst->rows == src->rows);
    PEACH_ASSERT(dst->cols == src->cols);

    const unsigned int rows = dst->rows;
    const unsigned int cols = dst->cols;

    const unsigned long long size = rows * cols * sizeof(peach_float_t);

    memcpy(dst->value, src->value, size);
}

PEACH_INLINE void peach_matrix_scale(peach_matrix_t* target, peach_float_t value) {
    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) *= value;
        }
    }
}

PEACH_INLINE peach_matrix_t* peach_matrix_sum(peach_matrix_t* a, peach_matrix_t* b) {
    PEACH_ASSERT(a->rows == b->rows);
    PEACH_ASSERT(a->cols == b->cols);

    const unsigned int rows = a->rows;
    const unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) + PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

PEACH_INLINE peach_matrix_t* peach_matrix_sub(peach_matrix_t* a, peach_matrix_t* b) {
    PEACH_ASSERT(a->rows == b->rows);
    PEACH_ASSERT(a->cols == b->cols);

    const unsigned int rows = a->rows;
    const unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) - PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

PEACH_INLINE peach_matrix_t* peach_matrix_mul(peach_matrix_t* a, peach_matrix_t* b) {
    PEACH_ASSERT(a->rows == b->rows);
    PEACH_ASSERT(a->cols == b->cols);

    const unsigned int rows = a->rows;
    const unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) * PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

PEACH_INLINE peach_matrix_t* peach_matrix_div(peach_matrix_t* a, peach_matrix_t* b) {
    PEACH_ASSERT(a->rows == b->rows);
    PEACH_ASSERT(a->cols == b->cols);

    const unsigned int rows = a->rows;
    const unsigned int cols = a->cols;

    peach_matrix_t* result = paech_new_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(result, i, j) = PEACH_MATRIX_AT(a, i, j) / PEACH_MATRIX_AT(b, i, j);
        }
    }

    return result;
}

PEACH_INLINE peach_matrix_t* peach_matrix_dot(peach_matrix_t* a, peach_matrix_t* b) {
    PEACH_ASSERT(a->cols == b->rows);

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

PEACH_INLINE void peach_matrix_sum_target(peach_matrix_t* target, peach_matrix_t* b) {
    PEACH_ASSERT(target->rows == b->rows);
    PEACH_ASSERT(target->cols == b->cols);

    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) += PEACH_MATRIX_AT(b, i, j);
        }
    }
}

PEACH_INLINE void peach_matrix_sub_target(peach_matrix_t* target, peach_matrix_t* b) {
    PEACH_ASSERT(target->rows == b->rows);
    PEACH_ASSERT(target->cols == b->cols);

    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) -= PEACH_MATRIX_AT(b, i, j);
        }
    }
}

PEACH_INLINE void peach_matrix_mul_target(peach_matrix_t* target, peach_matrix_t* b) {
    PEACH_ASSERT(target->rows == b->rows);
    PEACH_ASSERT(target->cols == b->cols);

    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) *= PEACH_MATRIX_AT(b, i, j);
        }
    }
}

PEACH_INLINE void peach_matrix_div_target(peach_matrix_t* target, peach_matrix_t* b) {
    PEACH_ASSERT(target->rows == b->rows);
    PEACH_ASSERT(target->cols == b->cols);

    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) /= PEACH_MATRIX_AT(b, i, j);
        }
    }
}

PEACH_INLINE void peach_matrix_dot_target(peach_matrix_t* target, peach_matrix_t* a, peach_matrix_t* b) {
    PEACH_ASSERT(a->cols == b->rows);

    unsigned int n = a->cols;
    
    unsigned int rows = a->rows;
    unsigned int cols = b->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) = 0;

            for (int k = 0; k < n; ++k) {
                PEACH_MATRIX_AT(target, i, j) += PEACH_MATRIX_AT(a, i, k) * PEACH_MATRIX_AT(b, k, j);
            }
        }
    }
}

PEACH_INLINE void peach_matrix_apply_sigmoid(peach_matrix_t* target) {
    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const peach_float_t value = peach_sigmoid(PEACH_MATRIX_AT(target, i, j));
            PEACH_MATRIX_AT(target, i, j) = value;
        }
    }
}

PEACH_INLINE void peach_matrix_apply_relu(peach_matrix_t* target) {
    const unsigned int rows = target->rows;
    const unsigned int cols = target->cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            PEACH_MATRIX_AT(target, i, j) = peach_relu(PEACH_MATRIX_AT(target, i, j));
        }
    }
}

PEACH_INLINE void peach_matrix_print(peach_matrix_t* m) {
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

#endif