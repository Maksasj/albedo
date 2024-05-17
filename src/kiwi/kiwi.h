#ifndef KIWI_H
#define KIWI_H

#include <stdint.h>

#define KIWI_FRACTIONAL_BITS 16
#define KIWI_SCALE (1 << KIWI_FRACTIONAL_BITS)

typedef int32_t kiwi_fixed_t;

static inline kiwi_fixed_t int_to_fixed(int32_t value) {
    return value * KIWI_SCALE;
}

static inline int32_t fixed_to_int(kiwi_fixed_t value) {
    return value / KIWI_SCALE;
}

static inline kiwi_fixed_t float_to_fixed(float value) {
    return (kiwi_fixed_t)(value * KIWI_SCALE);
}

static inline float fixed_to_float(kiwi_fixed_t value) {
    return (float)value / KIWI_SCALE;
}

static inline kiwi_fixed_t fixed_add(kiwi_fixed_t a, kiwi_fixed_t b) {
    return a + b;
}

static inline kiwi_fixed_t fixed_sub(kiwi_fixed_t a, kiwi_fixed_t b) {
    return a - b;
}

static inline kiwi_fixed_t fixed_mul(kiwi_fixed_t a, kiwi_fixed_t b) {
    return (kiwi_fixed_t)(((int64_t)a * b) >> KIWI_FRACTIONAL_BITS);
}

static inline kiwi_fixed_t fixed_div(kiwi_fixed_t a, kiwi_fixed_t b) {
    return (kiwi_fixed_t)(((int64_t)a << KIWI_FRACTIONAL_BITS) / b);
}

#endif