#ifndef ALBEDO_UTILS_H
#define ALBEDO_UTILS_H

#include <stdlib.h>

#define PEACH_IMPLEMENTATION
#include "kiwi.h"

#define albedo_min(a,b) (((a)<(b))?(a):(b))
#define albedo_max(a,b) (((a)>(b))?(a):(b))

double albedo_clampd(double d, double min, double max);
float albedo_clampf(float d, float min, float max);
kiwi_fixed_t albedo_clamp_fixed(kiwi_fixed_t d, kiwi_fixed_t min, kiwi_fixed_t max);

float albedo_randf(float min, float max);
kiwi_fixed_t albedo_rand_fixed(kiwi_fixed_t min, kiwi_fixed_t max);

#endif