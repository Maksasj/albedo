#ifndef ALBEDO_UTILS_H
#define ALBEDO_UTILS_H

#include <stdlib.h>

#define albedo_min(a,b) (((a)<(b))?(a):(b))
#define albedo_max(a,b) (((a)>(b))?(a):(b))

double albedo_clampd(double d, double min, double max) {
  const double t = d < min ? min : d;
  return t > max ? max : t;
}

float albedo_clampf(float d, float min, float max) {
  const float t = d < min ? min : d;
  return t > max ? max : t;
}

float albedo_randf(float min, float max) {
    return min + ((rand() % RAND_MAX) / (float) RAND_MAX) * (max - min);
}

#endif