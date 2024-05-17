#include "albedo_utils.h"

double albedo_clampd(double d, double min, double max) {
  	const double t = d < min ? min : d;
  	return t > max ? max : t;
}

float albedo_clampf(float d, float min, float max) {
  	const float t = d < min ? min : d;
  	return t > max ? max : t;
}

kiwi_fixed_t albedo_clamp_fixed(kiwi_fixed_t d, kiwi_fixed_t min, kiwi_fixed_t max) {
	const kiwi_fixed_t t = d < min ? min : d;
  	return t > max ? max : t;
}

float albedo_randf(float min, float max) {
    return min + ((rand() % RAND_MAX) / (float) RAND_MAX) * (max - min);
}

kiwi_fixed_t albedo_rand_fixed(kiwi_fixed_t min, kiwi_fixed_t max) {
	return kiwi_float_to_fixed(albedo_randf(kiwi_fixed_to_float(min), kiwi_fixed_to_float(max)));
}