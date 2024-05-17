#ifndef ALBEDO_VISUALIZATION_H
#define ALBEDO_VISUALIZATION_H 

#include <stdlib.h>
#include "stb_image_write.h"
#include "stb_image.h"

#include "albedo/albedo.h"

typedef struct RGB {
	unsigned char R;
	unsigned char G;
	unsigned char B;
} RGB;

typedef struct HSL {
	int H;
	float S;
	float L;
} HSL;

float hue_to_rgb(float v1, float v2, float vH) {
	if (vH < 0)
		vH += 1;

	if (vH > 1)
		vH -= 1;

	if ((6 * vH) < 1)
		return (v1 + (v2 - v1) * 6 * vH);

	if ((2 * vH) < 1)
		return v2;

	if ((3 * vH) < 2)
		return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

	return v1;
}

struct RGB hsl_to_rgb(struct HSL hsl) {
	RGB rgb;

	if (hsl.S == 0) {
		rgb.R = rgb.G = rgb.B = (unsigned char)(hsl.L * 255);
	} else {
		float v1, v2;
		float hue = (float)hsl.H / 360;

		v2 = (hsl.L < 0.5) ? (hsl.L * (1 + hsl.S)) : ((hsl.L + hsl.S) - (hsl.L * hsl.S));
		v1 = 2 * hsl.L - v2;

		rgb.R = (unsigned char)(255 * hue_to_rgb(v1, v2, hue + (1.0f / 3)));
		rgb.G = (unsigned char)(255 * hue_to_rgb(v1, v2, hue));
		rgb.B = (unsigned char)(255 * hue_to_rgb(v1, v2, hue - (1.0f / 3)));
	}

	return rgb;
}

void export_gradient_layer_to_png(AlbedoWeightsLayer* layer, char* fileName) {
    unsigned int size = layer->height * layer->width;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int i = 0; i < size; ++i) {
        unsigned char r = (layer->weights[i].kernel[1][0] + 0.5f) * 255.0f;   
        unsigned char g = (layer->weights[i].kernel[0][1] + 0.5f) * 255.0f;
        unsigned char b = (layer->weights[i].kernel[2][1] + 0.5f) * 255.0f;
        unsigned char a = (layer->weights[i].kernel[1][2] + 0.5f) * 255.0f;

        grid[i] = (a << 24) | (b << 16) | (g << 8) | (r);
    }   

    stbi_write_png(fileName, layer->width, layer->height, 4, grid, layer->width*4);
    
    free(grid);
}

void export_state_layer_to_png(AlbedoNeuronLayer* layer, char* fileName) {
    unsigned int size = layer->height * layer->width;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int i = 0; i < size; ++i) {
        HSL hsl;
        hsl.H = (1.0 - layer->neurons[i]) * 240;
        hsl.L = 0.5f;
        hsl.S = 1.0f;

        RGB rgb = hsl_to_rgb(hsl);

        grid[i] = (255 << 24) | (rgb.B << 16) | (rgb.G << 8) | (rgb.R);
    }   

    stbi_write_png(fileName, layer->width, layer->height, 4, grid, layer->width*4);
    
    free(grid);
}

#endif