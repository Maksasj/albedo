#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "albedo/albedo.h"

#include "stb_image_write.h"

#define GRID_WIDTH 8
#define GRID_HEIGHT 8

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846
#define EULER_NUMBER_L 2.71828182845904523536

double sigmoid(double n) {
    return (1 / (1 + pow(EULER_NUMBER, -n)));
}

float sigmoidf(float n) {
    return (1 / (1 + powf(EULER_NUMBER_F, -n)));
}

long double sigmoidl(long double n) {
    return (1 / (1 + powl(EULER_NUMBER_L, -n)));
}


struct RGB
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

struct HSL
{
	int H;
	float S;
	float L;
};

float HueToRGB(float v1, float v2, float vH)
{
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

struct RGB HSLToRGB(struct HSL hsl) {
	struct RGB rgb;

	if (hsl.S == 0)
	{
		rgb.R = rgb.G = rgb.B = (unsigned char)(hsl.L * 255);
	}
	else
	{
		float v1, v2;
		float hue = (float)hsl.H / 360;

		v2 = (hsl.L < 0.5) ? (hsl.L * (1 + hsl.S)) : ((hsl.L + hsl.S) - (hsl.L * hsl.S));
		v1 = 2 * hsl.L - v2;

		rgb.R = (unsigned char)(255 * HueToRGB(v1, v2, hue + (1.0f / 3)));
		rgb.G = (unsigned char)(255 * HueToRGB(v1, v2, hue));
		rgb.B = (unsigned char)(255 * HueToRGB(v1, v2, hue - (1.0f / 3)));
	}

	return rgb;
}

void export_gradient_layer_to_png(GradientLayer* layer, char* fileName) {
    unsigned int size = layer->height * layer->width;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int i = 0; i < size; ++i) {
        unsigned char r = (layer->cells[i].value[UP]    + 0.5f) * 255.0f;
        unsigned char g = (layer->cells[i].value[RIGHT] + 0.5f) * 255.0f;
        unsigned char b = (layer->cells[i].value[DOWN]  + 0.5f) * 255.0f;
        unsigned char a = (layer->cells[i].value[LEFT]  + 0.5f) * 255.0f;

        grid[i] = (a << 24) | (b << 16) | (g << 8) | (r);
    }   

    stbi_write_png(fileName, layer->width, layer->height, 4, grid, GRID_WIDTH*4);
    
    free(grid);
}

typedef struct StateLayer {
    unsigned int width;
    unsigned int height;

    float* cells;
} StateLayer;

StateLayer* create_state_layer(unsigned int width, unsigned int height) {
    StateLayer* layer = (StateLayer*) malloc(sizeof(StateLayer));

    layer->width = width;
    layer->height = height;

    unsigned int size = width * height;

    layer->cells = (float*) malloc(size * sizeof(float));

    for(int i = 0; i < size; ++i) {
        layer->cells[i] = 0.0f;
    }

    return layer;
}

void free_states(StateLayer* states) {
    free(states->cells);
    free(states);
}

void export_state_layer_to_png(StateLayer* layer, char* fileName) {
    unsigned int size = layer->height * layer->width;
    unsigned int* grid = malloc(size * sizeof(unsigned int));

    for(int i = 0; i < size; ++i) {
        // unsigned char v = layer->cells[i] * 255.0f;

        struct HSL hsl;
        hsl.H = (1.0 - layer->cells[i]) * 240;
        hsl.L = 0.5f;
        hsl.S = 1.0f;

        struct RGB rgb = HSLToRGB(hsl);

        grid[i] = (255 << 24) | (rgb.B << 16) | (rgb.G << 8) | (rgb.R);
    }   

    stbi_write_png(fileName, layer->width, layer->height, 4, grid, GRID_WIDTH*4);
    
    free(grid);
}

double clamp(double d, double min, double max) {
  const double t = d < min ? min : d;
  return t > max ? max : t;
}

void calculate_new_state(StateLayer* newState, StateLayer* oldState, GradientLayer* gradient) {
    unsigned int width = newState->width;
    unsigned int height = newState->height;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            float value = 0.0f;
            GradientCell grad = gradient->cells[x + y*width];

            if((x - 1) >= 0) {
                value += oldState->cells[(x - 1) + y*width] * grad.value[LEFT];
            }

            if((x + 1) < width) {
                value += oldState->cells[(x + 1) + y*width] * grad.value[RIGHT];
            }

            if((y - 1) >= 0) {
                value += oldState->cells[x + (y - 1)*width] * grad.value[UP];
            }

            if((y + 1) < height) {
                value += oldState->cells[x + (y + 1)*width] * grad.value[DOWN];
            }

            newState->cells[x + y*width] = clamp(value, 0.0, 1.0);

            // newState->cells[x + y*width] = sigmoidf(value);
        }
    }
}

float calcl_state_value(StateLayer* state) {
    unsigned int width = state->width;
    unsigned int height = state->height;
    
    float value = 0.0f;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            value += state->cells[x + y*width];
        }
    }

    return value;
}

float calcl_gradient_value(GradientLayer* layer) {  
    unsigned int width = layer->width;
    unsigned int height = layer->height;
    
    float value = 0.0f;

    for(int x = 0; x < width; ++x) {
        for(int y = 0; y < height; ++y) {
            value += layer->cells[x + y*width].value[LEFT];
            value += layer->cells[x + y*width].value[RIGHT];
            value += layer->cells[x + y*width].value[UP];
            value += layer->cells[x + y*width].value[DOWN];
        }
    }

    return value;
}

typedef struct Vector2D {
    int x;
    int y;
} Vector2D;

void trace_path(GradientDirection* dir, GradientLayer* layer, int startX, int startY, int x, int y) {
    unsigned int width = layer->width;
    unsigned int height = layer->height;

    if((x == startY) && (y == startY))
        return;

    if(x < 0)         return;
    if(x >= width)    return;
    if(y < 0)         return;
    if(y >= height)   return;

    int index = x + y*width;
    GradientDirection direction = dir[index];

    if(direction == LEFT) {
        layer->cells[index].value[UP]= 0.0f;
        layer->cells[index].value[RIGHT]= 0.0f;
        layer->cells[index].value[DOWN]= 0.0f;
        layer->cells[index].value[LEFT]= 1.0f;

        for(int i = 0; i < 4; ++i)
            layer->cells[index].value[i] = clamp(layer->cells[index].value[i], 0.0f, 1.0f);

        trace_path(dir, layer, startX, startY, x - 1, y);
    } else if(direction == RIGHT) {
        layer->cells[index].value[UP]= 0.0f;
        layer->cells[index].value[RIGHT]= 1.0f;
        layer->cells[index].value[DOWN]= 0.0f;
        layer->cells[index].value[LEFT]= 0.0f;

        for(int i = 0; i < 4; ++i)
            layer->cells[index].value[i] = clamp(layer->cells[index].value[i], 0.0f, 1.0f);

        trace_path(dir, layer, startX, startY, x + 1, y);
    } else if(direction == UP) {
        layer->cells[index].value[UP]= 1.0f;
        layer->cells[index].value[RIGHT]= 0.0f;
        layer->cells[index].value[DOWN]= 0.0f;
        layer->cells[index].value[LEFT]= 0.0f;

        for(int i = 0; i < 4; ++i)
            layer->cells[index].value[i] = clamp(layer->cells[index].value[i], 0.0f, 1.0f);

        trace_path(dir, layer, startX, startY, x, y - 1);
    } else if(direction == DOWN) {
        layer->cells[index].value[UP]= 0.0f;
        layer->cells[index].value[RIGHT]= 0.0f;
        layer->cells[index].value[DOWN]= 1.0f;
        layer->cells[index].value[LEFT]= 0.0f;

        for(int i = 0; i < 4; ++i)
            layer->cells[index].value[i] = clamp(layer->cells[index].value[i], 0.0f, 1.0f);

        trace_path(dir, layer, startX, startY, x, y + 1);
    }
}

float calc_path(GradientLayer* layer, int startX, int startY, int endX, int endY) {
    unsigned int width = layer->width;
    unsigned int height = layer->height;

    unsigned char visited[GRID_WIDTH][GRID_HEIGHT] = { 0 };
    GradientDirection dir[GRID_WIDTH * GRID_HEIGHT] = { 0 };

    memset(visited, 0, GRID_WIDTH * GRID_HEIGHT);

    Vector2D queue[GRID_WIDTH*GRID_HEIGHT];
    int queueSize = 1;

    queue[0].x = startX;
    queue[0].y = startY;

    while (queueSize > 0) {
        int x = queue[queueSize - 1].x;
        int y = queue[queueSize - 1].y;
        --queueSize;

        if(visited[x][y])
            continue;

        visited[x][y] = 1;

        if((x - 1) >= 0 && !visited[x - 1][y]) {
            queue[queueSize].x = x - 1;
            queue[queueSize].y = y;
            ++queueSize;

            dir[(x - 1) + width*y] = RIGHT;
        }

        if((x + 1) < width && !visited[x + 1][y]) {
            queue[queueSize].x = x + 1;
            queue[queueSize].y = y;
            ++queueSize;
                
            dir[(x + 1) + width*y] = LEFT;
        }

        if((y - 1) >= 0 && !visited[x][y - 1]) {
            queue[queueSize].x = x;
            queue[queueSize].y = y - 1;
            ++queueSize;

            dir[x + width * (y - 1)] = UP;
        }

        if((y + 1) < height && visited[x][y + 1]) {
            queue[queueSize].x = x;
            queue[queueSize].y = y + 1;
            ++queueSize;

            dir[x + width * (y + 1)] = DOWN;
        }
    }
 
    trace_path(dir, layer, startX, startY, endX, endY);
}

typedef struct Model {
    StateLayer* states[2];
    GradientLayer* rules;

    unsigned int iteration;
    unsigned char newIndex;    
} Model;

Model* create_model(unsigned int width, unsigned int height) {
    Model* model = (Model*) malloc(sizeof(Model));

    model->rules = create_gradient_layer(width, height);

    model->states[0] = create_state_layer(width, height);
    model->states[1] = create_state_layer(width, height);

    model->iteration = 0;

    return model;
}

void simulate_model(Model* model) {
    int old = model->iteration % 2;
    int new = (model->iteration + 1) % 2;

    model->newIndex = new;

    calculate_new_state(model->states[new], model->states[old], model->rules);
    ++model->iteration;
}

void set_inputs_model(Model* model, float input[]) {
    for(int i = 0; i < GRID_WIDTH; ++i) {
        model->states[0]->cells[i] = input[i];
        model->states[1]->cells[i] = input[i];
    }
}

float calculate_error(Model* model, float expectedOutput[]) {
    float error = 0.0;

    for(int i = 0; i < GRID_WIDTH; ++i) {
        error += fabs(model->states[model->newIndex]->cells[i + 7*GRID_WIDTH] - expectedOutput[i]);
    }

    return error / (float) GRID_WIDTH;
}

void free_model(Model* model) {
    free_gradient(model->rules);

    free_states(model->states[0]);
    free_states(model->states[1]);

    free(model);
}

int main() {
    srand(time(0));

    Model* model = NULL;

    float input[GRID_WIDTH] = {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    float output[GRID_WIDTH] = {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    int i = 0;
    for(;;++i) {
        model = create_model(GRID_WIDTH, GRID_HEIGHT);


        printf("Simulating model %d\n", i);

        for(int i = 0; i < 16; ++i) {
            set_inputs_model(model, input);
            simulate_model(model);
            set_inputs_model(model, input);
        }

        float error = calculate_error(model, output);

        if(error < 0.05)
            break;

        free_model(model);
    }

    export_gradient_layer_to_png(model->rules, "gradient.png");
    export_state_layer_to_png(model->states[model->newIndex], "state.png");

    printf("Model error %f\n", calculate_error(model, output));

    free_model(model);


    /*
    GradientLayer* layer = create_gradient_layer(GRID_WIDTH, GRID_HEIGHT);

    StateLayer* states[2];
    states[0] = create_state_layer(GRID_WIDTH, GRID_HEIGHT);
    states[1] = create_state_layer(GRID_WIDTH, GRID_HEIGHT);
    
    for(int i = 0; i < GRID_WIDTH * GRID_HEIGHT; ++i) {
        states[0]->cells[i] = (rand() % 255) / 255.0f;
    }

    
    export_gradient_layer_to_png(layer, "gradient.png");

    for(int i = 0;; ++i) {
        int oldS = i % 2;
        int newS = (i + 1) % 2;

        export_gradient_layer_to_png(layer, "gradient.png");

        if(i < 100) {
            for(int i = 0; i < GRID_WIDTH; ++i) {
                states[oldS]->cells[i] = input[i];
                states[newS]->cells[i] = input[i];

                states[oldS]->cells[i + 7*GRID_WIDTH] = output[i];
                states[newS]->cells[i + 7*GRID_WIDTH] = output[i];
            }
        }

        calculate_new_state(states[newS], states[oldS], layer);

        if(i < 100) {
            for(int i = 0; i < GRID_WIDTH; ++i) {
                states[oldS]->cells[i] = input[i];
                states[newS]->cells[i] = input[i];

                states[oldS]->cells[i + 7*GRID_WIDTH] = output[i];
                states[newS]->cells[i + 7*GRID_WIDTH] = output[i];
            }
        }

        export_state_layer_to_png(states[newS], "state.png");

        printf("System value %f, grad value %f\n", calcl_state_value(states[newS]), calcl_gradient_value(layer));

        int c = getchar();
    }
    */

    return 0;
}