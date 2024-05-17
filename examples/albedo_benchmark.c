#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "albedo_visualization.h"
#include "albedo/albedo.h"

#include "cherry/cherry.h"

#define GRID_WIDTH 512
#define GRID_HEIGHT 512
#define STEPS 25

#define SAMPLES 50

void benchmark_model(AlbedoModel* model) {
    for(int i = 0; i < SAMPLES; ++i) {
        CHERRY_START_RECORD(TIMER);
            albedo_simulate_model_steps(model, STEPS);
        CHERRY_STOP_RECORD(TIMER);
        
        printf("Sample %d, elapsed time %f\n", i, CHERRY_GET_ELAPSED_TIME(TIMER));
    }
}

int main() {
    srand(time(0));

    AlbedoModel* model = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);

    benchmark_model(model);

    albedo_free_model(model);

    return 0;
}