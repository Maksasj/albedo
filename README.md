# Albedo

<img align="left" src="https://github.com/Maksasj/albedo/blob/master/logo.gif" width="192px">

albedo - neural cellular automata model

> albedo is not a new javascript framework btw !

<br>
<br>
<br>
<br>

## Building
```bash
cmake -B build -G Ninja     
cmake --build build
```

## Example
```c
#include "albedo_visualization.h"
#include "albedo/albedo.h"

#define GRID_WIDTH 8
#define GRID_HEIGHT 8

#define TEST_CASES 4
#define INPUT_COUNT 2
#define OUTPUT_COUNT 1
#define STEPS 50

int main() {
    srand(time(0));

    AlbedoNeuronValue rawInputs[TEST_CASES][INPUT_COUNT] = {
        {{0, 0, 0.0f}, {0, 1, 0.0f}},
        {{0, 0, 1.0f}, {0, 1, 0.0f}},
        {{0, 0, 0.0f}, {0, 1, 1.0f}},
        {{0, 0, 1.0f}, {0, 1, 1.0f}}
    };

    AlbedoNeuronValue rawOutputs[TEST_CASES][OUTPUT_COUNT] = {
        {{7, 7, 0.0f}},
        {{7, 7, 0.0f}},
        {{7, 7, 0.0f}},
        {{7, 7, 1.0f}}
    };

    /* Convert input/output arrays to AlbedoNeuronValue** type */

    AlbedoModel* model = albedo_new_model(GRID_WIDTH, GRID_HEIGHT);

    albedo_genetic_algorithm_training(model, inputs, outputs, TEST_CASES, INPUT_COUNT, OUTPUT_COUNT, 0.004, STEPS);
    albedo_sumup_testing(model, inputs, outputs, TEST_CASES, INPUT_COUNT, OUTPUT_COUNT, STEPS);

    /* free model, input and output data */

    return 0;
}
```

## Dataset
```bash
tar -xvzf mnist_png.tar.gz -C ./mnist_png
```

## License
Albedo is free, open source model. All code in this repository is licensed under
- MIT License ([LICENSE.md](https://github.com/Maksasj/albedo/blob/master/LICENSE.md) or https://opensource.org/license/mit/)
