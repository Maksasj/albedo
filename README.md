<h1 align="center" id="title">albedo</h1>

<p align="center"><img src="https://github.com/Maksasj/albedo/blob/master/logo.gif" alt="project-image"></p>

Nowadays there are many models and architectures of neural networks. But more or less they are all based on very similar principles, such as layers of neurons, weights, bias, etc. But what if we try to rethink neural networks into something more organic? What if we combine neural networks and cellular automata?

Most of the cellular machines have very simple rules, such as [Game Of Live](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), [Rule 110](https://en.wikipedia.org/wiki/Rule_110) have no more than 4 rules. Despite the small number of rules, very complex behavior emerges in such systems.

Albedo is a cellular automaton (similar to [Game Of Live](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)) in which each cell is a separate neuron with 9 connections to neighboring neurons (including itself).

> albedo is not a new javascript framework btw !

## Table of Contents
- Lore
- Building
- Example
- License

## Building
```bash
cmake -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release && ./build/release/examples/albedo_image_example
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
