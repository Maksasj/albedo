# albedo 🪬 
albedo - is a simple neural cellular automata framework written in C

<p align="center">
    <img src="https://github.com/Maksasj/albedo/blob/master/logo.gif" alt="project-image">
</p>

> Neural cellular automata learns to draw digit **3**

### Overview

Nowadays there are many models and architectures of neural networks. But more or less they are all based on very similar principles, such as layers of neurons, weights, bias, etc. But what if we try to rethink neural networks into something more organic? What if we combine neural networks and cellular automata?

Most of the cellular machines have very simple rules, such as [Game Of Live](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), [Rule 110](https://en.wikipedia.org/wiki/Rule_110) have no more than 4 rules. Despite the small number of rules, very complex behavior emerges in such systems.

Albedo is a cellular automaton (similar to [Game Of Live](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)) in which each cell is a separate neuron with 9 connections to neighboring neurons (including itself).

> albedo is not a new javascript framework btw !

### Links
1. Live demonstration of a albedo is **coming soon** [maksasj.github.io/albedo]()
2. Source code avaiable at [github.com/Maksasj/albedo](https://github.com/Maksasj/albedo)
3. **blueberry.h** and other C libraries source code [github.com/Maksasj/caifu](https://github.com/Maksasj/caifu)

Cool looking widgets 
<img src="https://img.shields.io/github/stars/Maksasj/albedo" alt="stars">
<img src="https://img.shields.io/github/license/Maksasj/albedo" alt="build">

## Building
1. **Build manually**
  First of all requirements:
    - Cmake (At least version 3.21)
    - Ninja (At least version 1.11.1)
    - C++ compiler (Have test with Clang 15.0.5 and GCC 12.2.0) 

    Firstly lets clone albedo locally(note that you also need to clone all albedo git submodules).

    Secondly lets configure our Cmake configuration with
    ```bash
    cmake -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
    ```

    Finally you can simply build project with cmake 
    ```bash
    cmake --build build/release
    ```

    Now somewhere in build directory you can find all builded examples

## Examples
All examples and demos you can find under [examples](https://github.com/Maksasj/albedo/tree/master/examples) directory

## License
albedo is free, open source model. All code in this repository is licensed under
- MIT License ([LICENSE.md](https://github.com/Maksasj/albedo/blob/master/LICENSE.md) or https://opensource.org/license/mit/)
