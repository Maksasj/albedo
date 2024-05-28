cmake_minimum_required(VERSION 3.21)
project(albedo)

if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(3dparty)
add_subdirectory(src)
add_subdirectory(examples)