include_directories("../src")

include_directories("../3dparty/stb")

FILE(GLOB ALBEDO_EXAMPLE_0_FILES albedo.c)
add_executable(albedo_example_0 ${ALBEDO_EXAMPLE_0_FILES})

target_link_libraries(albedo_example_0 albedo)
target_link_libraries(albedo_example_0 stb_image_write)
target_link_libraries(albedo_example_0 m)