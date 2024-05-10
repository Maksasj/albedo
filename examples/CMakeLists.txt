include_directories("../src")

include_directories("../3dparty/stb")

FILE(GLOB ALBEDO_AND_GATE_EXAMPLE_FILES albedo_and_gate.c)
add_executable(albedo_and_gate_example ${ALBEDO_AND_GATE_EXAMPLE_FILES})

target_link_libraries(albedo_and_gate_example albedo)
target_link_libraries(albedo_and_gate_example stb_image_write)
target_link_libraries(albedo_and_gate_example stb_image)
target_link_libraries(albedo_and_gate_example m)

FILE(GLOB ALBEDO_IMAGE_EXAMPLE_FILES albedo_image.c)
add_executable(albedo_image_example ${ALBEDO_IMAGE_EXAMPLE_FILES})

target_link_libraries(albedo_image_example albedo)
target_link_libraries(albedo_image_example stb_image_write)
target_link_libraries(albedo_image_example stb_image)
target_link_libraries(albedo_image_example m)