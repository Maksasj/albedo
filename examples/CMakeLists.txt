include_directories("../src")

include_directories("../3dparty/caifu")
include_directories("../3dparty/caifu/lemon")

include_directories("../3dparty/stb")

# FILE(GLOB ALBEDO_AND_GATE_EXAMPLE_FILES albedo_and_gate.c)
# add_executable(albedo_and_gate_example ${ALBEDO_AND_GATE_EXAMPLE_FILES})
# target_link_libraries(albedo_and_gate_example albedo)
# target_link_libraries(albedo_and_gate_example stb_image_write)
# target_link_libraries(albedo_and_gate_example stb_image)
# target_link_libraries(albedo_and_gate_example m)
# 
# FILE(GLOB ALBEDO_XNOR_GATE_EXAMPLE_FILES albedo_xnor_gate.c)
# add_executable(albedo_xnor_gate_example ${ALBEDO_XNOR_GATE_EXAMPLE_FILES})
# target_link_libraries(albedo_xnor_gate_example albedo)
# target_link_libraries(albedo_xnor_gate_example stb_image_write)
# target_link_libraries(albedo_xnor_gate_example stb_image)
# target_link_libraries(albedo_xnor_gate_example m)
# 
# FILE(GLOB ALBEDO_NOT_EXAMPLE_FILES albedo_not_gate.c)
# add_executable(albedo_not_gate_example ${ALBEDO_NOT_EXAMPLE_FILES})
# target_link_libraries(albedo_not_gate_example albedo)
# target_link_libraries(albedo_not_gate_example stb_image_write)
# target_link_libraries(albedo_not_gate_example stb_image)
# target_link_libraries(albedo_not_gate_example m)
# 
FILE(GLOB ALBEDO_IMAGE_EXAMPLE_FILES albedo_image.c)
add_executable(albedo_image_example ${ALBEDO_IMAGE_EXAMPLE_FILES})
target_link_libraries(albedo_image_example albedo)
target_link_libraries(albedo_image_example stb_image_write)
target_link_libraries(albedo_image_example stb_image)
target_link_libraries(albedo_image_example m)
# 
# FILE(GLOB ALBEDO_BENCHMARK_EXAMPLE_FILES albedo_benchmark.c)
# add_executable(albedo_benchmark ${ALBEDO_BENCHMARK_EXAMPLE_FILES})
# target_link_libraries(albedo_benchmark albedo)
# target_link_libraries(albedo_benchmark stb_image_write)
# target_link_libraries(albedo_benchmark stb_image)
# target_link_libraries(albedo_benchmark m)

# FILE(GLOB PEACH_DEMO_EXAMPLE_FILES peach_demo.c)
# add_executable(peach_demo ${PEACH_DEMO_EXAMPLE_FILES})
# target_link_libraries(peach_demo m)
# 
# FILE(GLOB BLUEBERRY_DEMO_EXAMPLE_FILES blueberry_demo.c)
# add_executable(blueberry_demo ${BLUEBERRY_DEMO_EXAMPLE_FILES})
# target_link_libraries(blueberry_demo m)