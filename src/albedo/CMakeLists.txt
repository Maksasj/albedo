include_directories("../")
include_directories("../../3dparty/caifu")
include_directories("./")

FILE(GLOB_RECURSE ALBEDO_SRC_FILES *.c)
add_library(albedo STATIC ${ALBEDO_SRC_FILES})