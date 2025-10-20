cmake_minimum_required(VERSION 3.16)
project(dvs_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dvs_core
    src/pipeline.cpp
    src/refractory_filter.cpp
    src/baf_filter.cpp
    src/time_surface_filter.cpp
)

target_include_directories(dvs_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(dvs_core PRIVATE -O3 -march=native)

add_executable(dvs_cli src/main.cpp)
target_link_libraries(dvs_cli PRIVATE dvs_core)

target_compile_options(dvs_cli PRIVATE -O3 -march=native)

add_executable(dvs_filters_test tests/test_filters.cpp)
target_link_libraries(dvs_filters_test PRIVATE dvs_core)

target_compile_options(dvs_filters_test PRIVATE -O3 -march=native)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME dvs_filters COMMAND dvs_filters_test)
endif()
