cmake_minimum_required(VERSION 3.11)

project(xtensor-benchmark)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(xtensor REQUIRED)

include(CheckCXXCompilerFlag)

include(addGoogleTest)
include(addGoogleBench)

find_package(Threads)

include(FindPkgConfig)
pkg_check_modules(gpertools libprofiler)

add_compile_options(-g)
add_link_options(-g)
add_subdirectory(benchmark)
