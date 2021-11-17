cmake_minimum_required(VERSION 3.11)

project(xtensor-benchmark)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(xtensor REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CLI11 CONFIG REQUIRED)
# find_package(samurai REQUIRED)

include(CheckCXXCompilerFlag)

include(addGoogleTest)
include(addGoogleBench)

find_package(Threads)

add_subdirectory(benchmark)
# add_subdirectory(test)