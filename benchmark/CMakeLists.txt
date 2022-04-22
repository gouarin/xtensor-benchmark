set(XTENSOR_BENCHMARKS
    # bench_container.cpp
    # axpy.cpp
    laplacian_1D.cpp
    # laplacian_2D.cpp
    # laplacian_3D.cpp
    main.cpp
)
add_executable(bench_xtensor ${XTENSOR_BENCHMARKS})

# message(${gperftools_LIBRARY_DIRS})

# target_link_libraries(bench_xtensor benchmark::benchmark xtensor xtensor::optimize xtensor::use_xsimd Eigen3::Eigen)
# target_link_libraries(bench_xtensor benchmark::benchmark xtensor Eigen3::Eigen)
target_link_libraries(bench_xtensor benchmark::benchmark xtensor Eigen3::Eigen xtensor::use_xsimd )
# target_link_libraries(bench_xtensor benchmark::benchmark xtensor xtensor::optimize)
