
add_executable(x_square x_square_xtensor.cpp)
target_link_libraries(x_square xtensor::optimize xtensor::use_xsimd Eigen3::Eigen CLI11::CLI11)
# target_compile_options(x_square PRIVATE -g -Wa,-adhln)
# target_compile_options(x_square PRIVATE -g)
