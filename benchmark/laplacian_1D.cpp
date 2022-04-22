#include <vector>
#include <benchmark/benchmark.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

static void BM_std_vector_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::vector<double> u1(size), u2(size);

    for (auto _ : state)
    {
        for(std::size_t i=1; i<size-1; ++i)
        {
            u2[i] = u1[i-1] - 2*u1[i] + u1[i+1];
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_eigen_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);

    Eigen::ArrayXd u1(size);                                                                         \
    Eigen::ArrayXd u2(size);                                                                         \
    u1(Eigen::seq(0, size+2)) = 0.0;
    u2(Eigen::seq(0, size+2)) = 0.0;
    for (auto _ : state)
    {
        u2(Eigen::seq(1, size-1)) = u1(Eigen::seq(0, size-2)) - 2.*u1(Eigen::seq(1, size-1)) + u1(Eigen::seq(2, size));
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_step_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0)) = xt::view(u1, rm1)
                                           - 2*xt::view(u1, r0)
                                           + xt::view(u1, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_without_step_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0)) = xt::view(u1, rm1)
                                           - 2*xt::view(u1, r0)
                                           + xt::view(u1, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_loop_lap_1D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        for (std::size_t i=1; i<size-1; ++i)
        {
            u2(i) = u1(i-1) - 2*u1(i) + u1(i+1);
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_std_vector_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_eigen_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xtensor_with_step_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xtensor_without_step_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xtensor_with_loop_lap_1D)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
