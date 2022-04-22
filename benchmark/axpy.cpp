#include <vector>
#include <benchmark/benchmark.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

static void BM_std_vector_axpy(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::vector<double> u1(size), u2(size);

    for (auto _ : state)
    {
        for(std::size_t i=0; i<size; ++i)
        {
            u2[i] = 5*u1[i] + 1;
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_eigen_axpy(benchmark::State& state)
{
    std::size_t size = state.range(0);
    Eigen::ArrayXd u1(size), u2(size);

    for (auto _ : state)
    {
        u2 = 5*u1 + 1;
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_axpy(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    for (auto _ : state)
    {
        xt::noalias(u2) = 5*u1 + 1;
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xview1_axpy(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, xt::range(0, size))) = 5*xt::view(u1, xt::range(0, size)) + 1;
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xview2_axpy(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    for (auto _ : state)
    {
        xt::noalias(u2) = 5*xt::view(u1, xt::range(0, size)) + 1;
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xview3_axpy(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size});

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, xt::range(0, size))) = 5*u1 + 1;
    }
    state.SetComplexityN(state.range(0));
}

// BENCHMARK(BM_std_vector_axpy)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_eigen_axpy)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xtensor_axpy)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
BENCHMARK(BM_xview1_axpy)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
// BENCHMARK(BM_xview2_axpy)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);
// BENCHMARK(BM_xview3_axpy)->RangeMultiplier(2)->Ranges({{1<<14, 1<<20}})->Complexity(benchmark::oN);

