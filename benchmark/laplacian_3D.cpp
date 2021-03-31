#include <vector>
#include <benchmark/benchmark.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>

static void BM_std_vector_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::size_t size2 = size*size;
    std::vector<double> u1(size*size*size), u2(size*size*size);

    for (auto _ : state)
    {
        for(std::size_t k=1; k<size-1; ++k)
        {
            for(std::size_t j=1; j<size-1; ++j)
            {
                for(std::size_t i=1; i<size-1; ++i)
                {
                    u2[i + j*size + k*size2] = (u1[i + j*size + (k-1)*size2] + u1[i + (j-1)*size + k*size2]
                                    + u1[i-1 + j*size + k*size2] - 2*u1[i + j*size + k*size2] + u1[i+1 + j*size + k*size2]
                                              + u1[i + (j+1)*size + k*size2] + u1[i + j*size + (k+1)*size2]
                                    );
                }
            }
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_step_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0, r0, r0)) =
           + xt::view(u1, rm1, r0, r0)
           + xt::view(u1, r0, rm1, r0)
           + xt::view(u1, r0, r0, rm1)
           - 2*xt::view(u1, r0, r0, r0)
           + xt::view(u1, rp1, r0, r0)
           + xt::view(u1, r0, rp1, r0)
           + xt::view(u1, r0, r0, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_without_step_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0, r0, r0)) =
           + xt::view(u1, rm1, r0, r0)
           + xt::view(u1, r0, rm1, r0)
           + xt::view(u1, r0, r0, rm1)
           - 2*xt::view(u1, r0, r0, r0)
           + xt::view(u1, rp1, r0, r0)
           + xt::view(u1, r0, rp1, r0)
           + xt::view(u1, r0, r0, rp1);
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_loop_on_dim_0_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        for (std::size_t i=1; i<size-1; ++i)
        {
            xt::noalias(xt::view(u2, i, r0, r0)) =
            + xt::view(u1, i-1, r0, r0)
            + xt::view(u1, i, rm1, r0)
            + xt::view(u1, i, r0, rm1)
            - 2*xt::view(u1, i, r0, r0)
            + xt::view(u1, i+1, r0, r0)
            + xt::view(u1, i, rp1, r0)
            + xt::view(u1, i, r0, rp1);
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_loop_on_dim_0_and_1_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        for (std::size_t i=1; i<size-1; ++i)
        {
            for (std::size_t j=1; j<size-1; ++j)
            {
                xt::noalias(xt::view(u2, i, j, r0)) =
                + xt::view(u1, i-1, j, r0)
                + xt::view(u1, i, j-1, r0)
                + xt::view(u1, i, j, rm1)
                - 2*xt::view(u1, i, j, r0)
                + xt::view(u1, i+1, j, r0)
                + xt::view(u1, i, j+1, r0)
                + xt::view(u1, i, j, rp1);
            }
        }
    }
    state.SetComplexityN(state.range(0));
}

static void BM_xtensor_with_loop_lap_3D(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 3> u1 = xt::zeros<double>({size, size, size});
    xt::xtensor<double, 3> u2 = xt::zeros<double>({size, size, size});

    auto r0 = xt::range(1, size-1, 1);
    auto rm1 = xt::range(0, size-2, 1);
    auto rp1 = xt::range(2, size, 1);

    for (auto _ : state)
    {
        for (std::size_t i=1; i<size-1; ++i)
        {
            for (std::size_t j=1; j<size-1; ++j)
            {
                for (std::size_t k=1; k<size-1; ++k)
                {
                    u2(i, j, k) =
                    + u1(i-1, j, k)
                    + u1(i, j-1, k)
                    + u1(i, j, k-1)
                    - 2*u1(i, j, k)
                    + u1(i+1, j, k)
                    + u1(i, j+1, k)
                    + u1(i, j, k+1);
                }
            }
        }
    }
    state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_std_vector_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_with_step_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_without_step_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_with_loop_on_dim_0_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_with_loop_on_dim_0_and_1_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
BENCHMARK(BM_xtensor_with_loop_lap_3D)->RangeMultiplier(2)->Ranges({{1<<5, 1<<9}})->Complexity(benchmark::oNCubed);
