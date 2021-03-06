#include <vector>
#include <memory>
#include <benchmark/benchmark.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>

static void BM_std_vector(benchmark::State& state)
{
    std::size_t size = state.range(0);
    std::vector<double> u1(size+2), u2(size+2);

    auto address = std::addressof(u1[0]);
    auto store_address = std::addressof(u2[0]);

    for (auto _ : state)
    {
        for(std::size_t i=1; i< size-1; i+=4)
        {
            auto res = xsimd::load_simd(address + i - 1 , xsimd::unaligned_mode())
                     - xsimd::set_simd<double, double>(2)*xsimd::load_simd(address + i, xsimd::unaligned_mode())
                     + xsimd::load_simd(address + i + 1, xsimd::unaligned_mode());
            xsimd::store_simd(store_address + i, res, xsimd::unaligned_mode());
        }
    }
}

static void BM_svector(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::svector<double> u1(size+2), u2(size+2);

    auto address = std::addressof(u1[0]);
    auto store_address = std::addressof(u2[0]);

    for (auto _ : state)
    {
        for(std::size_t i=1; i< size-1; i+=4)
        {
            auto res = xsimd::load_simd(address + i - 1 , xsimd::unaligned_mode())
                     - xsimd::set_simd<double, double>(2)*xsimd::load_simd(address + i, xsimd::unaligned_mode())
                     + xsimd::load_simd(address + i + 1, xsimd::unaligned_mode());
            xsimd::store_simd(store_address + i, res, xsimd::unaligned_mode());
        }
    }
}

static void BM_uvector(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::uvector<double> u1(size+2), u2(size+2);

    auto address = std::addressof(u1[0]);
    auto store_address = std::addressof(u2[0]);

    for (auto _ : state)
    {
        for(std::size_t i=1; i< size-1; i+=4)
        {
            auto res = xsimd::load_simd(address + i - 1 , xsimd::unaligned_mode())
                     - xsimd::set_simd<double, double>(2)*xsimd::load_simd(address + i, xsimd::unaligned_mode())
                     + xsimd::load_simd(address + i + 1, xsimd::unaligned_mode());
            xsimd::store_simd(store_address + i, res, xsimd::unaligned_mode());
        }
    }
}

static void BM_xtensor(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size + 2});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size + 2});

    auto address = std::addressof(u1.storage()[0]);
    auto store_address = std::addressof(u2.storage()[0]);

    for (auto _ : state)
    {
        for(std::size_t i=1; i< size-1; i+=4)
        {
            auto res = xsimd::load_simd(address + i - 1 , xsimd::unaligned_mode())
                     - xsimd::set_simd<double, double>(2)*xsimd::load_simd(address + i, xsimd::unaligned_mode())
                     + xsimd::load_simd(address + i + 1, xsimd::unaligned_mode());
            xsimd::store_simd(store_address + i, res, xsimd::unaligned_mode());
        }
    }
}

static void BM_xtensor_2(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size + 2});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size + 2});

    auto address = std::addressof(u1.storage()[0]);
    auto store_address = std::addressof(u2.storage()[0]);

    for (auto _ : state)
    {
        for(std::size_t i=1; i< size-1; i+=4)
        {
            auto res = xsimd::load_simd(address + i - 1 , xsimd::unaligned_mode())
                     - xsimd::set_simd<int, double>(2)*xsimd::load_simd(address + i, xsimd::unaligned_mode())
                     + xsimd::load_simd(address + i + 1, xsimd::unaligned_mode());
            xsimd::store_simd(store_address + i, res, xsimd::unaligned_mode());
        }
    }
}

static void BM_xtensor_3(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size + 2});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size + 2});

    auto address = std::addressof(u1.storage()[0]);
    auto store_address = std::addressof(u2.storage()[0]);

    for (auto _ : state)
    {
        for(std::size_t i=1; i< size-1; i+=4)
        {
            auto res = xsimd::load_simd(std::addressof(u1.storage()[i-1]), xsimd::unaligned_mode())
                     - xsimd::set_simd<int, double>(2)*xsimd::load_simd(std::addressof(u1.storage()[i]), xsimd::unaligned_mode())
                     + xsimd::load_simd(std::addressof(u1.storage()[i+1]), xsimd::unaligned_mode());
            xsimd::store_simd(std::addressof(u2.storage()[i]), res, xsimd::unaligned_mode());
        }
    }
}

static void BM_xtensor_4(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size + 2});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size + 2});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    for (auto _ : state)
    {
        xt::noalias(xt::view(u2, r0)) = xt::view(u1, rm1)
                                      - 2*xt::view(u1, r0)
                                      + xt::view(u1, rp1);
    }
}

static void BM_xtensor_5(benchmark::State& state)
{
    std::size_t size = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({size + 2});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({size + 2});

    auto r0 = xt::range(1, size-1);
    auto rm1 = xt::range(0, size-2);
    auto rp1 = xt::range(2, size);

    auto v2 = xt::view(u2, r0);
    auto v1m1 = xt::view(u1, rm1);
    auto v10 = xt::view(u1, r0);
    auto v1p1 = xt::view(u1, rp1);

    for (auto _ : state)
    {
        xt::noalias(v2) = v1m1 - 2*v10 + v1p1;
    }
}


static void BM_xtensor_6(benchmark::State& state)
{
    std::size_t nelem = state.range(0);
    xt::xtensor<double, 1> u1 = xt::zeros<double>({nelem + 2});
    xt::xtensor<double, 1> u2 = xt::zeros<double>({nelem + 2});

    auto r0 = xt::range(1, nelem-1);
    auto rm1 = xt::range(0, nelem-2);
    auto rp1 = xt::range(2, nelem);

    auto v2 = xt::view(u2, r0);
    auto v1m1 = xt::view(u1, rm1);
    auto v10 = xt::view(u1, r0);
    auto v1p1 = xt::view(u1, rp1);
    size_t size = v2.size();
    size_t simd_size = xt_simd::simd_type<double>::size;
    size_t align_begin = xt_simd::get_alignment_offset(u1.data(), size, simd_size);
    size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));
    for (auto _ : state)
    {

        for (size_t i = 0; i < align_begin; ++i)
        {
            v2.data_element(i) = v1m1.data_element(i)
                                 - 2. * v10.data_element(i)
                                 + v1p1.data_element(i);
        }

        for (size_t i = align_begin; i < align_end; i += simd_size)
        {
            auto tmp = v1m1.template load_simd<xt::unaligned_mode, double>(i)
                       - 2. * v10.load_simd<xt::unaligned_mode, double>(i)
                       + v1p1.load_simd<xt::unaligned_mode, double>(i);
            v2.template store_simd<xt::aligned_mode>(i, tmp);
        }

        for (size_t i = align_end; i < size; ++i)
        {
            v2.data_element(i) = v1m1.data_element(i)
                                 - 2. * v10.data_element(i)
                                 + v1p1.data_element(i);
        }
    }
}

std::size_t min = 1<<7;
std::size_t max = 1<<20;

BENCHMARK(BM_std_vector)->RangeMultiplier(2)->Ranges({{min, max}});
BENCHMARK(BM_svector)->RangeMultiplier(2)->Ranges({{min, max}});
// BENCHMARK(BM_uvector)->RangeMultiplier(2)->Ranges({{min, max}});
// BENCHMARK(BM_xtensor)->RangeMultiplier(2)->Ranges({{min, max}});
// BENCHMARK(BM_xtensor_2)->RangeMultiplier(2)->Ranges({{min, max}});
// BENCHMARK(BM_xtensor_3)->RangeMultiplier(2)->Ranges({{min, max}});
BENCHMARK(BM_xtensor_4)->RangeMultiplier(2)->Ranges({{min, max}});
// BENCHMARK(BM_xtensor_5)->RangeMultiplier(2)->Ranges({{min, max}});
BENCHMARK(BM_xtensor_6)->RangeMultiplier(2)->Ranges({{min, max}});
