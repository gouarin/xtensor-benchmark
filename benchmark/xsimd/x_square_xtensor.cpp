#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include <Eigen/Dense>

struct fake_iterator
{
};

struct fake_allocator
{
};

template<class T>
struct my_container
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using const_reference = const value_type&;
    using reference = value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using difference_type = std::ptrdiff_t;

    using iterator = fake_iterator;
    using const_iterator = fake_iterator;
    using reverse_iterator = fake_iterator;
    using const_reverse_iterator = fake_iterator;

    using allocator_type = fake_allocator;

    my_container() = default;

    my_container(std::size_t size, double value)
    {
        void *tmp;
        posix_memalign(&tmp, 32, sizeof(T) * size);
        m_data = reinterpret_cast<T*>(tmp);
    }

    const_pointer data() const
    {
        return m_data;
    }

    pointer data()
    {
        return m_data;
    }

    const_reference operator[](std::size_t i) const
    {
        return m_data[i];
    }

    reference operator[](std::size_t i)
    {
        return m_data[i];
    }

    void resize(std::size_t size)
    {
        // free(m_data);
        void *tmp;
        posix_memalign(&tmp, 32, sizeof(T) * size);
        m_data = reinterpret_cast<T*>(tmp);
    }
private:
    T *m_data;
};

// #define XTENSOR_DEFAULT_DATA_CONTAINER(T, A) my_container<T>

#include <xsimd/xsimd.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

#include "CLI/CLI.hpp"

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

template<class CT>
void run(std::size_t size, std::size_t max_it, const CT& src, CT& dst)
{
    const std::size_t simd_size = 4;
    std::size_t align_begin = xsimd::get_alignment_offset(&src[0], size, simd_size);
    std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));
    std::cout << align_begin << " " << align_end << " " << size << std::endl;
    tic();
    for (size_t max_iteration = max_it; max_iteration > 0; --max_iteration)
    {
        for (std::size_t i = align_begin; i < align_end; i+=4)
        {
            auto a1 = _mm256_load_pd(&src[i]);
            auto a2 = _mm256_load_pd(&src[i]);
            auto res = _mm256_mul_pd(a1, a2);
            _mm256_store_pd(&dst[i], res);
        }
    }
    std::cout << toc() << std::endl;
}

template<class CT>
void run_3(std::size_t size, std::size_t max_it, const CT& src, CT& dst)
{
    const std::size_t simd_size = 4;
    std::size_t align_begin = xsimd::get_alignment_offset(&src[0], size, simd_size);
    std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));
    std::cout << align_begin << " " << align_end << " " << size << std::endl;
    tic();
    for (size_t max_iteration = max_it; max_iteration > 0; --max_iteration)
    {
        for (std::size_t i = align_begin; i < align_end; i+=4)
        {
            auto a1 = _mm256_load_pd(std::addressof(src.storage()[i]));
            auto a2 = _mm256_load_pd(std::addressof(src.storage()[i]));
            auto res = _mm256_mul_pd(a1, a2);
            _mm256_store_pd(&(dst.storage()[i]), res);
        }
    }
    std::cout << toc() << std::endl;
}

template<class CT>
void run_lazy(std::size_t size, std::size_t max_it, const CT& src, CT& dst)
{
    auto expr = src*src;
    tic();
    for (size_t max_iteration = max_it; max_iteration > 0; --max_iteration)
    {
        // if (src.shape().size() != dst.shape().size() || !std::equal(std::begin(src.shape()), std::end(src.shape()), std::begin(dst.shape())))
        // {
        //     dst.resize(src.shape());
        // }
        // xt::noalias(dst) = src*src;
        xt::noalias(dst) = expr;
    }
    std::cout << toc() << std::endl;
}

template<class CT>
void run_lazy_view(std::size_t size, std::size_t max_it, const CT& src, CT& dst)
{
    auto expr = xt::view(src, xt::all())*xt::view(src, xt::all());
    tic();
    for (size_t max_iteration = max_it; max_iteration > 0; --max_iteration)
    {
        // if (src.shape().size() != dst.shape().size() || !std::equal(std::begin(src.shape()), std::end(src.shape()), std::begin(dst.shape())))
        // {
        //     dst.resize(src.shape());
        // }
        // xt::noalias(dst) = src*src;
        xt::noalias(dst) = expr;
        // xt::noalias(xt::view(dst, xt::all())) = expr;
    }
    std::cout << toc() << std::endl;
}

template<class CT>
void run_simd(std::size_t size, std::size_t max_it, const CT& src, CT& dst)
{
    const std::size_t simd_size = 4;
    std::size_t align_begin = xsimd::get_alignment_offset(&src[0], size, simd_size);
    std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));
    std::cout << align_begin << " " << align_end << " " << size << std::endl;
    tic();
    for (size_t max_iteration = max_it; max_iteration > 0; --max_iteration)
    {
        for (std::size_t i = align_begin; i < align_end; i += simd_size)
        {
            // auto a1 = src.template load_simd<xsimd::aligned_mode, double>(i);
            // auto a2 = src.template load_simd<xsimd::aligned_mode, double>(i);
            dst.template store_simd<xsimd::unaligned_mode>(i, src.template load_simd<xsimd::unaligned_mode, double>(i)*src.template load_simd<xsimd::unaligned_mode, double>(i));
        }
    }
    std::cout << toc() << std::endl;
}

template<class CT>
void run_2(std::size_t size, std::size_t max_it, const CT& src, CT& dst)
{
    const std::size_t simd_size = 4;
    std::size_t align_begin = xsimd::get_alignment_offset(&src[0], size, simd_size);
    std::size_t align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));
    std::cout << align_begin << " " << align_end << " " << size << std::endl;

    tic();
    auto add1 = std::addressof(src[0]);
    auto add2 = std::addressof(dst[0]);

    for (size_t max_iteration = max_it; max_iteration > 0; --max_iteration)
    {
        for (std::size_t i = align_begin; i < align_end; i+=4)
        {
            auto a1 = _mm256_load_pd(add1 + i);
            auto a2 = _mm256_load_pd(add1 + i);
            auto res = _mm256_mul_pd(a1, a2);
            _mm256_store_pd(add2 + i, res);
        }
    }
    std::cout << toc() << std::endl;
}

int main(int argc, char **argv)
{
    CLI::App app{"Y = X**2 performance"};

    // Define options
    std::size_t size = 10;
    app.add_option("--size", size, "Container size");
    std::size_t max_it = 10;
    app.add_option("--maxit", max_it, "Number of repetitions");
    std::string ct = "malloc";
    app.add_option("--ct", ct, "Container type");

    CLI11_PARSE(app, argc, argv);

    if (ct == "malloc")
    {
        // double *src = (double *)std::malloc((size) * sizeof(double));
        // double *dst = (double *)std::malloc((size) * sizeof(double));
        void *src1, *dst1;
        posix_memalign(&src1, 32, sizeof(double) * size);
        posix_memalign(&dst1, 32, sizeof(double) * size);
        auto src = reinterpret_cast<double*>(src1);
        auto dst = reinterpret_cast<double*>(dst1);
        for(std::size_t i=0; i<size; ++i)
        {
            src[i] = 0.;
            dst[i] = 0.;
        }
        std::cout << src << " " << dst << std::endl;
        run(size, max_it, src, dst);
    }
    else if (ct == "xtensor")
    {
        xt::xtensor<double, 1> src({size}, 0);
        xt::xtensor<double, 1> dst({size}, 0);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run_3(size, max_it, src, dst);
    }
    else if (ct == "xtensor_lazy")
    {
        xt::xtensor<double, 1> src({size}, 0);
        xt::xtensor<double, 1> dst({size}, 0);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run_lazy(size, max_it, src, dst);
    }
    else if (ct == "xtensor_lazy_view")
    {
        xt::xtensor<double, 1> src({size}, 0);
        xt::xtensor<double, 1> dst({size}, 0);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run_lazy_view(size, max_it, src, dst);
    }
    else if (ct == "xtensor_simd")
    {
        xt::xtensor<double, 1> src({size}, 0);
        xt::xtensor<double, 1> dst({size}, 0);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run_simd(size, max_it, src, dst);
    }
    else if (ct == "xtensor_adapt")
    {
        void *src1, *dst1;
        posix_memalign(&src1, 32, sizeof(double) * size);
        posix_memalign(&dst1, 32, sizeof(double) * size);
        auto src2 = reinterpret_cast<double*>(src1);
        auto dst2 = reinterpret_cast<double*>(dst1);
        std::vector<std::size_t> shape = { size };
        auto src = xt::adapt(src2, size, xt::no_ownership(), shape);
        auto dst = xt::adapt(dst2, size, xt::no_ownership(), shape);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run(size, max_it, src, dst);
    }
    else if (ct == "eigen")
    {
        Eigen::ArrayXd src(size);
        Eigen::ArrayXd dst(size);
        src.fill(0.);
        dst.fill(0.);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run(size, max_it, src, dst);
    }
    else if (ct == "vector")
    {
        std::vector<double> src(size, 0);
        std::vector<double> dst(size, 0);
        std::cout << src.data() << " " << dst.data() << std::endl;
        run(size, max_it, src, dst);
    }
    return 0;
}