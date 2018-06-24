#include "render.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <smmintrin.h>
#include <vector>

struct rgb8_t {
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
};

struct niter {
    unsigned short niter;
    unsigned char _;
}__attribute__((packed));

rgb8_t heat_lut(float x) {
    assert(0 <= x && x <= 1);
    float x0 = 1.f / 4.f;
    float x1 = 2.f / 4.f;
    float x2 = 3.f / 4.f;

    if (x < x0) {
        auto g = static_cast<std::uint8_t>(x / x0 * 255);
        return rgb8_t{0, g, 255};
    } else if (x < x1) {
        auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
        return rgb8_t{0, 255, b};
    } else if (x < x2) {
        auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
        return rgb8_t{r, 255, 0};
    } else {
        auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
        return rgb8_t{255, b, 0};
    }
}

template <typename H>
static void color_mandelbrot(H histo, int n_iterations, std::byte *buffer,
        std::ptrdiff_t& stride, int width, int height) {
    int total = 0;
#pragma omp simd
    for (int i = 0; i < n_iterations; ++i)
        total += histo[i];

    float *hue_map = new float[n_iterations + 1];
    rgb8_t *lut_map = new rgb8_t[n_iterations + 1];

    hue_map[0] = float(histo[0]) / float(total);
    lut_map[0] = heat_lut(hue_map[0]);
    for (int i = 1; i < n_iterations - 1; ++i) {
        hue_map[i] = hue_map[i - 1] + (float(histo[i]) / float(total));
        lut_map[i] = heat_lut(hue_map[i]);
    }
    lut_map[n_iterations - 1] = heat_lut(1.0);
    lut_map[n_iterations] = rgb8_t{0, 0, 0};

    for (int py = 0; py < height / 2; ++py) {
        rgb8_t *lineptr = reinterpret_cast<rgb8_t *>(buffer);
        struct niter *iter_map = reinterpret_cast<struct niter *>(buffer);

#pragma omp simd
        for (int px = 0; px < width; ++px)
            lineptr[px] = lut_map[iter_map[px].niter];

        buffer += stride;
    }

    buffer -= stride;
    std::byte *sym_buffer = buffer;
    for (int py = 0; py < height / 2; ++py) {
        rgb8_t *lineptrsym = reinterpret_cast<rgb8_t *>(sym_buffer);
        rgb8_t *lineptr = reinterpret_cast<rgb8_t *>(buffer);
        for (int px = 0; px < width; ++px) {
            lineptrsym[px] = lineptr[px];
        }

        buffer -= stride;
        sym_buffer += stride;
    }

    delete[] hue_map;
    delete[] lut_map;
}

template <typename H>
static void color_mandelbrot_mt(H histo, int n_iterations, std::byte *buffer,
        std::ptrdiff_t& stride, int width, int height) {
    int total = 0;
#pragma omp simd
    for (int i = 0; i < n_iterations; ++i)
        total += histo[i];

    float *hue_map = new float[n_iterations + 1];
    rgb8_t *lut_map = new rgb8_t[n_iterations + 1];

    hue_map[0] = float(histo[0]) / float(total);
    lut_map[0] = heat_lut(hue_map[0]);
    for (int i = 1; i < n_iterations - 1; ++i) {
        hue_map[i] = hue_map[i - 1] + (float(histo[i]) / float(total));
        lut_map[i] = heat_lut(hue_map[i]);
    }
    lut_map[n_iterations - 1] = heat_lut(1.0);
    lut_map[n_iterations] = rgb8_t{0, 0, 0};

#pragma omp parallel for schedule(dynamic)
    for (int py = 0; py < height / 2; ++py) {
        rgb8_t *lineptr = reinterpret_cast<rgb8_t *>(buffer + py * stride);
        struct niter *iter_map = reinterpret_cast<struct niter *>(buffer + py * stride);

#pragma omp simd
        for (int px = 0; px < width; ++px)
            lineptr[px] = lut_map[iter_map[px].niter];
    }

    buffer += stride * (height / 2 - 1);
    std::byte *sym_buffer = buffer;

#pragma omp parallel for schedule(dynamic)
    for (int py = 0; py < height / 2; ++py) {
        rgb8_t *lineptrsym = reinterpret_cast<rgb8_t *>(sym_buffer + stride * py);
        rgb8_t *lineptr = reinterpret_cast<rgb8_t *>(buffer - stride * py);
        for (int px = 0; px < width; ++px) {
            lineptrsym[px] = lineptr[px];
        }
    }

    delete[] hue_map;
    delete[] lut_map;
}

// Computes q = (x - 0.25)^2 + y^2
// Returns q(q + (x - 0.25)) < 0.25y^2
static bool cardioid(__m256 &x, __m256 &y) {
    static float quarter_cst = 0.25f;
    static float hex_cst = 0.0625f;
    static float unit_cst = 1.0f;
    static __m256 quarter = _mm256_broadcast_ss(&quarter_cst);
    static __m256 hex = _mm256_broadcast_ss(&hex_cst);
    static __m256 unit = _mm256_broadcast_ss(&unit_cst);

    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 xs2 = _mm256_add_ps(x, unit);
    xs2 = _mm256_mul_ps(xs2, xs2);
    xs2 = _mm256_add_ps(xs2, y2);
    __m256 cmp = _mm256_cmp_ps(xs2, hex, _CMP_LT_OQ);
    if (_mm256_movemask_ps(cmp) == 0xff)
        return true;

    __m256 tmp = _mm256_sub_ps(x, quarter);
    __m256 tmp2 = _mm256_mul_ps(tmp, tmp);
    __m256 q = _mm256_add_ps(tmp2, y2);

    __m256 tst1 = _mm256_mul_ps(q, _mm256_add_ps(q, tmp));
    __m256 tst2 = _mm256_mul_ps(quarter, y2);
    cmp = _mm256_cmp_ps(tst1, tst2, _CMP_LT_OQ);

    return _mm256_movemask_ps(cmp) == 0xff;
}

void render(std::byte *buffer, int width, int height, std::ptrdiff_t stride,
        int n_iterations) {
    const float ydiff = YSUP - YINF; // scale on (-1, 1)
    const float xdiff = XSUP - XINF; // scale on (-2.5, 1)

    const float dx = xdiff / float(width - 1);
    const float dy = ydiff / float(height - 1);

    int *histo = new int[n_iterations + 1];
#pragma omp simd
    for (int i = 0; i < n_iterations; ++i)
        histo[i] = 0;

    std::byte *start_buf = buffer;

    // round up width to next multiple of 8
    int roundedWidth = (width + 7) & ~7UL;

    float constants[] = {dx, dy, XINF, YINF, 1.0f, 4.0f, 8.0f};
    __m256 dx256 = _mm256_broadcast_ss(constants);   // all dx
    __m256 dy256 = _mm256_broadcast_ss(constants+1); // all dy
    __m256 xinf256 = _mm256_broadcast_ss(constants+2); // all x1
    __m256 yinf256 = _mm256_broadcast_ss(constants+3); // all y1
    __m256 incrtor = _mm256_broadcast_ss(constants+4); // all 1's (iter increments)
    __m256 cmptor = _mm256_broadcast_ss(constants+5); // all 4's (comparisons)
    __m256 iterator = _mm256_broadcast_ss(constants+6);

    // Zero out j counter (dx256 is just a dummy)
    __m256 jcnt = _mm256_xor_ps(dx256,dx256);

    // used to reset the i position when j increases
    float incr[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    for (int j = 0; j < height / 2; j+=1)
    {
        struct niter *iter_map = reinterpret_cast<struct niter *>(buffer);
        __m256 icnt  = _mm256_load_ps(incr);  // i counter set to 0,1,2,..,7
        for (int i = 0; i < roundedWidth; i+=8)
        {
            int top = (i+7) < width? 8: width&7;

            __m256 x0 = _mm256_mul_ps(icnt, dx256); // x0 = (i+k)*dx
            x0 = _mm256_add_ps(x0, xinf256);        // x0 = x1+(i+k)*dx
            __m256 y0 = _mm256_mul_ps(jcnt, dy256); // y0 = j*dy
            y0 = _mm256_add_ps(y0, yinf256);        // y0 = y1+j*dy
            __m256 itercntor = _mm256_xor_ps(dx256,dx256); // zero out iteration counter
            __m256 xi = itercntor, yi = itercntor; // set initial xi=0, yi=0

            unsigned int test = 0;
            int iter = 0;
            if (cardioid(x0, y0)) {
                for (int k = 0; k < top; ++k)
                    iter_map[i + k].niter = n_iterations;

                // next i position - increment each slot by 8
                icnt = _mm256_add_ps(icnt, iterator);
                continue;
            }

            do
            {
                __m256 xi2 = _mm256_mul_ps(xi,xi); // xi*xi
                __m256 yi2 = _mm256_mul_ps(yi,yi); // yi*yi
                __m256 xyi2 = _mm256_add_ps(xi2,yi2); // xi*xi+yi*yi

                // xi*xi+yi*yi < 4 in each slot
                xyi2 = _mm256_cmp_ps(xyi2,cmptor, _CMP_LT_OQ);
                // now xyi2 has all 1s in the non overflowed locations
                test = _mm256_movemask_ps(xyi2) & 255; // lower 8 bits are comparisons
                if (!test)
                    break;

                xyi2 = _mm256_and_ps(xyi2,incrtor);
                // get 1.0f or 0.0f in each field as counters
                // counters for each pixel iteration
                itercntor = _mm256_add_ps(itercntor,xyi2);

                xyi2 = _mm256_mul_ps(xi,yi); // xi*yi
                xi = _mm256_sub_ps(xi2,yi2); // xi*xi-yi*yi
                xi = _mm256_add_ps(xi,x0); // xi <- xi*xi-yi*yi+x0 done!
                yi = _mm256_add_ps(xyi2,xyi2); // 2*xi*yi
                yi = _mm256_add_ps(yi,y0); // yi <- 2*xi*yi+y0

                ++iter;
            } while (iter < n_iterations);

            for (int k = 0; k < top; ++k) {
                int iterations = itercntor[k];
                iter_map[i + k].niter = iterations;
                histo[iterations] += 2;
            }

            // next i position - increment each slot by 8
            icnt = _mm256_add_ps(icnt, iterator);
        }

        jcnt = _mm256_add_ps(jcnt, incrtor); // increment j counter
        buffer += stride;
    }

    color_mandelbrot(histo, n_iterations, start_buf, stride, width, height);
}

void addVects(std::vector<int> &out, const std::vector<int> &in) {
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] += in[i];
}

#pragma omp declare reduction(AddHisto: std::vector<int>: \
        addVects(omp_out, omp_in)) \
        initializer(omp_priv = omp_orig)

void render_mt(std::byte *buffer, int width, int height, std::ptrdiff_t stride,
        int n_iterations) {
    const float ydiff = YSUP - YINF; // scale on (-1, 1)
    const float xdiff = XSUP - XINF; // scale on (-2.5, 1)

    const float dx = xdiff / float(width - 1);
    const float dy = ydiff / float(height - 1);

    // round up width to next multiple of 8
    int roundedWidth = (width + 7) & ~7UL;

    float constants[] = {dx, dy, XINF, YINF, 1.0f, 4.0f, 8.0f};
    __m256 dx256 = _mm256_broadcast_ss(constants);   // all dx
    __m256 dy256 = _mm256_broadcast_ss(constants+1); // all dy
    __m256 xinf256 = _mm256_broadcast_ss(constants+2); // all x1
    __m256 yinf256 = _mm256_broadcast_ss(constants+3); // all y1
    __m256 incrtor = _mm256_broadcast_ss(constants+4); // all 1's (iter increments)
    __m256 cmptor = _mm256_broadcast_ss(constants+5); // all 4's (comparisons)
    __m256 iterator = _mm256_broadcast_ss(constants+6);


    // used to reset the i position when j increases
    float incr[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    std::vector<int> histo(n_iterations + 1, 0);
    std::byte *start_buf = buffer;

#pragma omp parallel for schedule(dynamic) reduction(AddHisto:histo)
    for (int j = 0; j < height / 2; j+=1)
    {
        float toto = j;
        // Zero out j counter (dx256 is just a dummy)
        __m256 jcnt = _mm256_broadcast_ss(&toto);

        struct niter *iter_map =
            reinterpret_cast<struct niter *>(buffer + j * stride);
        __m256 icnt  = _mm256_load_ps(incr);  // i counter set to 0,1,2,..,7
        for (int i = 0; i < roundedWidth; i+=8)
        {
            int top = (i+7) < width? 8: width&7;

            __m256 x0 = _mm256_mul_ps(icnt, dx256); // x0 = (i+k)*dx
            x0 = _mm256_add_ps(x0, xinf256);        // x0 = x1+(i+k)*dx
            __m256 y0 = _mm256_mul_ps(jcnt, dy256); // y0 = j*dy
            y0 = _mm256_add_ps(y0, yinf256);        // y0 = y1+j*dy
            __m256 itercntor = _mm256_xor_ps(dx256,dx256); // zero out iteration counter
            __m256 xi = itercntor, yi = itercntor; // set initial xi=0, yi=0

            unsigned int test = 0;
            int iter = 0;
            if (cardioid(x0, y0)) {
                for (int k = 0; k < top; ++k)
                    iter_map[i + k].niter = n_iterations;

                // next i position - increment each slot by 8
                icnt = _mm256_add_ps(icnt, iterator);
                continue;
            }

            do
            {
                __m256 xi2 = _mm256_mul_ps(xi,xi); // xi*xi
                __m256 yi2 = _mm256_mul_ps(yi,yi); // yi*yi
                __m256 xyi2 = _mm256_add_ps(xi2,yi2); // xi*xi+yi*yi

                // xi*xi+yi*yi < 4 in each slot
                xyi2 = _mm256_cmp_ps(xyi2,cmptor, _CMP_LT_OQ);
                // now xyi2 has all 1s in the non overflowed locations
                test = _mm256_movemask_ps(xyi2) & 255; // lower 8 bits are comparisons
                if (!test)
                    break;

                xyi2 = _mm256_and_ps(xyi2,incrtor);
                // get 1.0f or 0.0f in each field as counters
                // counters for each pixel iteration
                itercntor = _mm256_add_ps(itercntor,xyi2);

                xyi2 = _mm256_mul_ps(xi,yi); // xi*yi
                xi = _mm256_sub_ps(xi2,yi2); // xi*xi-yi*yi
                xi = _mm256_add_ps(xi,x0); // xi <- xi*xi-yi*yi+x0 done!
                yi = _mm256_add_ps(xyi2,xyi2); // 2*xi*yi
                yi = _mm256_add_ps(yi,y0); // yi <- 2*xi*yi+y0

                ++iter;
            } while (iter < n_iterations);

            for (int k = 0; k < top; ++k) {
                int iterations = itercntor[k];
                iter_map[i + k].niter = iterations;
                histo[iterations] += 2;
            }

            // next i position - increment each slot by 8
            icnt = _mm256_add_ps(icnt, iterator);
        }

        jcnt = _mm256_add_ps(jcnt, incrtor); // increment j counter
    }

    color_mandelbrot_mt(histo, n_iterations, start_buf, stride, width, height);
}
