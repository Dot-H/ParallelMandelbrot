#pragma once
#include <cstddef>

#define XINF (-2.5f)
#define XSUP (1.0f)
#define YINF (-1.0f)
#define YSUP (1.0f)

/// \param buffer The RGB24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
void render(std::byte* buffer,
            int width,
            int height,
            std::ptrdiff_t stride,
            int n_iterations = 100);

void render_mt(std::byte* buffer,
               int width,
               int height,
               std::ptrdiff_t stride,
               int n_iterations = 100);



void render_optimized(std::byte* buffer,
                      int width,
                      int height,
                      std::ptrdiff_t stride,
                      int n_iterations = 100);

void render_optimized_mt(std::byte* buffer,
                         int width,
                         int height,
                         std::ptrdiff_t stride,
                         int n_iterations = 100);
