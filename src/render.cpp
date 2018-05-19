#include "render.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>

struct rgb8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

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

static int trace_mandelbroot(float y0, float x0, int n_iterations) {

    int iteration = 0;
    float x = 0.0;
    float y = 0.0;
    float x2 = 0;
    float y2 = 0;

    for (; x2 + y2 < 4.0 && iteration < n_iterations; ++iteration) {
        float xtemp = x2 - y2 + x0;
        y = 2 * x * y + y0;
        x = xtemp;

        x2 = x * x;
        y2 = y * y;
    }

    return iteration;
}

static void color_pixel(rgb8_t *lineptr, int px, float hue) {
    if (hue < 0)
        lineptr[px] = rgb8_t{255, 255, 255};
    else if (hue >= 1)
        lineptr[px] = rgb8_t{0, 0, 0};
    else
        lineptr[px] = heat_lut(hue);
}

static void color_mandelbroot(std::vector<int>& histo, std::byte *buffer,
                              std::ptrdiff_t& stride,
                              std::vector<std::vector<int>>& iter_map) {
    const int n_iterations = histo.size();
    const int width = iter_map.size();
    const int height = iter_map[0].size();

    float total = 0;
    for (int i = 0; i < n_iterations; ++i)
        total += histo[i];


    for (int py = 0; py < height; ++py) {
        rgb8_t *lineptr = reinterpret_cast<rgb8_t *>(buffer);

        for (int px = 0; px < width; ++px) {
            int iterations = iter_map[px][py];
            float hue = 0.0;
            for (int i = 0; i <= iterations; ++i)
                hue += float(histo[i]) / total;

            color_pixel(lineptr, px, hue);
        }

        buffer += stride;
    }
}

void render(std::byte *buffer, int width, int height, std::ptrdiff_t stride,
            int n_iterations) {
    const float yinf = -1;
    const float ysup = 1;
    const float xinf = -2.5;
    const float xsup = 1;

    const float ydist = ysup - yinf; // scale on (-1, 1)
    const float xdist = xsup - xinf; // scale on (-2.5, 1)

    std::vector<int> histo(n_iterations, 0);
    std::vector<std::vector<int>> iter_map(width, std::vector<int>(height, 0));
    std::byte *start_buf = buffer;

    for (int py = 0; py < height; ++py) {
        float y0 = float(py) / float(height - 1) * ydist + yinf; // Scaled y

        for (int px = 0; px < width; ++px) {
            float x0 = float(px) / float(width - 1) * xdist + xinf; // Scaled x
            int iterations = trace_mandelbroot(y0, x0, n_iterations);
            iter_map[px][py] = iterations;
            histo[iterations] += 1;
        }

        buffer += stride;
    }


    color_mandelbroot(histo, start_buf, stride, iter_map);
}

void render_mt(std::byte *buffer, int width, int height, std::ptrdiff_t stride,
               int n_iterations) {
  render(buffer, width, height, stride, n_iterations);
}
