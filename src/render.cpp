#include "render.hpp"
#include <cassert>
#include <cstdint>

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

static int trace_mandelbroot(float y0, int pix_x, int n_iterations) {
    float x0 = float(pix_x); // FIXME SCALE

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

void render(std::byte *buffer, int width, int height, std::ptrdiff_t stride,
            int n_iterations) {

    for (int py = 0; py < height; ++py) {
        rgb8_t *lineptr = reinterpret_cast<rgb8_t *>(buffer);

        float y0 = 0.0; // FIXME SCALE

        for (int px = 0; px < width; ++px) {
            trace_mandelbroot(y0, px, n_iterations);

            lineptr[px] = heat_lut((px * px + py * py)
                            / float(width * width + height * height));
        }

        /* FIXME Color */
        buffer += stride;
    }
}

void render_mt(std::byte *buffer, int width, int height, std::ptrdiff_t stride,
               int n_iterations) {
  render(buffer, width, height, stride, n_iterations);
}
