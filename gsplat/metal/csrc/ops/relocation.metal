#include <metal_stdlib>

using namespace metal;

kernel void relocation_kernel(
    device const float* opacities [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const int* ratios [[buffer(2)]],
    device const float* binoms [[buffer(3)]],
    device float* new_opacities [[buffer(4)]],
    device float* new_scales [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    constant uint& n_max [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n) {
        return;
    }

    const int n_idx = ratios[id];
    float denom_sum = 0.0f;
    const float opacity = opacities[id];
    const float new_opacity = 1.0f - pow(1.0f - opacity, 1.0f / static_cast<float>(n_idx));
    new_opacities[id] = new_opacity;

    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= i - 1; ++k) {
            const float bin_coeff = binoms[(i - 1) * n_max + k];
            const float sign = (k & 1) == 0 ? 1.0f : -1.0f;
            const float term = (sign / sqrt(static_cast<float>(k + 1))) * pow(new_opacity, static_cast<float>(k + 1));
            denom_sum += bin_coeff * term;
        }
    }

    const float coeff = opacity / denom_sum;
    const uint base = id * 3u;
    new_scales[base + 0u] = coeff * scales[base + 0u];
    new_scales[base + 1u] = coeff * scales[base + 1u];
    new_scales[base + 2u] = coeff * scales[base + 2u];
}
