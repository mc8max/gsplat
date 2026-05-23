#include <metal_stdlib>

using namespace metal;

template <typename scalar_t>
inline void adam_update(
    device scalar_t* param,
    device const scalar_t* param_grad,
    device scalar_t* exp_avg,
    device scalar_t* exp_avg_sq,
    device const uchar* valid,
    constant uint& n,
    constant uint& d,
    constant float& lr,
    constant float& b1,
    constant float& b2,
    constant float& eps,
    uint id
) {
    const uint numel = n * d;
    if (id >= numel) {
        return;
    }

    const uint row = id / d;
    if (valid != nullptr && valid[row] == 0) {
        return;
    }

    const float grad = static_cast<float>(param_grad[id]);
    const float next_exp_avg = b1 * static_cast<float>(exp_avg[id]) + (1.0f - b1) * grad;
    const float next_exp_avg_sq =
        b2 * static_cast<float>(exp_avg_sq[id]) + (1.0f - b2) * grad * grad;
    param[id] = static_cast<scalar_t>(
        static_cast<float>(param[id]) + (-lr * next_exp_avg / (sqrt(next_exp_avg_sq) + eps))
    );
    exp_avg[id] = static_cast<scalar_t>(next_exp_avg);
    exp_avg_sq[id] = static_cast<scalar_t>(next_exp_avg_sq);
}

kernel void adam_kernel_float(
    device float* param [[buffer(0)]],
    device const float* param_grad [[buffer(1)]],
    device float* exp_avg [[buffer(2)]],
    device float* exp_avg_sq [[buffer(3)]],
    device const uchar* valid [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant float& lr [[buffer(7)]],
    constant float& b1 [[buffer(8)]],
    constant float& b2 [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    adam_update<float>(param, param_grad, exp_avg, exp_avg_sq, valid, n, d, lr, b1, b2, eps, id);
}

kernel void adam_kernel_half(
    device half* param [[buffer(0)]],
    device const half* param_grad [[buffer(1)]],
    device half* exp_avg [[buffer(2)]],
    device half* exp_avg_sq [[buffer(3)]],
    device const uchar* valid [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant float& lr [[buffer(7)]],
    constant float& b1 [[buffer(8)]],
    constant float& b2 [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint id [[thread_position_in_grid]]
) {
    adam_update<half>(param, param_grad, exp_avg, exp_avg_sq, valid, n, d, lr, b1, b2, eps, id);
}
