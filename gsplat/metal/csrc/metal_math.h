// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <metal_stdlib>

using namespace metal;

// Build a diagonal matrix from per-axis scale factors.
inline float3x3 make_diag(float3 d) {
    return float3x3(
        float3(d.x, 0.0f, 0.0f),
        float3(0.0f, d.y, 0.0f),
        float3(0.0f, 0.0f, d.z)
    );
}

// Load the gradient of a symmetric 3x3 matrix from either packed upper-triangular
// storage (6 floats) or dense row-major storage (9 floats). Off-diagonal packed
// entries are halved so the reconstructed matrix contributes the correct shared
// gradient to both symmetric positions.
inline float3x3 load_symmetric_grad(device const float* src, uint index, uint triu) {
    if (triu != 0u) {
        device const float* base = src + index * 6;
        float xy = base[1] * 0.5f;
        float xz = base[2] * 0.5f;
        float yz = base[4] * 0.5f;
        return float3x3(
            float3(base[0], xy, xz),
            float3(xy, base[3], yz),
            float3(xz, yz, base[5])
        );
    }

    device const float* base = src + index * 9;
    return float3x3(
        float3(base[0], base[3], base[6]),
        float3(base[1], base[4], base[7]),
        float3(base[2], base[5], base[8])
    );
}

// Store a symmetric 3x3 matrix using either packed upper-triangular layout or
// dense row-major tensor layout expected by the PyTorch-facing bridge.
inline void store_symmetric_matrix(device float* out, uint index, float3x3 M, uint triu) {
    if (triu != 0u) {
        device float* dst = out + index * 6;
        dst[0] = M[0][0];
        dst[1] = M[1][0];
        dst[2] = M[2][0];
        dst[3] = M[1][1];
        dst[4] = M[2][1];
        dst[5] = M[2][2];
        return;
    }

    device float* dst = out + index * 9;
    dst[0] = M[0][0];
    dst[1] = M[1][0];
    dst[2] = M[2][0];
    dst[3] = M[0][1];
    dst[4] = M[1][1];
    dst[5] = M[2][1];
    dst[6] = M[0][2];
    dst[7] = M[1][2];
    dst[8] = M[2][2];
}


// Convert a quaternion in (w, x, y, z) order into a normalized rotation matrix.
inline float3x3 quat_to_rotmat(float4 quat) {
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;

    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return float3x3(
        float3(1.0f - 2.0f * (y2 + z2), 2.0f * (xy + wz), 2.0f * (xz - wy)),
        float3(2.0f * (xy - wz), 1.0f - 2.0f * (x2 + z2), 2.0f * (yz + wx)),
        float3(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (x2 + y2))
    );
}

// Vector-Jacobian product for quat_to_rotmat with respect to the unnormalized
// input quaternion. The output gradient v_R is expressed in Metal's column-major
// float3x3 convention.
inline float4 quat_to_rotmat_vjp(float4 quat, float3x3 v_R) {
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;

    float4 v_quat_n = float4(
        2.0f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
                z * (v_R[0][1] - v_R[1][0])),
        2.0f *
            (-2.0f * x * (v_R[1][1] + v_R[2][2]) +
             y * (v_R[0][1] + v_R[1][0]) +
             z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.0f * (x * (v_R[0][1] + v_R[1][0]) -
                2.0f * y * (v_R[0][0] + v_R[2][2]) +
                z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.0f * (x * (v_R[0][2] + v_R[2][0]) +
                y * (v_R[1][2] + v_R[2][1]) -
                2.0f * z * (v_R[0][0] + v_R[1][1]) +
                w * (v_R[0][1] - v_R[1][0]))
    );

    float4 quat_n = float4(w, x, y, z);
    return (v_quat_n - dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

// Form Sigma = R * diag(s) * diag(s)^T * R^T via M = R * diag(s), matching the
// covariance construction used by the CUDA implementation.
inline float3x3 covar_from_RS(float3x3 R, float3 s) {
    // float3x3 is column-major: m[col][row].
    // Tensor I/O is row-major: element (row i, col j) is at byte offset (i*3+j)*4.
    float3x3 S = make_diag(s);
    float3x3 M = R * S;
    return M * transpose(M);
}

// Accumulate vector-Jacobian products for covar_from_RS into the rotation matrix
// and scale parameters.
inline void covar_from_RS_vjp(
    float3x3 R,
    float3 s,
    float3x3 v_covar,
    thread float3x3& v_R,
    thread float3& v_s
) {
    float3x3 S = make_diag(s);
    float3x3 M = R * S;
    float3x3 v_M = (v_covar + transpose(v_covar)) * M;
    v_R += v_M * S;

    v_s.x += dot(R[0], v_M[0]);
    v_s.y += dot(R[1], v_M[1]);
    v_s.z += dot(R[2], v_M[2]);
}
