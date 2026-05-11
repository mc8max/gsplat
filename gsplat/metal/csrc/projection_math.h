// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------------
// Matrix I/O helpers (row-major tensor ↔ column-major MSL float2x2/float3x3)
// ---------------------------------------------------------------------------

inline float2 row(float2x2 M, uint r) {
    return float2(M[0][r], M[1][r]);
}

inline float3 row(float3x3 M, uint r) {
    return float3(M[0][r], M[1][r], M[2][r]);
}

inline float2x2 load_mat2_row_major(device const float* src) {
    return float2x2(
        float2(src[0], src[2]),
        float2(src[1], src[3])
    );
}

inline float3x3 load_mat3_row_major(device const float* src) {
    return float3x3(
        float3(src[0], src[3], src[6]),
        float3(src[1], src[4], src[7]),
        float3(src[2], src[5], src[8])
    );
}

inline void store_mat2_row_major(device float* dst, float2x2 M) {
    dst[0] = M[0][0];
    dst[1] = M[1][0];
    dst[2] = M[0][1];
    dst[3] = M[1][1];
}

inline void store_mat3_row_major(device float* dst, float3x3 M) {
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

// ---------------------------------------------------------------------------
// EWA covariance projection helpers
// ---------------------------------------------------------------------------

// Compute row i of M as a vector (column-major M[col][row] convention).
inline float3 row_mul_mat3(float3 v, float3x3 M) {
    return float3(dot(v, M[0]), dot(v, M[1]), dot(v, M[2]));
}

// Outer product: outer3(a, b)[c][r] = a[r] * b[c].
inline float3x3 outer3(float3 a, float3 b) {
    return float3x3(a * b.x, a * b.y, a * b.z);
}

// Project 3D covariance Σ through a 2×3 Jacobian with rows j0, j1:
//   C = J · Σ · Jᵀ
inline float2x2 project_covar_2x3(float3 j0, float3 j1, float3x3 cov3d) {
    float3 cov_j0 = cov3d * j0;
    float3 cov_j1 = cov3d * j1;
    return float2x2(
        float2(dot(j0, cov_j0), dot(j1, cov_j0)),
        float2(dot(j0, cov_j1), dot(j1, cov_j1))
    );
}

// VJP of project_covar_2x3 w.r.t. Σ, j0, j1.
inline void project_covar_2x3_vjp(
    float3 j0,
    float3 j1,
    float3x3 cov3d,
    float2x2 v_cov2d,
    thread float3x3& v_cov3d,
    thread float3& v_j0,
    thread float3& v_j1
) {
    float g00 = v_cov2d[0][0];
    float g01 = v_cov2d[1][0];
    float g10 = v_cov2d[0][1];
    float g11 = v_cov2d[1][1];

    v_cov3d += outer3(j0, j0) * g00;
    v_cov3d += outer3(j0, j1) * g01;
    v_cov3d += outer3(j1, j0) * g10;
    v_cov3d += outer3(j1, j1) * g11;

    float3 s0 = row_mul_mat3(j0, cov3d);
    float3 s1 = row_mul_mat3(j1, cov3d);
    v_j0 += s0 * (2.0f * g00) + s1 * (g01 + g10);
    v_j1 += s0 * (g01 + g10) + s1 * (2.0f * g11);
}

// ---------------------------------------------------------------------------
// Orthographic projection
// ---------------------------------------------------------------------------

inline void ortho_proj_metal(
    float3 mean3d,
    float3x3 cov3d,
    float fx,
    float fy,
    float cx,
    float cy,
    uint width,
    uint height,
    thread float2x2& cov2d,
    thread float2& mean2d
) {
    (void)width;
    (void)height;
    float3 j0 = float3(fx, 0.0f, 0.0f);
    float3 j1 = float3(0.0f, fy, 0.0f);
    cov2d = project_covar_2x3(j0, j1, cov3d);
    mean2d = float2(fx * mean3d.x + cx, fy * mean3d.y + cy);
}

inline void ortho_proj_vjp_metal(
    float3 mean3d,
    float3x3 cov3d,
    float fx,
    float fy,
    float cx,
    float cy,
    uint width,
    uint height,
    float2x2 v_cov2d,
    float2 v_mean2d,
    thread float3& v_mean3d,
    thread float3x3& v_cov3d
) {
    (void)mean3d;
    (void)cx;
    (void)cy;
    (void)width;
    (void)height;
    float3 j0 = float3(fx, 0.0f, 0.0f);
    float3 j1 = float3(0.0f, fy, 0.0f);
    // v_j0 and v_j1 are computed but intentionally discarded: j0 and j1 are
    // constants for ortho projection, so their gradients do not propagate.
    float3 v_j0 = float3(0.0f);
    float3 v_j1 = float3(0.0f);
    project_covar_2x3_vjp(j0, j1, cov3d, v_cov2d, v_cov3d, v_j0, v_j1);
    v_mean3d += float3(fx * v_mean2d.x, fy * v_mean2d.y, 0.0f);
}

// ---------------------------------------------------------------------------
// Perspective (pinhole) projection
// ---------------------------------------------------------------------------

inline void persp_proj_metal(
    float3 mean3d,
    float3x3 cov3d,
    float fx,
    float fy,
    float cx,
    float cy,
    uint width,
    uint height,
    thread float2x2& cov2d,
    thread float2& mean2d
) {
    float x = mean3d.x;
    float y = mean3d.y;
    float z = mean3d.z;

    float tan_fovx = 0.5f * float(width) / fx;
    float tan_fovy = 0.5f * float(height) / fy;
    float lim_x_pos = (float(width) - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (float(height) - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.0f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    float3 j0 = float3(fx * rz, 0.0f, -fx * tx * rz2);
    float3 j1 = float3(0.0f, fy * rz, -fy * ty * rz2);
    cov2d = project_covar_2x3(j0, j1, cov3d);
    mean2d = float2(fx * x * rz + cx, fy * y * rz + cy);
}

inline void persp_proj_vjp_metal(
    float3 mean3d,
    float3x3 cov3d,
    float fx,
    float fy,
    float cx,
    float cy,
    uint width,
    uint height,
    float2x2 v_cov2d,
    float2 v_mean2d,
    thread float3& v_mean3d,
    thread float3x3& v_cov3d
) {
    float x = mean3d.x;
    float y = mean3d.y;
    float z = mean3d.z;

    float tan_fovx = 0.5f * float(width) / fx;
    float tan_fovy = 0.5f * float(height) / fy;
    float lim_x_pos = (float(width) - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (float(height) - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.0f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));
    float3 j0 = float3(fx * rz, 0.0f, -fx * tx * rz2);
    float3 j1 = float3(0.0f, fy * rz, -fy * ty * rz2);

    float3 v_j0 = float3(0.0f);
    float3 v_j1 = float3(0.0f);
    project_covar_2x3_vjp(j0, j1, cov3d, v_cov2d, v_cov3d, v_j0, v_j1);

    v_mean3d += float3(
        fx * rz * v_mean2d.x,
        fy * rz * v_mean2d.y,
        -(fx * x * v_mean2d.x + fy * y * v_mean2d.y) * rz2
    );

    float rz3 = rz2 * rz;
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_j0.z;
    } else {
        v_mean3d.z += -fx * rz3 * v_j0.z * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_j1.z;
    } else {
        v_mean3d.z += -fy * rz3 * v_j1.z * ty;
    }
    v_mean3d.z += -fx * rz2 * v_j0.x - fy * rz2 * v_j1.y +
                  2.0f * fx * tx * rz3 * v_j0.z +
                  2.0f * fy * ty * rz3 * v_j1.z;
}

// ---------------------------------------------------------------------------
// Fisheye projection
// ---------------------------------------------------------------------------

inline void fisheye_proj_metal(
    float3 mean3d,
    float3x3 cov3d,
    float fx,
    float fy,
    float cx,
    float cy,
    uint width,
    uint height,
    thread float2x2& cov2d,
    thread float2& mean2d
) {
    (void)width;
    (void)height;
    float x = mean3d.x;
    float y = mean3d.y;
    float z = mean3d.z;

    float eps = 0.0000001f;
    float xy_len = length(float2(x, y)) + eps;
    float theta = atan2(xy_len, z + eps);
    mean2d = float2(x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy);

    float x2_eps = x * x + eps;  // x² + eps for numerical stability at x=0
    float y2 = y * y;
    float xy = x * y;
    float x2y2 = x2_eps + y2;
    float x2y2z2_inv = 1.0f / (x2y2 + z * z);

    float b = atan2(xy_len, z) / xy_len / x2y2;
    float a = z * x2y2z2_inv / x2y2;
    float3 j0 = float3(
        fx * (x2_eps * a + y2 * b),
        fx * xy * (a - b),
        -fx * x * x2y2z2_inv
    );
    float3 j1 = float3(
        fy * xy * (a - b),
        fy * (y2 * a + x2_eps * b),
        -fy * y * x2y2z2_inv
    );
    cov2d = project_covar_2x3(j0, j1, cov3d);
}

inline void fisheye_proj_vjp_metal(
    float3 mean3d,
    float3x3 cov3d,
    float fx,
    float fy,
    float cx,
    float cy,
    uint width,
    uint height,
    float2x2 v_cov2d,
    float2 v_mean2d,
    thread float3& v_mean3d,
    thread float3x3& v_cov3d
) {
    (void)cx;
    (void)cy;
    (void)width;
    (void)height;
    float x = mean3d.x;
    float y = mean3d.y;
    float z = mean3d.z;

    const float eps = 0.0000001f;
    float x2_eps = x * x + eps;  // x² + eps for numerical stability at x=0
    float y2 = y * y;
    float xy = x * y;
    float x2y2 = x2_eps + y2;
    float len_xy = length(float2(x, y)) + eps;
    float x2y2z2 = x2y2 + z * z;
    float x2y2z2_inv = 1.0f / x2y2z2;
    float b = atan2(len_xy, z) / len_xy / x2y2;
    float a = z * x2y2z2_inv / x2y2;

    v_mean3d += float3(
        fx * (x2_eps * a + y2 * b) * v_mean2d.x + fy * xy * (a - b) * v_mean2d.y,
        fx * xy * (a - b) * v_mean2d.x + fy * (y2 * a + x2_eps * b) * v_mean2d.y,
        -fx * x * x2y2z2_inv * v_mean2d.x - fy * y * x2y2z2_inv * v_mean2d.y
    );

    float theta = atan2(len_xy, z);
    float j_b = theta / len_xy / x2y2;
    float j_a = z * x2y2z2_inv / x2y2;
    float3 j0 = float3(
        fx * (x2_eps * j_a + y2 * j_b),
        fx * xy * (j_a - j_b),
        -fx * x * x2y2z2_inv
    );
    float3 j1 = float3(
        fy * xy * (j_a - j_b),
        fy * (y2 * j_a + x2_eps * j_b),
        -fy * y * x2y2z2_inv
    );

    float3 v_j0 = float3(0.0f);
    float3 v_j1 = float3(0.0f);
    project_covar_2x3_vjp(j0, j1, cov3d, v_cov2d, v_cov3d, v_j0, v_j1);

    float l4 = x2y2z2 * x2y2z2;
    float E = -l4 * x2y2 * theta + x2y2z2 * x2y2 * len_xy * z;
    float F = 3.0f * l4 * theta - 3.0f * x2y2z2 * len_xy * z - 2.0f * x2y2 * len_xy * z;

    float A = x * (3.0f * E + x2_eps * F);
    float B = y * (E + x2_eps * F);
    float C = x * (E + y2 * F);
    float D = y * (3.0f * E + y2 * F);

    float S1 = x2_eps - y2 - z * z;
    float S2 = y2 - x2_eps - z * z;
    float inv1 = x2y2z2_inv * x2y2z2_inv;
    float inv2 = inv1 / (x2y2 * x2y2 * len_xy);

    float dJ_dx00 = fx * A * inv2;
    float dJ_dx01 = fx * B * inv2;
    float dJ_dx02 = fx * S1 * inv1;
    float dJ_dx10 = fy * B * inv2;
    float dJ_dx11 = fy * C * inv2;
    float dJ_dx12 = 2.0f * fy * xy * inv1;

    float dJ_dy00 = dJ_dx01;
    float dJ_dy01 = fx * C * inv2;
    float dJ_dy02 = 2.0f * fx * xy * inv1;
    float dJ_dy10 = dJ_dx11;
    float dJ_dy11 = fy * D * inv2;
    float dJ_dy12 = fy * S2 * inv1;

    float dJ_dz00 = dJ_dx02;
    float dJ_dz01 = dJ_dy02;
    float dJ_dz02 = 2.0f * fx * x * z * inv1;
    float dJ_dz10 = dJ_dx12;
    float dJ_dz11 = dJ_dy12;
    float dJ_dz12 = 2.0f * fy * y * z * inv1;

    float dL_dx = dJ_dx00 * v_j0.x + dJ_dx01 * v_j0.y + dJ_dx02 * v_j0.z +
                  dJ_dx10 * v_j1.x + dJ_dx11 * v_j1.y + dJ_dx12 * v_j1.z;
    float dL_dy = dJ_dy00 * v_j0.x + dJ_dy01 * v_j0.y + dJ_dy02 * v_j0.z +
                  dJ_dy10 * v_j1.x + dJ_dy11 * v_j1.y + dJ_dy12 * v_j1.z;
    float dL_dz = dJ_dz00 * v_j0.x + dJ_dz01 * v_j0.y + dJ_dz02 * v_j0.z +
                  dJ_dz10 * v_j1.x + dJ_dz11 * v_j1.y + dJ_dz12 * v_j1.z;
    v_mean3d += float3(dL_dx, dL_dy, dL_dz);
}
