// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../metal_math.h"

using namespace metal;

constant float kAlphaThreshold = 1.0f / 255.0f;
constant float kGaussianExtend = 3.33f;
constant float kMinCompensation = 0.25f;
constant float kEps = 1e-8f;
constant uint kCameraModelPinhole = 0u;
constant uint kCameraModelFisheye = 2u;
constant uint kCameraModelFTheta = 3u;
constant uint kCameraModelLidar = 4u;
constant uint kFThetaRefPixeldistToAngle = 0u;
constant float kLidarAngleToPixelScaling = 1024.0f;
constant float kTwoPi = 6.28318530717958647692f;
constant uint kRollingShutterTopToBottom = 0u;
constant uint kRollingShutterLeftToRight = 1u;
constant uint kRollingShutterBottomToTop = 2u;
constant uint kRollingShutterRightToLeft = 3u;
constant uint kRollingShutterGlobal = 4u;
constant uint kRollingShutterIterations = 10u;

struct ProjectionUT3DGSFusedParams {
    uint B;
    uint C;
    uint N;
    uint image_width;
    uint image_height;
    float eps2d;
    float near_plane;
    float far_plane;
    float radius_clip;
    uint global_z_order;
    float ut_alpha;
    float ut_beta;
    float ut_kappa;
    float ut_in_image_margin_factor;
    uint has_opacities;
    uint has_compensations;
    uint ut_require_all_sigma_points_valid;
    uint camera_model;
    uint has_external_distortion;
    uint has_viewmats_rs;
    uint has_pose_end;
    uint rolling_shutter;
    uint ftheta_reference_poly;
    uint external_h_order;
    uint external_v_order;
    float lidar_fov_horiz_start;
    float lidar_fov_horiz_span;
    float lidar_fov_vert_start;
    float lidar_fov_vert_span;
    float lidar_fov_eps;
    uint lidar_spinning_direction;
};

inline float2 project_pinhole_point(
    float3 cam_point,
    float fx,
    float fy,
    float cx,
    float cy,
    thread bool& valid
) {
    valid = cam_point.z > 0.0f;
    if (!valid) {
        return float2(0.0f);
    }
    float inv_z = 1.0f / cam_point.z;
    return float2(fx * cam_point.x * inv_z + cx, fy * cam_point.y * inv_z + cy);
}

inline float4 quat_conjugate(float4 q) {
    return float4(q.x, -q.y, -q.z, -q.w);
}

inline float4 quat_normalize_safe(float4 q) {
    float norm2 = dot(q, q);
    if (norm2 <= 0.0f) {
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    }
    return q * rsqrt(norm2);
}

inline float4 quat_multiply(float4 q1, float4 q2) {
    float w1 = q1.x;
    float x1 = q1.y;
    float y1 = q1.z;
    float z1 = q1.w;
    float w2 = q2.x;
    float x2 = q2.y;
    float y2 = q2.z;
    float z2 = q2.w;
    return float4(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    );
}

inline float3 quat_rotate(float4 q, float3 v) {
    float4 qn = quat_normalize_safe(q);
    float3 qvec = float3(qn.y, qn.z, qn.w);
    float3 uv = cross(qvec, v);
    float3 uuv = cross(qvec, uv);
    uv *= (2.0f * qn.x);
    uuv *= 2.0f;
    return v + uv + uuv;
}

inline float4 quat_slerp(float4 x, float4 y, float a) {
    float4 x_n = quat_normalize_safe(x);
    float4 y_n = quat_normalize_safe(y);
    float cosTheta = dot(x_n, y_n);
    float4 z = cosTheta < 0.0f ? -y_n : y_n;
    cosTheta = fabs(cosTheta);
    float4 result_lerp = (1.0f - a) * x_n + a * z;
    if (cosTheta > 1.0f - 1e-6f) {
        return quat_normalize_safe(result_lerp);
    }
    float theta = acos(min(cosTheta, 1.0f));
    float sinTheta = sin(theta);
    float4 result_slerp = (sin((1.0f - a) * theta) * x_n + sin(a * theta) * z) / sinTheta;
    return quat_normalize_safe(result_slerp);
}

inline float3 pose_world_point_to_camera(
    float3 t,
    float4 q,
    float3 world_point
) {
    return quat_rotate(q, world_point) + t;
}

inline void interpolate_pose(
    float3 t_start,
    float4 q_start,
    float3 t_end,
    float4 q_end,
    float alpha,
    thread float3& t_rs,
    thread float4& q_rs
) {
    t_rs = mix(t_start, t_end, alpha);
    q_rs = quat_slerp(q_start, q_end, alpha);
}

inline bool image_point_in_margin(
    float2 point,
    uint image_width,
    uint image_height,
    float margin_factor
) {
    float margin_x = float(image_width) * margin_factor;
    float margin_y = float(image_height) * margin_factor;
    return point.x >= -margin_x && point.x < float(image_width) + margin_x &&
        point.y >= -margin_y && point.y < float(image_height) + margin_y;
}

inline float2x2 outer2(float2 a, float2 b) {
    return float2x2(a * b.x, a * b.y);
}

inline float eval_poly_horner(
    device const float* coeffs,
    uint coeff_count,
    float x
) {
    float result = 0.0f;
    for (int i = int(coeff_count) - 1; i >= 0; --i) {
        result = result * x + coeffs[uint(i)];
    }
    return result;
}

inline float eval_poly_inverse_horner_newton(
    device const float* reference_poly,
    device const float* reference_dpoly,
    device const float* approx_inverse_poly,
    float value,
    thread bool& converged
) {
    float x = eval_poly_horner(approx_inverse_poly, 6u, value);
    converged = true;
    for (uint i = 0; i < 3u; ++i) {
        float fx = eval_poly_horner(reference_poly, 6u, x) - value;
        float dfx = eval_poly_horner(reference_dpoly, 5u, x);
        if (fabs(dfx) <= 1e-8f || !isfinite(dfx)) {
            converged = false;
            break;
        }
        x -= fx / dfx;
    }
    converged = converged && isfinite(x);
    return converged ? x : 0.0f;
}

inline float add_blur_metal(float eps2d, thread float2x2& covar, thread float& compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(kMinCompensation * kMinCompensation, det_orig / det_blur));
    return det_blur;
}

inline float2 project_ftheta_point(
    float3 cam_point,
    float cx,
    float cy,
    device const float* pixeldist_to_angle_poly,
    device const float* angle_to_pixeldist_poly,
    device const float* dreference_poly,
    device const float* linear_cde,
    device const float* max_angle_ptr,
    uint reference_poly_type,
    thread bool& valid
) {
    bool not_behind_camera = cam_point.z > 0.0f;
    float2 xy = cam_point.xy;
    float xy_norm = length(xy);
    if (xy_norm <= 0.0f) {
        xy_norm = numeric_limits<float>::epsilon();
    }

    float theta_full = atan2(xy_norm, cam_point.z);
    float max_angle = max_angle_ptr[0];
    float theta = min(theta_full, max_angle);

    bool converged = true;
    float delta = 0.0f;
    if (reference_poly_type == kFThetaRefPixeldistToAngle) {
        delta = eval_poly_inverse_horner_newton(
            pixeldist_to_angle_poly,
            dreference_poly,
            angle_to_pixeldist_poly,
            theta,
            converged
        );
    } else {
        delta = eval_poly_horner(angle_to_pixeldist_poly, 6u, theta);
    }

    float image_point_x = delta * cam_point.x / xy_norm;
    float image_point_y = delta * cam_point.y / xy_norm;
    float c = linear_cde[0];
    float d = linear_cde[1];
    float e = linear_cde[2];
    float2 image_point = float2(
        c * image_point_x + d * image_point_y + (cx + 0.5f),
        e * image_point_x + image_point_y + (cy + 0.5f)
    );

    valid = not_behind_camera && converged && (theta_full <= max_angle);
    return valid ? image_point : float2(0.0f);
}

inline float positive_fmod(float x, float y) {
    float r = fmod(x, y);
    return r < 0.0f ? r + y : r;
}

inline float relative_angle_lidar(float angle_ref, float angle, uint spinning_direction) {
    float rel = spinning_direction == 0u ? angle_ref - angle : angle - angle_ref;
    return positive_fmod(rel, kTwoPi);
}

inline float relative_clock_rotation_lidar(float angle_ref, float angle, uint spinning_direction) {
    return spinning_direction == 0u ? angle_ref - angle : angle - angle_ref;
}

inline float2 project_lidar_point(
    float3 cam_point,
    constant ProjectionUT3DGSFusedParams& params,
    float margin_factor,
    thread bool& valid
) {
    float3 ray = normalize(cam_point);
    float azimuth = atan2(ray.y, ray.x);
    float elevation = asin(clamp(ray.z, -1.0f, 1.0f));
    float column = azimuth * kLidarAngleToPixelScaling;
    float row = elevation * kLidarAngleToPixelScaling;
    float2 image_point = float2(column, row);

    float rel_az = relative_angle_lidar(params.lidar_fov_horiz_start, azimuth, params.lidar_spinning_direction);
    float rel_el = relative_clock_rotation_lidar(params.lidar_fov_vert_start, elevation, 0u);
    float margin_elevation = margin_factor * params.lidar_fov_vert_span;
    float margin_azimuth = margin_factor * params.lidar_fov_horiz_span;
    valid =
        (rel_el <= params.lidar_fov_vert_span + margin_elevation) &&
        (rel_az <= params.lidar_fov_horiz_span + margin_azimuth) &&
        (rel_el >= -margin_elevation) &&
        (rel_az >= -margin_azimuth);
    return image_point;
}

inline float shutter_relative_frame_time(
    float2 pixel_coords,
    uint image_width,
    uint image_height,
    uint rolling_shutter
) {
    float px = pixel_coords.x;
    float py = pixel_coords.y;
    if (rolling_shutter == kRollingShutterTopToBottom) {
        return image_height > 1u ? floor(py) / float(image_height - 1u) : 0.5f;
    }
    if (rolling_shutter == kRollingShutterLeftToRight) {
        return image_width > 1u ? floor(px) / float(image_width - 1u) : 0.5f;
    }
    if (rolling_shutter == kRollingShutterBottomToTop) {
        return image_height > 1u ? (float(image_height) - ceil(py)) / float(image_height - 1u) : 0.5f;
    }
    if (rolling_shutter == kRollingShutterRightToLeft) {
        return image_width > 1u ? (float(image_width) - ceil(px)) / float(image_width - 1u) : 0.5f;
    }
    return 0.0f;
}

inline float2 project_camera_point(
    float3 sigma_c,
    float fx,
    float fy,
    float cx,
    float cy,
    uint camera_index,
    device const float* radial_coeffs,
    device const float* tangential_coeffs,
    device const float* thin_prism_coeffs,
    device const float* fisheye_max_angle,
    device const float* ftheta_pixeldist_to_angle_poly,
    device const float* ftheta_angle_to_pixeldist_poly,
    device const float* ftheta_dreference_poly,
    device const float* ftheta_linear_cde,
    device const float* ftheta_max_angle,
    device const float* external_h_poly,
    device const float* external_v_poly,
    constant ProjectionUT3DGSFusedParams& params,
    thread bool& valid
) {
    if (params.has_external_distortion != 0u) {
        sigma_c = distort_camera_ray_metal(
            sigma_c,
            external_h_poly,
            external_v_poly,
            params.external_h_order,
            params.external_v_order
        );
    }
    float2 image_point = float2(0.0f);
    if (params.camera_model == kCameraModelLidar) {
        return project_lidar_point(sigma_c, params, params.ut_in_image_margin_factor, valid);
    }
    if (params.camera_model == kCameraModelPinhole) {
        bool valid_depth = false;
        image_point = project_pinhole_point(sigma_c, fx, fy, cx, cy, valid_depth);
        valid = valid_depth;
        if (valid && (radial_coeffs != nullptr || tangential_coeffs != nullptr || thin_prism_coeffs != nullptr)) {
            float2 uv = sigma_c.xy / sigma_c.z;
            float u = uv.x;
            float v = uv.y;
            float r2 = u * u + v * v;
            float a1 = 2.0f * u * v;
            float a2 = r2 + 2.0f * u * u;
            float a3 = r2 + 2.0f * v * v;

            float k1 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 6u + 0u] : 0.0f;
            float k2 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 6u + 1u] : 0.0f;
            float k3 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 6u + 2u] : 0.0f;
            float k4 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 6u + 3u] : 0.0f;
            float k5 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 6u + 4u] : 0.0f;
            float k6 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 6u + 5u] : 0.0f;
            float p1 = tangential_coeffs != nullptr ? tangential_coeffs[camera_index * 2u + 0u] : 0.0f;
            float p2 = tangential_coeffs != nullptr ? tangential_coeffs[camera_index * 2u + 1u] : 0.0f;
            float s1 = thin_prism_coeffs != nullptr ? thin_prism_coeffs[camera_index * 4u + 0u] : 0.0f;
            float s2 = thin_prism_coeffs != nullptr ? thin_prism_coeffs[camera_index * 4u + 1u] : 0.0f;
            float s3 = thin_prism_coeffs != nullptr ? thin_prism_coeffs[camera_index * 4u + 2u] : 0.0f;
            float s4 = thin_prism_coeffs != nullptr ? thin_prism_coeffs[camera_index * 4u + 3u] : 0.0f;

            float icD_num = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
            float icD_den = 1.0f + r2 * (k4 + r2 * (k5 + r2 * k6));
            float icD = icD_num / icD_den;
            float2 delta = float2(
                p1 * a1 + p2 * a2 + r2 * (s1 + r2 * s2),
                p1 * a3 + p2 * a1 + r2 * (s3 + r2 * s4)
            );
            float2 uvND = icD * uv + delta;
            image_point = float2(fx * uvND.x + cx, fy * uvND.y + cy);
            valid = valid && (icD > 0.8f);
        }
        return image_point;
    }
    if (params.camera_model == kCameraModelFisheye) {
        bool valid_depth = sigma_c.z > 0.0f;
        valid = valid_depth;
        float2 xy = sigma_c.xy;
        float xy_norm = length(xy);
        if (xy_norm <= 0.0f) {
            xy_norm = kEps;
        }
        float theta_full = atan2(xy_norm, sigma_c.z);
        float max_angle = fisheye_max_angle[camera_index];
        float theta = min(theta_full, max_angle);
        float theta2 = theta * theta;
        float k1 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 4u + 0u] : 0.0f;
        float k2 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 4u + 1u] : 0.0f;
        float k3 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 4u + 2u] : 0.0f;
        float k4 = radial_coeffs != nullptr ? radial_coeffs[camera_index * 4u + 3u] : 0.0f;
        float poly_value = theta * (1.0f + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))));
        float delta = poly_value / xy_norm;
        valid = valid && (delta > 0.0f) && (theta_full <= max_angle);
        float2 uv = delta * xy;
        return float2(fx * uv.x + cx, fy * uv.y + cy);
    }
    return project_ftheta_point(
        sigma_c,
        cx,
        cy,
        ftheta_pixeldist_to_angle_poly,
        ftheta_angle_to_pixeldist_poly,
        ftheta_dreference_poly,
        ftheta_linear_cde,
        ftheta_max_angle,
        params.ftheta_reference_poly,
        valid
    );
}

inline float2 project_world_point_shutter_pose(
    float3 world_point,
    float3 t_start,
    float4 q_start,
    float3 t_end,
    float4 q_end,
    float fx,
    float fy,
    float cx,
    float cy,
    uint camera_index,
    device const float* radial_coeffs,
    device const float* tangential_coeffs,
    device const float* thin_prism_coeffs,
    device const float* fisheye_max_angle,
    device const float* ftheta_pixeldist_to_angle_poly,
    device const float* ftheta_angle_to_pixeldist_poly,
    device const float* ftheta_dreference_poly,
    device const float* ftheta_linear_cde,
    device const float* ftheta_max_angle,
    device const float* external_h_poly,
    device const float* external_v_poly,
    constant ProjectionUT3DGSFusedParams& params,
    thread bool& valid
) {
    float3 sigma_c_start = pose_world_point_to_camera(t_start, q_start, world_point);
    bool valid_start = false;
    float2 image_points_start = project_camera_point(
        sigma_c_start, fx, fy, cx, cy, camera_index,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, fisheye_max_angle,
        ftheta_pixeldist_to_angle_poly, ftheta_angle_to_pixeldist_poly, ftheta_dreference_poly,
        ftheta_linear_cde, ftheta_max_angle, external_h_poly, external_v_poly, params, valid_start
    );
    bool valid_bounds_start = params.camera_model == kCameraModelLidar ? true : image_point_in_margin(
        image_points_start, params.image_width, params.image_height, params.ut_in_image_margin_factor
    );
    valid_start = valid_start && valid_bounds_start;
    if (params.rolling_shutter == kRollingShutterGlobal) {
        valid = valid_start;
        return image_points_start;
    }

    float3 sigma_c_end = pose_world_point_to_camera(t_end, q_end, world_point);
    bool valid_end = false;
    float2 image_points_end = project_camera_point(
        sigma_c_end, fx, fy, cx, cy, camera_index,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, fisheye_max_angle,
        ftheta_pixeldist_to_angle_poly, ftheta_angle_to_pixeldist_poly, ftheta_dreference_poly,
        ftheta_linear_cde, ftheta_max_angle, external_h_poly, external_v_poly, params, valid_end
    );
    bool valid_bounds_end = params.camera_model == kCameraModelLidar ? true : image_point_in_margin(
        image_points_end, params.image_width, params.image_height, params.ut_in_image_margin_factor
    );
    valid_end = valid_end && valid_bounds_end;

    float2 init_image_points = valid_start ? image_points_start : image_points_end;
    bool valid_any = valid_start || valid_end;
    float2 image_points_prev = init_image_points;
    bool valid_rs = valid_any;

    for (uint iter = 0; iter < kRollingShutterIterations; ++iter) {
        if (!valid_any) {
            break;
        }
        float relative_time = shutter_relative_frame_time(
            image_points_prev, params.image_width, params.image_height, params.rolling_shutter
        );
        float3 t_rs;
        float4 q_rs;
        interpolate_pose(t_start, q_start, t_end, q_end, relative_time, t_rs, q_rs);
        float3 sigma_c_rs = pose_world_point_to_camera(t_rs, q_rs, world_point);
        float2 image_points_rs = project_camera_point(
            sigma_c_rs, fx, fy, cx, cy, camera_index,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs, fisheye_max_angle,
            ftheta_pixeldist_to_angle_poly, ftheta_angle_to_pixeldist_poly, ftheta_dreference_poly,
            ftheta_linear_cde, ftheta_max_angle, external_h_poly, external_v_poly, params, valid_rs
        );
        bool valid_bounds_rs = params.camera_model == kCameraModelLidar ? true : image_point_in_margin(
            image_points_rs, params.image_width, params.image_height, params.ut_in_image_margin_factor
        );
        valid_rs = valid_rs && valid_bounds_rs;
        image_points_prev = image_points_rs;
    }

    valid = valid_any && valid_rs;
    return valid_any ? image_points_prev : init_image_points;
}

kernel void projection_ut_3dgs_fused_kernel(
    device const float* means [[buffer(0)]],
    device const float* quats [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* opacities [[buffer(3)]],
    device const float* viewmats [[buffer(4)]],
    device const float* viewmats_rs [[buffer(5)]],
    device const float* pose_start [[buffer(6)]],
    device const float* pose_end [[buffer(7)]],
    device const float* Ks [[buffer(8)]],
    device const float* radial_coeffs [[buffer(9)]],
    device const float* tangential_coeffs [[buffer(10)]],
    device const float* thin_prism_coeffs [[buffer(11)]],
    device const float* fisheye_max_angle [[buffer(12)]],
    device const float* ftheta_pixeldist_to_angle_poly [[buffer(13)]],
    device const float* ftheta_angle_to_pixeldist_poly [[buffer(14)]],
    device const float* ftheta_dreference_poly [[buffer(15)]],
    device const float* ftheta_linear_cde [[buffer(16)]],
    device const float* ftheta_max_angle [[buffer(17)]],
    device const float* external_h_poly [[buffer(18)]],
    device const float* external_v_poly [[buffer(19)]],
    device int* radii [[buffer(20)]],
    device float* means2d [[buffer(21)]],
    device float* depths [[buffer(22)]],
    device float* conics [[buffer(23)]],
    device float* compensations [[buffer(24)]],
    constant ProjectionUT3DGSFusedParams& params [[buffer(25)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.B * params.C * params.N) {
        return;
    }

    const uint bid = id / (params.C * params.N);
    const uint cid = (id / params.N) % params.C;
    const uint gid = id % params.N;

    device const float* mean_ptr = means + (bid * params.N + gid) * 3u;
    device const float* quat_ptr = quats + (bid * params.N + gid) * 4u;
    device const float* scale_ptr = scales + (bid * params.N + gid) * 3u;
    device const float* K_ptr = Ks + (bid * params.C + cid) * 9u;
    device const float* pose_start_ptr = pose_start + (bid * params.C + cid) * 7u;
    device const float* pose_end_ptr = params.has_pose_end != 0u ? pose_end + (bid * params.C + cid) * 7u : nullptr;

    float4 quat_raw = float4(quat_ptr[0], quat_ptr[1], quat_ptr[2], quat_ptr[3]);
    float quat_norm2 = dot(quat_raw, quat_raw);
    if (quat_norm2 <= kEps) {
        return;
    }

    float3 scale = float3(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
    if (scale.x <= kEps || scale.y <= kEps || scale.z <= kEps) {
        return;
    }

    float3 mean = float3(mean_ptr[0], mean_ptr[1], mean_ptr[2]);
    float3x3 R_world = quat_to_rotmat(quat_raw);
    float3 t_start = float3(pose_start_ptr[0], pose_start_ptr[1], pose_start_ptr[2]);
    float4 q_start = float4(pose_start_ptr[3], pose_start_ptr[4], pose_start_ptr[5], pose_start_ptr[6]);
    float3 t_center = t_start;
    float4 q_center = q_start;
    if (params.rolling_shutter != kRollingShutterGlobal) {
        float3 t_end = float3(pose_end_ptr[0], pose_end_ptr[1], pose_end_ptr[2]);
        float4 q_end = float4(pose_end_ptr[3], pose_end_ptr[4], pose_end_ptr[5], pose_end_ptr[6]);
        interpolate_pose(t_start, q_start, t_end, q_end, 0.5f, t_center, q_center);
    }
    float3 mean_c = pose_world_point_to_camera(t_center, q_center, mean);
    float cull_depth = params.global_z_order != 0u ? mean_c.z : length(mean_c);
    if (cull_depth < params.near_plane || cull_depth > params.far_plane) {
        return;
    }

    float fx = K_ptr[0];
    float fy = K_ptr[4];
    float cx = K_ptr[2];
    float cy = K_ptr[5];

    float lambda = params.ut_alpha * params.ut_alpha * (3.0f + params.ut_kappa) - 3.0f;
    float scale_factor = sqrt(3.0f + lambda);
    float w_mean_center = lambda / (3.0f + lambda);
    float w_cov_center = w_mean_center + (1.0f - params.ut_alpha * params.ut_alpha + params.ut_beta);
    float w_other = 1.0f / (2.0f * (3.0f + lambda));

    float3 delta0 = R_world[0] * (scale.x * scale_factor);
    float3 delta1 = R_world[1] * (scale.y * scale_factor);
    float3 delta2 = R_world[2] * (scale.z * scale_factor);
    float3 sigma_points[7] = {
        mean,
        mean + delta0,
        mean + delta1,
        mean + delta2,
        mean - delta0,
        mean - delta1,
        mean - delta2,
    };

    const uint camera_index = bid * params.C + cid;
    float2 image_points[7];
    bool valid_points[7];
    float3 t_end = t_start;
    float4 q_end = q_start;
    if (params.rolling_shutter != kRollingShutterGlobal) {
        t_end = float3(pose_end_ptr[0], pose_end_ptr[1], pose_end_ptr[2]);
        q_end = float4(pose_end_ptr[3], pose_end_ptr[4], pose_end_ptr[5], pose_end_ptr[6]);
    }
    for (uint i = 0; i < 7u; ++i) {
        bool valid = false;
        float2 image_point = project_world_point_shutter_pose(
            sigma_points[i],
            t_start,
            q_start,
            t_end,
            q_end,
            fx,
            fy,
            cx,
            cy,
            camera_index,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            fisheye_max_angle,
            ftheta_pixeldist_to_angle_poly,
            ftheta_angle_to_pixeldist_poly,
            ftheta_dreference_poly,
            ftheta_linear_cde,
            ftheta_max_angle,
            external_h_poly,
            external_v_poly,
            params,
            valid
        );
        valid_points[i] = valid;
        image_points[i] = image_point;
    }

    bool valid_gaussian = false;
    float2 mean2d_val = float2(0.0f);
    float weights_cov_eff[7];
    if (params.ut_require_all_sigma_points_valid != 0u) {
        bool cumulative = true;
        float mean_weights[7];
        for (uint i = 0; i < 7u; ++i) {
            cumulative = cumulative && valid_points[i];
            float base_mean = i == 0u ? w_mean_center : w_other;
            float base_cov = i == 0u ? w_cov_center : w_other;
            float mask = cumulative ? 1.0f : 0.0f;
            mean_weights[i] = base_mean * mask;
            weights_cov_eff[i] = base_cov * mask;
            mean2d_val += image_points[i] * mean_weights[i];
        }
        valid_gaussian = cumulative;
    } else {
        for (uint i = 0; i < 7u; ++i) {
            if (valid_points[i]) {
                valid_gaussian = true;
            }
            float mean_weight = i == 0u ? w_mean_center : w_other;
            weights_cov_eff[i] = i == 0u ? w_cov_center : w_other;
            mean2d_val += image_points[i] * mean_weight;
        }
    }

    if (!valid_gaussian) {
        return;
    }

    float2x2 cov2d = float2x2(float2(0.0f), float2(0.0f));
    for (uint i = 0; i < 7u; ++i) {
        float2 delta = image_points[i] - mean2d_val;
        cov2d += outer2(delta, delta) * weights_cov_eff[i];
    }

    float compensation = 1.0f;
    float det = add_blur_metal(params.eps2d, cov2d, compensation);
    if (det <= 0.0f) {
        return;
    }
    if (cov2d[0][0] < 0.0f || cov2d[1][1] < 0.0f) {
        return;
    }

    float effective_opacity = 1.0f;
    float extend = kGaussianExtend;
    if (params.has_opacities != 0u) {
        effective_opacity = opacities[bid * params.N + gid] * compensation;
        if (effective_opacity < kAlphaThreshold) {
            return;
        }
        extend = min(
            kGaussianExtend,
            sqrt(max(0.0f, 2.0f * log(effective_opacity / kAlphaThreshold)))
        );
    }

    float b = 0.5f * (cov2d[0][0] + cov2d[1][1]);
    float tmp = sqrt(max(0.01f, b * b - det));
    float v1 = b + tmp;
    float r1 = extend * sqrt(max(0.0f, v1));
    float radius_x = ceil(min(extend * sqrt(max(0.0f, cov2d[0][0])), r1));
    float radius_y = ceil(min(extend * sqrt(max(0.0f, cov2d[1][1])), r1));

    if (max(radius_x, radius_y) <= params.radius_clip) {
        return;
    }
    if (params.camera_model != kCameraModelLidar) {
        if (mean2d_val.x + radius_x <= 0.0f || mean2d_val.x - radius_x >= float(params.image_width) ||
            mean2d_val.y + radius_y <= 0.0f || mean2d_val.y - radius_y >= float(params.image_height)) {
            return;
        }
    }

    float inv_det = 1.0f / det;
    float2x2 cov2d_inv = float2x2(
        float2(cov2d[1][1] * inv_det, -cov2d[0][1] * inv_det),
        float2(-cov2d[1][0] * inv_det, cov2d[0][0] * inv_det)
    );

    radii[id * 2u] = int(radius_x);
    radii[id * 2u + 1u] = int(radius_y);
    means2d[id * 2u] = mean2d_val.x;
    means2d[id * 2u + 1u] = mean2d_val.y;
    depths[id] = params.global_z_order != 0u ? mean_c.z : length(mean_c);
    conics[id * 3u] = cov2d_inv[0][0];
    conics[id * 3u + 1u] = cov2d_inv[0][1];
    conics[id * 3u + 2u] = cov2d_inv[1][1];
    if (params.has_compensations != 0u) {
        compensations[id] = compensation;
    }
}
