// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Per-gaussian projection backward on the blend backward's tile-summed record gradients; exact because
// the backward is linear in its seed: sum_k J^T g_k == J^T sum_k g_k.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void projectionBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    // Tile-summed BlendData seeds; this kernel is their only reader, so it also resets them.
    RWStructuredBuffer<fixed_t> grad_mean2d_x,
    RWStructuredBuffer<fixed_t> grad_mean2d_y,
    RWStructuredBuffer<fixed_t> grad_conic_a,
    RWStructuredBuffer<fixed_t> grad_conic_b,
    RWStructuredBuffer<fixed_t> grad_conic_c,
    RWStructuredBuffer<fixed_t> grad_amplitude,
    // The blend backward already added any direct (opaque_core falloff) term; this adds the amplitude's.
    RWStructuredBuffer<fixed_t> grad_opacity,
    // 3D parameter gradients (this kernel is the sole writer per gaussian).
    RWStructuredBuffer<fixed_t> grad_mean_x,
    RWStructuredBuffer<fixed_t> grad_mean_y,
    RWStructuredBuffer<fixed_t> grad_mean_z,
    RWStructuredBuffer<fixed_t> grad_log_scale_x,
    RWStructuredBuffer<fixed_t> grad_log_scale_y,
    RWStructuredBuffer<fixed_t> grad_log_scale_z,
    RWStructuredBuffer<fixed_t> grad_rot_w,
    RWStructuredBuffer<fixed_t> grad_rot_x,
    RWStructuredBuffer<fixed_t> grad_rot_y,
    RWStructuredBuffer<fixed_t> grad_rot_z,
    // Mip-Splatting inputs to rebuild bin_count's exact cov2d; filter_3D is garbage unless mip_enabled.
    StructuredBuffer<float> filter_3D,
    uniform uint mip_enabled,
    uniform float kernel_size,
    // Cluster-cull mask: seeded all-ones when disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    // Dead slots leave the cleared 3D grads at zero (matches adam.slang).
    if (gid >= num_gaussians_counter[0]) return;
    // Culled clusters have zero 2D seeds; their 3D grads stay zero (Adam's freeze-state contract).
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    fixed_t seed_mean_x = grad_mean2d_x[gid], seed_mean_y = grad_mean2d_y[gid];
    fixed_t seed_conic_a = grad_conic_a[gid], seed_conic_b = grad_conic_b[gid];
    fixed_t seed_conic_c = grad_conic_c[gid], seed_amplitude = grad_amplitude[gid];
    // Reset for the next step; untouched slots are already zero, so no capacity-wide clear is needed.
    if ((seed_mean_x | seed_mean_y | seed_conic_a | seed_conic_b | seed_conic_c | seed_amplitude)
        != fixed_t(0))
    {
        grad_mean2d_x[gid] = fixed_t(0); grad_mean2d_y[gid] = fixed_t(0);
        grad_conic_a[gid] = fixed_t(0); grad_conic_b[gid] = fixed_t(0);
        grad_conic_c[gid] = fixed_t(0); grad_amplitude[gid] = fixed_t(0);
    }
    else
    {
        grad_mean_x[gid] = 0; grad_mean_y[gid] = 0; grad_mean_z[gid] = 0;
        grad_log_scale_x[gid] = 0; grad_log_scale_y[gid] = 0; grad_log_scale_z[gid] = 0;
        grad_rot_w[gid] = 0; grad_rot_x[gid] = 0; grad_rot_y[gid] = 0; grad_rot_z[gid] = 0;
        return;
    }
    float2 grad_mean2d = float2(fromFixed(seed_mean_x), fromFixed(seed_mean_y));
    float3 grad_conic = float3(fromFixed(seed_conic_a), fromFixed(seed_conic_b), fromFixed(seed_conic_c));

    Gaussian3D g = gaussians[gid];

    GaussianModel model;
    model.mean = g.mean;
    model.log_scale = g.log_scale;
    model.rot_quat = g.rot_quat;
    var dp_model = diffPair(model);

    float f3d = (mip_enabled != 0u) ? filter_3D[gid] : 0.0;

    // Rebuild bin_count's cov2d and Mip compensation, then chain the record seeds back to them.
    float2 mean_2d; float3 cov2d; float depth_vs, cov2d_det, comp;
    model.projectInto(camera[0], f3d, kernel_size, mip_enabled,
                      mean_2d, cov2d, depth_vs, cov2d_det, comp);
    var dp_cov2d = diffPair(cov2d);
    bwd_diff(conicFromCov)(dp_cov2d, grad_conic);
    var dp_logit = diffPair(g.opacity);
    var dp_comp = diffPair(comp);
    if (seed_amplitude != fixed_t(0))
    {
        bwd_diff(splatAmplitude)(dp_logit, dp_comp, fromFixed(seed_amplitude));
        grad_opacity[gid] += toFixed(dp_logit.d);
    }

    // Depth/det seeds are 0; the projectModel wrapper exists because interface methods aren't bwd_diff-able.
    bwd_diff(projectModel<GaussianModel>)(
        dp_model, camera[0], f3d, kernel_size, mip_enabled,
        grad_mean2d, dp_cov2d.d, 0.0, 0.0, dp_comp.d);

    // Sole writer, so a plain store; later regularization/Adam dispatches InterlockedAdd on top.
    grad_mean_x[gid] = toFixed(dp_model.d.mean.x);
    grad_mean_y[gid] = toFixed(dp_model.d.mean.y);
    grad_mean_z[gid] = toFixed(dp_model.d.mean.z);
    grad_log_scale_x[gid] = toFixed(dp_model.d.log_scale.x);
    grad_log_scale_y[gid] = toFixed(dp_model.d.log_scale.y);
    grad_log_scale_z[gid] = toFixed(dp_model.d.log_scale.z);
    grad_rot_w[gid] = toFixed(dp_model.d.rot_quat.x);
    grad_rot_x[gid] = toFixed(dp_model.d.rot_quat.y);
    grad_rot_y[gid] = toFixed(dp_model.d.rot_quat.z);
    grad_rot_z[gid] = toFixed(dp_model.d.rot_quat.w);
}
