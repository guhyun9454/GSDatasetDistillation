#include "backward2d.cuh"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__global__ void project_gaussians_2d_backward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float2* __restrict__ v_mean2d,
    float3* __restrict__ v_L_elements
) {
    unsigned idx = cg::this_grid().thread_rank();
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }

    float3 v_cov_after_clamp;
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov_after_clamp);  // dL/dΣ''

    const float l11 = L_elements[idx].x;
    const float l21 = L_elements[idx].y;
    const float l22 = L_elements[idx].z;

    float sigma_xx = l11 * l11;
    float sigma_xy = l11 * l21;
    float sigma_yy = l21 * l21 + l22 * l22;

    const float px_var = 1.0f / 12.0f;
    const float min_var = 0.25f * 0.25f;

    float sigma_xx_p = sigma_xx + px_var;
    float sigma_yy_p = sigma_yy + px_var;

    const float3 v_cov_before_clamp = make_float3(
        (sigma_xx_p > min_var) ? v_cov_after_clamp.x : 0.0f,
        v_cov_after_clamp.y,
        (sigma_yy_p > min_var) ? v_cov_after_clamp.z : 0.0f
    );

    const float G_11 = v_cov_before_clamp.x;
    const float G_12 = v_cov_before_clamp.y;
    const float G_22 = v_cov_before_clamp.z;

    v_cov2d[idx].x = G_11;
    v_cov2d[idx].y = G_12;
    v_cov2d[idx].z = G_22;

    float grad_l11 = 2.f * l11 * G_11 + G_12 * l21;
    float grad_l21 = G_12 * l11 + 2.f * l21 * G_22;
    float grad_l22 = 2.f * l22 * G_22;

    v_L_elements[idx].x = grad_l11;
    v_L_elements[idx].y = grad_l21;
    v_L_elements[idx].z = grad_l22;

    v_mean2d[idx].x = v_xy[idx].x * (0.5f * img_size.x);
    v_mean2d[idx].y = v_xy[idx].y * (0.5f * img_size.y);

}


__global__ void project_gaussians_2d_scale_rot_backward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float2* __restrict__ v_mean2d,
    float2* __restrict__ v_scale,
    float* __restrict__ v_rot
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points || radii[idx] <= 0) {
        return;
    }
    // get v_cov2d
    cov2d_to_conic_vjp(conics[idx], v_conic[idx], v_cov2d[idx]);

    // get v_scale and v_rot
    // scale_rot_to_cov2d_vjp(
    //     scales2d[idx],
    //     rotation[idx],
    //     v_cov2d[idx],
    //     v_scale[idx],
    //     v_rot[idx]
    // );

    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 R_g = rotmat2d_gradient(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 theta_g = R_g * S * glm::transpose(M) + M * glm::transpose(S) * glm::transpose(R_g);
    
    glm::mat2 scale_x_g = glm::mat2(0.f);
    scale_x_g[0][0] = 2.f * scales2d[idx].x;
    glm::mat2 scale_y_g = glm::mat2(0.f);
    scale_y_g[1][1] = 2.f * scales2d[idx].y;

    glm::mat2 sigma_x_g = R * scale_x_g * glm::transpose(R);
    glm::mat2 sigma_y_g = R * scale_y_g * glm::transpose(R);

    float G_11 = v_cov2d[idx].x; // dL/dSigma_11
    float G_12 = v_cov2d[idx].y; // dL/dSigma_12, which is the same as dL/dSigma_21
    float G_22 = v_cov2d[idx].z; // dL/dSigma_22

    v_scale[idx].x = G_11 * sigma_x_g[0][0] + 2 * G_12 * sigma_x_g[0][1] + G_22 * sigma_x_g[1][1];
    v_scale[idx].y = G_11 * sigma_y_g[0][0] + 2 * G_12 * sigma_y_g[0][1] + G_22 * sigma_y_g[1][1];
    v_rot[idx] = G_11 * theta_g[0][0] + 2 * G_12 * theta_g[0][1] + G_22 * theta_g[1][1];

    v_mean2d[idx].x = v_xy[idx].x * (0.5f * img_size.x);
    v_mean2d[idx].y = v_xy[idx].y * (0.5f * img_size.y);

}


