/*
 * turbo-quant.cu — TurboQuant turbo3 CUDA kernels for NVIDIA GPUs
 *
 * Part of the turbo3 CUDA port for llama.cpp.
 * Implements dequantize (contiguous + non-contiguous), FWHT rotation kernel,
 * and SET_ROWS quantize kernel (normalize → WHT rotate → 3-bit pack) for
 * GGML_TYPE_TURBO3_0 KV cache compression.
 *
 * Block format: 14 bytes per 32 values = 3.5 bits/value = 4.6× vs fp16.
 * WHT rotation is at graph level via GGML_OP_TURBO_WHT (128-element groups).
 *
 * Based on TheTom's Metal implementation (llama-cpp-turboquant)
 * and the TurboQuant paper (arXiv:2504.19874, ICLR 2026).
 *
 * Author: Erol Germain (@erolgermain)
 * Date:   March 2026
 * License: MIT (matching upstream llama.cpp)
 */

#include "common.cuh"
#include "ggml-common.h"
#ifdef GGML_USE_HIP
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

// Lloyd-Max codebooks for d=128 rotation groups.
// Since FWHT groups are always 128 elements regardless of head_dim,
// these codebooks are correct for all head dimensions.
static __constant__ float TURBO3_CENTROIDS_C[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static __constant__ float TURBO3_MIDPOINTS_C[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

static __constant__ float TURBO2_CENTROIDS_C[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __constant__ float TURBO2_MIDPOINTS_C[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// QJL sign arrays for turbo4 cross-space residual (seed=1042)
static __constant__ float d_turbo_qjl_signs1[128] = {
     1,-1,-1,-1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,
     1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1, 1,
     1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,
    -1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,
    -1,-1, 1, 1, 1, 1,-1, 1, 1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1, 1, 1,-1,
    -1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1
};

static __constant__ float d_turbo_qjl_signs2[128] = {
     1, 1,-1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1,
    -1,-1,-1, 1,-1, 1, 1, 1,-1, 1,-1,-1,-1,-1, 1, 1,
    -1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,
     1, 1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1, 1,
    -1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1,-1,
     1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1,
     1,-1,-1, 1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1, 1,
    -1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1
};

// ═══════════════════════════════════════════════════════════════════════
//  Turbo4 dequantize kernels
// ═══════════════════════════════════════════════════════════════════════

static __device__ __forceinline__ uint8_t turbo4_unpack_3bit(const uint8_t * qs, int j) {
    int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
    uint16_t raw = (uint16_t)qs[byte_idx];
    if (byte_idx + 1 < 48) raw |= (uint16_t)qs[byte_idx + 1] << 8;
    return (uint8_t)((raw >> bit_pos) & 0x7);
}

template<typename dst_t>
static __global__ void dequantize_block_turbo4_0_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    const int64_t blk_idx = i / QK_TURBO4;
    const int     elem    = (int)(i % QK_TURBO4);
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx + blk_idx;

    const float norm = __half2float(x->norm);
    const float rnorm = __half2float(x->rnorm);
    const float qjl_scale = 1.2533141f / 128.0f * rnorm;

    uint8_t idx = turbo4_unpack_3bit(x->qs, elem);
    float s = (x->signs[elem / 8] & (1 << (elem % 8))) ? 1.0f : -1.0f;

    y[i] = (dst_t)((TURBO3_CENTROIDS_C[idx] + s * qjl_scale) * norm);
}

template<typename dst_t>
static __global__ void dequantize_block_turbo3_0_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    const int64_t blk_idx = i / QK_TURBO3;
    const int     elem    = (int)(i % QK_TURBO3);
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx + blk_idx;

    const float norm = __half2float(x->norm);
    uint8_t low2 = (x->qs[elem >> 2] >> ((elem & 3) << 1)) & 0x3;
    uint8_t hi1  = (x->signs[elem >> 3] >> (elem & 7)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);

    y[i] = (dst_t)(TURBO3_CENTROIDS_C[idx] * norm);
}

void dequantize_row_turbo3_0_fp16_cuda(
    const void * vx, half * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo3_0_kernel<half><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo3_0_fp32_cuda(
    const void * vx, float * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo3_0_kernel<float><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo3_0_bf16_cuda(
    const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo3_0_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo4_0_fp16_cuda(
    const void * vx, half * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo4_0_kernel<half><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo4_0_fp32_cuda(
    const void * vx, float * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo4_0_kernel<float><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo4_0_bf16_cuda(
    const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo4_0_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(vx, y, k);
}

// Non-contiguous dequant (for nc dispatch tables)
template<typename dst_t>
static __global__ void dequantize_block_turbo3_0_nc_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01,  const int64_t s02,  const int64_t s03
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne00 * ne01 * ne02 * ne03) return;

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i / (ne00 * ne01)) % ne02;
    const int64_t i01 = (i / ne00) % ne01;
    const int64_t i00 = i % ne00;

    const int64_t src_offset = i01*s01 + i02*s02 + i03*s03;
    const char * src = (const char *)vx + src_offset;

    const int64_t blk_idx = i00 / QK_TURBO3;
    const int     elem    = (int)(i00 % QK_TURBO3);
    const block_turbo3_0 * x = (const block_turbo3_0 *)src + blk_idx;

    const float norm = __half2float(x->norm);
    uint8_t low2 = (x->qs[elem >> 2] >> ((elem & 3) << 1)) & 0x3;
    uint8_t hi1  = (x->signs[elem >> 3] >> (elem & 7)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);

    y[i] = (dst_t)(TURBO3_CENTROIDS_C[idx] * norm);
}

void dequantize_row_turbo3_0_fp16_nc_cuda(
    const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo3_0_nc_kernel<half><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

void dequantize_row_turbo3_0_fp32_nc_cuda(
    const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo3_0_nc_kernel<float><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

void dequantize_row_turbo3_0_bf16_nc_cuda(
    const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo3_0_nc_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

// ═══════════════════════════════════════════════════════════════════════
//  Turbo2 dequantize kernels — 2-bit, 4 centroids, no sign byte
// ═══════════════════════════════════════════════════════════════════════

template<typename dst_t>
static __global__ void dequantize_block_turbo2_0_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;

    const int64_t blk_idx = i / QK_TURBO2;
    const int     elem    = (int)(i % QK_TURBO2);
    const block_turbo2_0 * x = (const block_turbo2_0 *)vx + blk_idx;

    const float norm = __half2float(x->norm);
    uint8_t idx = (x->qs[elem >> 2] >> ((elem & 3) << 1)) & 0x3;

    y[i] = (dst_t)(TURBO2_CENTROIDS_C[idx] * norm);
}

void dequantize_row_turbo2_0_fp16_cuda(
    const void * vx, half * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo2_0_kernel<half><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo2_0_fp32_cuda(
    const void * vx, float * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo2_0_kernel<float><<<blocks, threads, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo2_0_bf16_cuda(
    const void * vx, nv_bfloat16 * y, int64_t k, cudaStream_t stream
) {
    const int threads = 256;
    const int blocks  = (k + threads - 1) / threads;
    dequantize_block_turbo2_0_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(vx, y, k);
}

// Non-contiguous dequant for turbo2
template<typename dst_t>
static __global__ void dequantize_block_turbo2_0_nc_kernel(
    const void * __restrict__ vx,
    dst_t      * __restrict__ y,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t s01,  const int64_t s02,  const int64_t s03
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne00 * ne01 * ne02 * ne03) return;

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i / (ne00 * ne01)) % ne02;
    const int64_t i01 = (i / ne00) % ne01;
    const int64_t i00 = i % ne00;

    const int64_t src_offset = i01*s01 + i02*s02 + i03*s03;
    const char * src = (const char *)vx + src_offset;

    const int64_t blk_idx = i00 / QK_TURBO2;
    const int     elem    = (int)(i00 % QK_TURBO2);
    const block_turbo2_0 * x = (const block_turbo2_0 *)src + blk_idx;

    const float norm = __half2float(x->norm);
    uint8_t idx = (x->qs[elem >> 2] >> ((elem & 3) << 1)) & 0x3;

    y[i] = (dst_t)(TURBO2_CENTROIDS_C[idx] * norm);
}

void dequantize_row_turbo2_0_fp16_nc_cuda(
    const void * vx, half * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo2_0_nc_kernel<half><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

void dequantize_row_turbo2_0_fp32_nc_cuda(
    const void * vx, float * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo2_0_nc_kernel<float><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

void dequantize_row_turbo2_0_bf16_nc_cuda(
    const void * vx, nv_bfloat16 * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream
) {
    const int64_t total = ne00 * ne01 * ne02 * ne03;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    dequantize_block_turbo2_0_nc_kernel<nv_bfloat16><<<blocks, threads, 0, stream>>>(
        vx, y, ne00, ne01, ne02, ne03, s01, s02, s03);
}

// ═══════════════════════════════════════════════════════════════════════
//  GGML_OP_TURBO_WHT — Fast Walsh-Hadamard Transform with sign rotation
//
//  Ported from Metal kernel_turbo_wht (ggml-metal.metal line 3018)
//  and CPU ggml_compute_forward_turbo_wht_f32 (ops.cpp line 10594).
//
//  Each CUDA thread processes one 128-element group (256 threads/block).
//  direction=0: forward (signs1 -> FWHT -> signs2)
//  direction=1: inverse (signs2 -> FWHT -> signs1)
// ═══════════════════════════════════════════════════════════════════════

static __constant__ float d_turbo_wht_signs1[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,
     1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,
     1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,
     1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1,
     1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1
};

static __constant__ float d_turbo_wht_signs2[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1,
     1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,
    -1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1
};

// V-specific FWHT signs: independent rotation for value tensors.
// Generated from seed=12345 (K signs use seed=42). Truly independent rotation.
static __constant__ float d_turbo_wht_signs1_v[128] = {
     1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1,
     1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1,
    -1, 1,-1, 1,-1,-1, 1,-1, 1,-1,-1,-1, 1,-1,-1, 1,
     1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,
     1,-1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1, 1, 1,-1, 1,
    -1,-1,-1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,
     1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,
    -1,-1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1
};

static __constant__ float d_turbo_wht_signs2_v[128] = {
    -1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1,-1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,
     1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,
    -1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1, 1,-1,-1,
    -1, 1,-1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,
    -1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,
     1, 1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1,
     1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1,-1,-1, 1
};

// Dense rotation matrix for K (loaded from turbo-rotation-data.h at init)
static float * d_rotation_k = nullptr;  // 128*128 floats on device
static float * d_rotation_k_inv = nullptr;
// V uses different rotation (different seed)
static float * d_rotation_v = nullptr;
static float * d_rotation_v_inv = nullptr;

// Include the precomputed rotation matrix
#include "../../src/turbo-rotation-data.h"

static void turbo_init_rotation_device() {
    if (d_rotation_k != nullptr) return;
    // TURBO_ROTATION_RT is R^T (128x128), stored row-major in turbo-rotation-data.h
    // For forward rotation y = R*x, we need R. Since R^T is stored, R = transpose(R^T).
    // But for matrix multiply y[i] = sum_j R[i][j] * x[j], and R^T is stored row-major,
    // R^T[i][j] = R[j][i], so R[i][j] = R^T[j][i]. We can just use R^T with transposed indexing.
    // Simpler: store R^T as-is, and compute y = R^T^T * x = R * x by reading columns of R^T.
    //
    // Actually: just upload R^T and use it. The kernel will compute y[i] = sum_j RT[j][i] * x[j]
    // which equals y = R * x where R = (R^T)^T.
    hipMalloc(&d_rotation_k, 128*128*sizeof(float));
    hipMemcpy(d_rotation_k, TURBO_ROTATION_RT, 128*128*sizeof(float), hipMemcpyHostToDevice);

    // For inverse: y = R^T * x, just use R^T directly: y[i] = sum_j RT[i][j] * x[j]
    hipMalloc(&d_rotation_k_inv, 128*128*sizeof(float));
    hipMemcpy(d_rotation_k_inv, TURBO_ROTATION_RT, 128*128*sizeof(float), hipMemcpyHostToDevice);

    // TODO: V rotation uses different seed — for now use same as K
    d_rotation_v = d_rotation_k;
    d_rotation_v_inv = d_rotation_k_inv;
}

// Dense rotation kernel: y[i] = sum_j R[i][j] * x[j]
// R is stored as R^T row-major, so R[i][j] = RT[j][i] = RT[j*128+i]
__launch_bounds__(1, 1)
static __global__ void kernel_turbo_dense_rotate(
    const float * __restrict__ src,
    float * __restrict__ dst,
    const float * __restrict__ RT,  // R^T stored row-major (128x128)
    const int64_t n_elements,
    const int transpose  // 0: y=R*x (forward), 1: y=R^T*x (inverse)
) {
    const int64_t group_idx = blockIdx.x;
    const int64_t n_groups = n_elements / 128;
    if (group_idx >= n_groups) return;

    const float * in = src + group_idx * 128;
    float * out = dst + group_idx * 128;

    for (int i = 0; i < 128; i++) {
        float sum = 0;
        for (int j = 0; j < 128; j++) {
            // transpose=0: y[i] = sum R[i][j]*x[j] = sum RT[j*128+i]*x[j]
            // transpose=1: y[i] = sum RT[i][j]*x[j] = sum RT[i*128+j]*x[j]
            float r = transpose ? RT[i*128+j] : RT[j*128+i];
            sum += r * in[j];
        }
        out[i] = sum;
    }
}

// FWHT via shared memory — one thread per block, avoids HIP compiler bug on gfx906
__launch_bounds__(1, 1)
static __global__ void kernel_turbo_wht(
    const float * __restrict__ src,
    float       * __restrict__ dst,
    const int64_t n_elements,
    const int     direction
) {
    __shared__ float x[128];

    const int64_t group_idx = blockIdx.x;
    const int64_t n_groups = n_elements / 128;
    if (group_idx >= n_groups) return;

    const float * in  = src + group_idx * 128;
    float       * out = dst + group_idx * 128;

    // direction 0: forward rotation for Q (uses K signs: signs1 -> signs2)
    // direction 1: inverse rotation for V output (uses V signs: signs2_v -> signs1_v)
    const float * s_first  = (direction == 0) ? d_turbo_wht_signs1   : d_turbo_wht_signs2_v;
    const float * s_second = (direction == 0) ? d_turbo_wht_signs2   : d_turbo_wht_signs1_v;

    for (int i = 0; i < 128; i++) x[i] = in[i] * s_first[i];

    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) out[i] = x[i] * inv_sqrt_128 * s_second[i];
}

// ═══════════════════════════════════════════════════════════════════════
//  SET_ROWS for turbo3 — custom kernel that groups 128 elements,
//  normalizes, rotates via WHT, then quantizes into 4 turbo3 blocks.
//
//  Each thread processes one 128-element group.
// ═══════════════════════════════════════════════════════════════════════

static __constant__ float TURBO3_MIDPOINTS_QC[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// FWHT via shared memory — avoids HIP compiler optimization bug on gfx906
// One thread per block, FWHT buffer in shared memory (not registers)
__launch_bounds__(1, 1)
static __global__ void kernel_set_rows_turbo3(
    const float * __restrict__ src0,
    const int64_t * __restrict__ src1,
    block_turbo3_0 * __restrict__ dst,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t nb01,
    const int64_t nb1,
    const int n_groups_per_row,
    const int use_v_signs
) {
    __shared__ float x[128];

    const float * wht_signs1 = use_v_signs ? d_turbo_wht_signs1_v : d_turbo_wht_signs1;
    const float * wht_signs2 = use_v_signs ? d_turbo_wht_signs2_v : d_turbo_wht_signs2;

    const int64_t row = blockIdx.x;
    if (row >= ne01) return;

    const int grp_idx = blockIdx.y;
    if (grp_idx >= n_groups_per_row) return;

    const float * src_row = (const float *)((const char *)src0 + row * nb01);
    const int64_t dst_row_idx = src1[row];
    block_turbo3_0 * dst_row = (block_turbo3_0 *)((char *)dst + dst_row_idx * nb1);

    const float * grp_src = src_row + grp_idx * 128;

    // Step 1: Compute 128-element group norm
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += grp_src[j] * grp_src[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    // Dense rotation: x = R * (src / norm)
    // R^T is passed via use_v_signs overloaded as pointer index
    // Actually we can't easily pass the matrix pointer to this kernel,
    // so we use the FWHT approach but this time it should work since
    // we verified FWHT is correct in standalone tests.
    // The issue must be elsewhere — let's re-enable FWHT.
    for (int i = 0; i < 128; i++) x[i] = grp_src[i] * inv_norm * wht_signs1[i];
    for (int h = 1; h < 128; h *= 2)
        for (int i = 0; i < 128; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j+h];
                x[j] = a+b; x[j+h] = a-b;
            }
    float inv_sqrt_128 = 1.0f / sqrtf(128.0f);
    for (int i = 0; i < 128; i++) x[i] = x[i] * inv_sqrt_128 * wht_signs2[i];

    // Step 3: Quantize into 4 blocks of 32, accumulating reconstruction norm
    // Inline centroids/midpoints to avoid __constant__ memory issues on HIP
    const float C[8] = {
        -0.190685f, -0.117832f, -0.065717f, -0.021460f,
         0.021460f,  0.065717f,  0.117832f,  0.190685f
    };
    const float M[7] = {
        -0.154259f, -0.091775f, -0.043589f, 0.0f,
         0.043589f,  0.091775f,  0.154259f
    };

    float recon_norm_sq = 0.0f;

    for (int b = 0; b < 4; b++) {
        block_turbo3_0 * blk = &dst_row[grp_idx * 4 + b];
        const int off = b * 32;

        // Clear packed bytes
        for (int j = 0; j < 8; j++) blk->qs[j] = 0;
        for (int j = 0; j < 4; j++) blk->signs[j] = 0;

        for (int j = 0; j < 32; j++) {
            float rv = x[off + j];

            // Nearest centroid via midpoints
            uint8_t idx = 0;
            idx += (rv >= M[0]);
            idx += (rv >= M[1]);
            idx += (rv >= M[2]);
            idx += (rv >= M[3]);
            idx += (rv >= M[4]);
            idx += (rv >= M[5]);
            idx += (rv >= M[6]);

            // Pack lower 2 bits
            blk->qs[j >> 2] |= ((idx & 0x3) << ((j & 3) << 1));

            // Pack upper 1 bit
            if (idx & 0x4) {
                blk->signs[j >> 3] |= (1u << (j & 7));
            }

            recon_norm_sq += C[idx] * C[idx];
        }
    }

    // Norm correction: store corrected norm so dequant produces vectors with exact original L2 norm.
    // Codebook quantization systematically shrinks reconstruction norm; this corrects the bias.
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    for (int b = 0; b < 4; b++) {
        dst_row[grp_idx * 4 + b].norm = __float2half(corrected_norm);
    }
}

// Detect whether a tensor is a V cache (name contains "cache_v")
static bool turbo_is_v_tensor(const ggml_tensor * t) {
    // Check view_src name if this is a view, otherwise check tensor name
    const ggml_tensor * src = t->view_src ? t->view_src : t;
    return src->name[0] != '\0' && strstr(src->name, "cache_v") != nullptr;
}

void ggml_cuda_op_set_rows_turbo3(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;
    block_turbo3_0 * dst_d = (block_turbo3_0 *)dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb1  = dst->nb[1];

    GGML_ASSERT(ne00 % 128 == 0);
    const int n_groups_per_row = ne00 / 128;

    // 1 thread per block, 1 block per group (avoids register pressure issues on gfx906)
    dim3 grid(ne01, n_groups_per_row);
    dim3 block(1);

    const int is_v = turbo_is_v_tensor(dst) ? 1 : 0;

    kernel_set_rows_turbo3<<<grid, block, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d, ne00, ne01, nb01, nb1, n_groups_per_row, is_v);
}

// ═══════════════════════════════════════════════════════════════════════
//  SET_ROWS for turbo4 — quantize with QJL cross-space residual
//  Each thread processes one 128-element group (QK_TURBO4 = 128).
// ═══════════════════════════════════════════════════════════════════════

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float inv = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv;
}

__launch_bounds__(256, 1)
static __global__ void kernel_set_rows_turbo4(
    const float * __restrict__ src0,
    const int64_t * __restrict__ src1,
    block_turbo4_0 * __restrict__ dst,
    const int64_t ne00, const int64_t ne01,
    const int64_t nb01, const int64_t nb1,
    const int n_groups_per_row,
    const int use_v_signs
) {
    const float * wht_signs1 = use_v_signs ? d_turbo_wht_signs1_v : d_turbo_wht_signs1;
    const float * wht_signs2 = use_v_signs ? d_turbo_wht_signs2_v : d_turbo_wht_signs2;
    const int64_t row = blockIdx.x;
    if (row >= ne01) return;
    const int grp_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (grp_idx >= n_groups_per_row) return;

    const float * src_row = (const float *)((const char *)src0 + row * nb01);
    const int64_t dst_row_idx = src1[row];
    block_turbo4_0 * dst_blk = (block_turbo4_0 *)((char *)dst + dst_row_idx * nb1) + grp_idx;

    const float * grp_src = src_row + grp_idx * 128;

    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += grp_src[j] * grp_src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;

    // Forward FWHT rotation (K or V specific signs)
    for (int i = 0; i < 128; i++) x[i] *= wht_signs1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= wht_signs2[i];

    // 3-bit quantize
    for (int j = 0; j < 48; j++) dst_blk->qs[j] = 0;
    for (int j = 0; j < 16; j++) dst_blk->signs[j] = 0;

    float recon[128];
    for (int j = 0; j < 128; j++) {
        uint8_t idx = 0;
        idx += (x[j] >= TURBO3_MIDPOINTS_C[0]);
        idx += (x[j] >= TURBO3_MIDPOINTS_C[1]);
        idx += (x[j] >= TURBO3_MIDPOINTS_C[2]);
        idx += (x[j] >= TURBO3_MIDPOINTS_C[3]);
        idx += (x[j] >= TURBO3_MIDPOINTS_C[4]);
        idx += (x[j] >= TURBO3_MIDPOINTS_C[5]);
        idx += (x[j] >= TURBO3_MIDPOINTS_C[6]);

        recon[j] = TURBO3_CENTROIDS_C[idx];
        int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
        dst_blk->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < 48)
            dst_blk->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
    }

    // Original-space residual for QJL (per paper Algorithm 2):
    // 1. Inverse-rotate the reconstruction back to original space
    // 2. Compute residual = normalized_original - recon_original
    float recon_orig[128];
    for (int j = 0; j < 128; j++) recon_orig[j] = recon[j];

    // Inverse FWHT: apply signs2, inverse butterfly, apply signs1, normalize
    for (int i = 0; i < 128; i++) recon_orig[i] *= wht_signs2[i];
    // Inverse butterfly = same butterfly (FWHT is self-inverse up to scale)
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int jj = i; jj < i + h; jj++) {
                float a = recon_orig[jj], b = recon_orig[jj + h];
                recon_orig[jj] = a + b; recon_orig[jj + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128_qjl = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) recon_orig[i] = recon_orig[i] * inv_sqrt_128_qjl * wht_signs1[i];

    // Residual in ORIGINAL space (grp_src * inv_norm was the normalized original)
    // We saved the original normalized values before rotation — but they were
    // overwritten by x[]. Recompute from grp_src:
    float residual[128];
    float rnorm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        float normalized_j = grp_src[j] * inv_norm;
        residual[j] = normalized_j - recon_orig[j];
        rnorm_sq += residual[j] * residual[j];
    }
    float rnorm = sqrtf(rnorm_sq);
    dst_blk->rnorm = __float2half(rnorm);

    // QJL projection of original-space residual (using QJL-specific FWHT signs)
    for (int i = 0; i < 128; i++) residual[i] *= d_turbo_qjl_signs1[i];
    turbo_fwht_128(residual);
    for (int i = 0; i < 128; i++) residual[i] *= d_turbo_qjl_signs2[i];
    for (int j = 0; j < 128; j++) {
        if (residual[j] >= 0.0f) dst_blk->signs[j / 8] |= (1 << (j % 8));
    }

    // Norm correction: MSE-only (centroid reconstruction norm, no QJL).
    // QJL correction is applied at score level, not fused into dequanted values.
    // The signs[] and rnorm fields store QJL data for future score-level correction.
    float recon_norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        recon_norm_sq += recon[j] * recon[j];
    }
    float recon_norm = sqrtf(recon_norm_sq);
    dst_blk->norm = __float2half((recon_norm > 1e-10f) ? norm / recon_norm : norm);
}

void ggml_cuda_op_set_rows_turbo4(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;
    block_turbo4_0 * dst_d = (block_turbo4_0 *)dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb1  = dst->nb[1];

    GGML_ASSERT(ne00 % 128 == 0);
    const int n_groups_per_row = ne00 / 128;

    const int threads = 32;
    const int grp_blocks = (n_groups_per_row + threads - 1) / threads;

    dim3 grid(ne01, grp_blocks);
    dim3 block(threads);

    const int is_v = turbo_is_v_tensor(dst) ? 1 : 0;

    kernel_set_rows_turbo4<<<grid, block, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d, ne00, ne01, nb01, nb1, n_groups_per_row, is_v);
}

// ═══════════════════════════════════════════════════════════════════════
//  SET_ROWS for turbo2 — 2-bit quantization with FWHT rotation
//  Same rotation as turbo3, but quantizes to 4 centroids with 2-bit packing.
//  Block: 10 bytes per 32 values (norm fp16 + qs[8]).
//  No signs field needed — all 2 bits stored in qs.
// ═══════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 1)
static __global__ void kernel_set_rows_turbo2(
    const float * __restrict__ src0,
    const int64_t * __restrict__ src1,
    block_turbo2_0 * __restrict__ dst,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t nb01,
    const int64_t nb1,
    const int n_groups_per_row,
    const int use_v_signs
) {
    const float * wht_signs1 = use_v_signs ? d_turbo_wht_signs1_v : d_turbo_wht_signs1;
    const float * wht_signs2 = use_v_signs ? d_turbo_wht_signs2_v : d_turbo_wht_signs2;
    const int64_t row = blockIdx.x;
    if (row >= ne01) return;

    const int grp_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (grp_idx >= n_groups_per_row) return;

    const float * src_row = (const float *)((const char *)src0 + row * nb01);
    const int64_t dst_row_idx = src1[row];
    block_turbo2_0 * dst_row = (block_turbo2_0 *)((char *)dst + dst_row_idx * nb1);

    const float * grp_src = src_row + grp_idx * 128;

    // Step 1: Compute 128-element group norm
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += grp_src[j] * grp_src[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    // Step 2: Normalize and rotate
    float x[128];
    for (int i = 0; i < 128; i++) x[i] = grp_src[i] * inv_norm;

    for (int i = 0; i < 128; i++) x[i] *= wht_signs1[i];

    // FWHT butterfly (7 stages for 128 elements)
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // Normalize and apply signs2
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] = x[i] * inv_sqrt_128 * wht_signs2[i];

    // Step 3: Quantize into 4 blocks of 32, accumulating reconstruction norm
    float recon_norm_sq = 0.0f;

    for (int b = 0; b < 4; b++) {
        block_turbo2_0 * blk = &dst_row[grp_idx * 4 + b];
        const int off = b * 32;

        // Clear packed bytes
        for (int j = 0; j < 8; j++) blk->qs[j] = 0;

        for (int j = 0; j < 32; j++) {
            float rv = x[off + j];

            // Nearest centroid via midpoints (4 centroids, 3 midpoints)
            uint8_t idx = 0;
            idx += (rv >= TURBO2_MIDPOINTS_C[0]);
            idx += (rv >= TURBO2_MIDPOINTS_C[1]);
            idx += (rv >= TURBO2_MIDPOINTS_C[2]);

            // Pack 2-bit index
            blk->qs[j >> 2] |= ((idx & 0x3) << ((j & 3) << 1));

            recon_norm_sq += TURBO2_CENTROIDS_C[idx] * TURBO2_CENTROIDS_C[idx];
        }
    }

    // Norm correction
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    for (int b = 0; b < 4; b++) {
        dst_row[grp_idx * 4 + b].norm = __float2half(corrected_norm);
    }
}

void ggml_cuda_op_set_rows_turbo2(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;
    block_turbo2_0 * dst_d = (block_turbo2_0 *)dst->data;

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb1  = dst->nb[1];

    GGML_ASSERT(ne00 % 128 == 0);
    const int n_groups_per_row = ne00 / 128;

    const int threads = 32;
    const int grp_blocks = (n_groups_per_row + threads - 1) / threads;

    dim3 grid(ne01, grp_blocks);
    dim3 block(threads);

    const int is_v = turbo_is_v_tensor(dst) ? 1 : 0;

    kernel_set_rows_turbo2<<<grid, block, 0, ctx.stream()>>>(
        src0_d, src1_d, dst_d, ne00, ne01, nb01, nb1, n_groups_per_row, is_v);
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const float * src_data = (const float *)src0->data;
    float * dst_data = (float *)dst->data;

    int32_t direction;
    memcpy(&direction, dst->op_params, sizeof(int32_t));

    const int64_t n_elements = ggml_nelements(src0);
    GGML_ASSERT(n_elements % 128 == 0);

    const int64_t n_groups = n_elements / 128;

    // FWHT kernel (verified correct in standalone tests)
    kernel_turbo_wht<<<(int)n_groups, 1, 0, ctx.stream()>>>(
        src_data, dst_data, n_elements, direction);
}
