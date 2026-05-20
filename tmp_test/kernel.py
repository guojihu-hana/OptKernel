"""
Hand-crafted GEMM C = A @ B (FP32 in / FP32 out) for H200 (sm_90).

Custom CUDA only — no cuBLAS, no cuDNN, no library GEMM.

Strategy:
  1. Custom FP32 → FP16 cast kernel (bandwidth-bound).
  2. WMMA tensor-core GEMM (m16n16k16, FP16 in, FP32 accumulate).
  3. 2-stage cp.async pipeline (gmem → smem) with dynamic shared memory.
  4. Block tile 128 x 128, BK = 64, 8 warps in a 2x4 (M x N) layout, each
     warp owns a 64x32 sub-tile = 4x2 wmma fragments.
  5. Register-level software pipelining inside the K-tile: while issuing
     the mma_sync ops for kw=k we prefetch the WMMA fragments for kw=k+1.
"""

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ─── FP32 → FP16 cast kernel (bandwidth-bound) ────────────────────────────
__global__ void cast_f32_to_f16_kernel(
    const float* __restrict__ src, half* __restrict__ dst, int64_t n)
{
    int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step   = (int64_t)gridDim.x * blockDim.x;
    int64_t i      = tid * 8;
    int64_t stride = step * 8;
    for (; i + 7 < n; i += stride) {
        float4 a = *reinterpret_cast<const float4*>(src + i);
        float4 b = *reinterpret_cast<const float4*>(src + i + 4);
        half2 h0 = __floats2half2_rn(a.x, a.y);
        half2 h1 = __floats2half2_rn(a.z, a.w);
        half2 h2 = __floats2half2_rn(b.x, b.y);
        half2 h3 = __floats2half2_rn(b.z, b.w);
        uint4 packed = make_uint4(
            *reinterpret_cast<unsigned*>(&h0),
            *reinterpret_cast<unsigned*>(&h1),
            *reinterpret_cast<unsigned*>(&h2),
            *reinterpret_cast<unsigned*>(&h3));
        *reinterpret_cast<uint4*>(dst + i) = packed;
    }
    for (int64_t j = i; j < n; ++j)
        dst[j] = __float2half(src[j]);
}


// ─── GEMM tile config ──────────────────────────────────────────────────────
#define BM 128
#define BN 128
#define BK 64
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_PER_BLOCK 8
#define WARP_LAYOUT_M 2
#define WARP_LAYOUT_N 4
#define WARP_TILE_M (BM / WARP_LAYOUT_M)        // 64
#define WARP_TILE_N (BN / WARP_LAYOUT_N)        // 32
#define WMMA_PER_WARP_M (WARP_TILE_M / WMMA_M)  // 4
#define WMMA_PER_WARP_N (WARP_TILE_N / WMMA_N)  // 2
#define KW_PER_TILE (BK / WMMA_K)               // 4

#define A_PAD 8
#define B_PAD 8
#define A_STRIDE (BK + A_PAD)   // 72
#define B_STRIDE (BN + B_PAD)   // 136
#define STAGES 2

#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)  // 256

#define A_BYTES_PER_STAGE (BM * A_STRIDE * (int)sizeof(half))
#define B_BYTES_PER_STAGE (BK * B_STRIDE * (int)sizeof(half))
#define A_BYTES_TOTAL    (STAGES * A_BYTES_PER_STAGE)
#define B_BYTES_TOTAL    (STAGES * B_BYTES_PER_STAGE)
#define SMEM_BYTES_TOTAL (A_BYTES_TOTAL + B_BYTES_TOTAL)


// ─── cp.async helpers (sm_80+) ─────────────────────────────────────────────
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_int), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}


using AFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using BFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
using CFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;


__global__ __launch_bounds__(THREADS_PER_BLOCK, 2)
void gemm_wmma_v3(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m  = warp_id / WARP_LAYOUT_N;
    const int warp_n  = warp_id % WARP_LAYOUT_N;

    extern __shared__ char smem_raw[];
    half* As_base = reinterpret_cast<half*>(smem_raw);
    half* Bs_base = reinterpret_cast<half*>(smem_raw + A_BYTES_TOTAL);
    auto As_stage = [&] (int s) -> half* { return As_base + s * (BM * A_STRIDE); };
    auto Bs_stage = [&] (int s) -> half* { return Bs_base + s * (BK * B_STRIDE); };

    CFrag Cfrag[WMMA_PER_WARP_M][WMMA_PER_WARP_N];
    #pragma unroll
    for (int i = 0; i < WMMA_PER_WARP_M; ++i)
        #pragma unroll
        for (int j = 0; j < WMMA_PER_WARP_N; ++j)
            wmma::fill_fragment(Cfrag[i][j], 0.0f);

    const int A_row_base = by * BM;
    const int B_col_base = bx * BN;
    const int num_k_tiles = K / BK;

    // ─── Load helpers ────────────────────────────────────────────────────
    // A tile 128 rows x 64 cols (FP16) = 8192 halves. 16-byte (8h) chunks.
    //   8 chunks/row → 256 threads / 8 = 32 rows/iter → 4 outer iters
    // B tile  64 rows x 128 cols (FP16) = 8192 halves
    //   16 chunks/row → 256 threads / 16 = 16 rows/iter → 4 outer iters
    auto issue_load = [&] (int kt, int stage) {
        const int A_col_base = kt * BK;
        const int B_row_base = kt * BK;
        half* As_s = As_stage(stage);
        half* Bs_s = Bs_stage(stage);

        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx   = tid + it * THREADS_PER_BLOCK;
            int chunk = idx & 7;
            int row   = idx >> 3;
            int col   = chunk * 8;
            const half* gA = A + (A_row_base + row) * K + (A_col_base + col);
            half* sA = &As_s[row * A_STRIDE + col];
            cp_async_16(sA, gA);
        }
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx   = tid + it * THREADS_PER_BLOCK;
            int chunk = idx & 15;
            int row   = idx >> 4;
            int col   = chunk * 8;
            const half* gB = B + (B_row_base + row) * N + (B_col_base + col);
            half* sB = &Bs_s[row * B_STRIDE + col];
            cp_async_16(sB, gB);
        }
    };

    // ─── Compute one K-tile, software-pipelined kw loads ─────────────────
    // 2 register-buffered fragment sets; while we mma_sync for kw=k we
    // load_matrix_sync for kw=k+1 into the alt buffer.
    auto compute_stage = [&] (int stage) {
        AFrag Afrag[2][WMMA_PER_WARP_M];
        BFrag Bfrag[2][WMMA_PER_WARP_N];

        half* As_s = As_stage(stage);
        half* Bs_s = Bs_stage(stage);

        auto load_kw = [&] (int kw, AFrag a_out[WMMA_PER_WARP_M], BFrag b_out[WMMA_PER_WARP_N]) {
            #pragma unroll
            for (int i = 0; i < WMMA_PER_WARP_M; ++i) {
                int a_row = warp_m * WARP_TILE_M + i * WMMA_M;
                wmma::load_matrix_sync(
                    a_out[i],
                    &As_s[a_row * A_STRIDE + kw * WMMA_K],
                    A_STRIDE);
            }
            #pragma unroll
            for (int j = 0; j < WMMA_PER_WARP_N; ++j) {
                int b_col = warp_n * WARP_TILE_N + j * WMMA_N;
                wmma::load_matrix_sync(
                    b_out[j],
                    &Bs_s[(kw * WMMA_K) * B_STRIDE + b_col],
                    B_STRIDE);
            }
        };

        int cur = 0, nxt = 1;
        load_kw(0, Afrag[cur], Bfrag[cur]);

        #pragma unroll
        for (int kw = 0; kw < KW_PER_TILE; ++kw) {
            if (kw + 1 < KW_PER_TILE) {
                load_kw(kw + 1, Afrag[nxt], Bfrag[nxt]);
            }
            #pragma unroll
            for (int i = 0; i < WMMA_PER_WARP_M; ++i)
                #pragma unroll
                for (int j = 0; j < WMMA_PER_WARP_N; ++j)
                    wmma::mma_sync(Cfrag[i][j],
                                   Afrag[cur][i], Bfrag[cur][j],
                                   Cfrag[i][j]);
            int t = cur; cur = nxt; nxt = t;
        }
    };

    // ─── 2-stage cp.async pipeline (gmem → smem) ─────────────────────────
    issue_load(0, 0);
    cp_async_commit_group();
    cp_async_wait_group<0>();
    __syncthreads();

    int stage = 0;
    for (int kt = 0; kt < num_k_tiles - 1; ++kt) {
        issue_load(kt + 1, stage ^ 1);
        cp_async_commit_group();

        compute_stage(stage);

        cp_async_wait_group<0>();
        __syncthreads();
        stage ^= 1;
    }
    compute_stage(stage);

    // ─── Store C ─────────────────────────────────────────────────────────
    int C_row_base = by * BM + warp_m * WARP_TILE_M;
    int C_col_base = bx * BN + warp_n * WARP_TILE_N;
    #pragma unroll
    for (int i = 0; i < WMMA_PER_WARP_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WMMA_PER_WARP_N; ++j) {
            int c_row = C_row_base + i * WMMA_M;
            int c_col = C_col_base + j * WMMA_N;
            wmma::store_matrix_sync(
                C + c_row * N + c_col, Cfrag[i][j], N, wmma::mem_row_major);
        }
    }
}


// ─── Host launcher ─────────────────────────────────────────────────────────
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A,B must be on CUDA");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    A = A.contiguous();
    B = B.contiguous();
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "K mismatch");
    TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0,
                "shapes must be multiples of tile sizes");

    auto A_h = torch::empty({M, K}, A.options().dtype(torch::kHalf));
    auto B_h = torch::empty({K, N}, B.options().dtype(torch::kHalf));
    auto C   = torch::empty({M, N}, A.options());

    {
        int64_t nA = (int64_t)M * K;
        int64_t nB = (int64_t)K * N;
        int block = 256;
        int grid_a = (int)std::min<int64_t>(132 * 8, (nA + block * 8 - 1) / (block * 8));
        int grid_b = (int)std::min<int64_t>(132 * 8, (nB + block * 8 - 1) / (block * 8));
        cast_f32_to_f16_kernel<<<grid_a, block>>>(
            A.data_ptr<float>(), reinterpret_cast<half*>(A_h.data_ptr<at::Half>()), nA);
        cast_f32_to_f16_kernel<<<grid_b, block>>>(
            B.data_ptr<float>(), reinterpret_cast<half*>(B_h.data_ptr<at::Half>()), nB);
    }

    static bool s_smem_set = false;
    if (!s_smem_set) {
        auto err = cudaFuncSetAttribute(
            (const void*)gemm_wmma_v3,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_BYTES_TOTAL);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
        s_smem_set = true;
    }

    dim3 grid(N / BN, M / BM);
    dim3 block(THREADS_PER_BLOCK);
    gemm_wmma_v3<<<grid, block, SMEM_BYTES_TOTAL>>>(
        reinterpret_cast<const half*>(A_h.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_h.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        M, N, K);
    return C;
}
"""

CPP_SRC = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

_module = load_inline(
    name="custom_gemm_final",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=["matmul_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "-arch=sm_90", "--use_fast_math",
                       "-Xcompiler=-fno-strict-aliasing",
                       "--expt-relaxed-constexpr"],
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _module.matmul_cuda(A, B)
