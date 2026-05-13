import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

N = 4096

cpp_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" void triu_matmul_kernel_launcher(
    const float* A, const float* B, float* C, int N);

torch::Tensor triu_matmul_launcher(torch::Tensor A, torch::Tensor B) {
    int N_dim = A.size(0);
    auto C = torch::empty({N_dim, N_dim}, A.options());
    A = A.contiguous();
    B = B.contiguous();
    triu_matmul_kernel_launcher(
        A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), N_dim);
    return C;
}
"""

cuda_source = r"""
#include <cuda_runtime.h>
#include <algorithm>
using std::min;

// Tile dimensions
#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8

// Padding to avoid bank conflicts
#define PAD_A 8            // AS_STRIDE = 40, not multiple of 32
#define PAD_B 8            // BS_STRIDE = 136, not multiple of 32
#define AS_STRIDE (BK + PAD_A)   // 40
#define BS_STRIDE (BN + PAD_B)   // 136

// Inline PTX helpers for cp.async (Hopper SM90)
__device__ inline void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

__device__ inline void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ inline void cp_async_float4(float* __restrict__ shared_dst,
                                       const float* __restrict__ global_src) {
    uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(dst_addr), "l"(global_src) : "memory");
}

// Main kernel
__global__ void triu_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    // Dynamic shared memory layout: [As0, As1, Bs0, Bs1]
    extern __shared__ float sh[];
    const int as_buf_size = BM * AS_STRIDE;   // 128 x 40 = 5120
    const int bs_buf_size = BK * BS_STRIDE;   // 32 x 136 = 4352

    float* As0 = sh;
    float* As1 = As0 + as_buf_size;
    float* Bs0 = As1 + as_buf_size;
    float* Bs1 = Bs0 + bs_buf_size;

    // 1D grid -> upper‑triangular tile (by, bx) with by <= bx
    int M_blk = (N + BM - 1) / BM;
    int bid = blockIdx.x;
    int by = 0, bx = 0;
    {
        int rem = bid;
        for (int r = 0; r < M_blk; ++r) {
            int cnt = M_blk - r;
            if (rem < cnt) {
                by = r;
                bx = r + rem;
                break;
            }
            rem -= cnt;
        }
    }

    int row_start = by * BM;
    int col_start = bx * BN;
    int col_end   = min(col_start + BN - 1, N - 1);

    if (row_start > col_end) return;

    int tx = threadIdx.x;          // 0..15
    int ty = threadIdx.y;          // 0..15
    int tid = ty * blockDim.x + tx; // 0..255
    int total_threads = 256;

    int k_start = row_start;
    int k_end   = col_end;

    float sum[TM][TN] = {{0.0f}};

    // Double‑buffer pointers
    float* A_cur = As0;
    float* B_cur = Bs0;
    float* A_next = As1;
    float* B_next = Bs1;

    // Prologue: load first K‑tile into As0 / Bs0 with cp.async
    {
        int k_cur = k_start;

        // Load tile A (BM x BK)
        for (int i = tid * 4; i < BM * BK; i += total_threads * 4) {
            int row = i / BK;
            int col = i % BK;
            int g_row = row_start + row;
            int g_col = k_cur + col;

            if (g_row < N && g_col + 3 < N) {
                cp_async_float4(&A_cur[row * AS_STRIDE + col],
                                &A[g_row * N + g_col]);
            } else {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int gc = g_col + c;
                    float val = (g_row < N && gc < N) ? A[g_row * N + gc] : 0.0f;
                    A_cur[row * AS_STRIDE + col + c] = val;
                }
            }
        }

        // Load tile B (BK x BN)
        for (int i = tid * 4; i < BK * BN; i += total_threads * 4) {
            int row = i / BN;
            int col = i % BN;
            int g_row = k_cur + row;
            int g_col = col_start + col;

            if (g_row < N && g_col + 3 < N) {
                cp_async_float4(&B_cur[row * BS_STRIDE + col],
                                &B[g_row * N + g_col]);
            } else {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    int gc = g_col + c;
                    float val = (g_row < N && gc < N) ? B[g_row * N + gc] : 0.0f;
                    B_cur[row * BS_STRIDE + col + c] = val;
                }
            }
        }

        cp_async_commit_group();
        cp_async_wait_all();
        __syncthreads();
    }

    // Main K‑loop with double buffering and async prefetching
    for (int kk = k_start; kk <= k_end; kk += BK) {
        int next_kk = kk + BK;
        bool has_next = (next_kk <= k_end);

        // Launch async loads for the next tile (if any)
        if (has_next) {
            // A tile
            for (int i = tid * 4; i < BM * BK; i += total_threads * 4) {
                int row = i / BK;
                int col = i % BK;
                int g_row = row_start + row;
                int g_col = next_kk + col;

                if (g_row < N && g_col + 3 < N) {
                    cp_async_float4(&A_next[row * AS_STRIDE + col],
                                    &A[g_row * N + g_col]);
                } else {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int gc = g_col + c;
                        float val = (g_row < N && gc < N) ? A[g_row * N + gc] : 0.0f;
                        A_next[row * AS_STRIDE + col + c] = val;
                    }
                }
            }

            // B tile
            for (int i = tid * 4; i < BK * BN; i += total_threads * 4) {
                int row = i / BN;
                int col = i % BN;
                int g_row = next_kk + row;
                int g_col = col_start + col;

                if (g_row < N && g_col + 3 < N) {
                    cp_async_float4(&B_next[row * BS_STRIDE + col],
                                    &B[g_row * N + g_col]);
                } else {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int gc = g_col + c;
                        float val = (g_row < N && gc < N) ? B[g_row * N + gc] : 0.0f;
                        B_next[row * BS_STRIDE + col + c] = val;
                    }
                }
            }

            cp_async_commit_group();
        }

        // Compute on current buffers
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float4 B4 = *reinterpret_cast<const float4*>(&B_cur[k * BS_STRIDE + tx * TN]);
            float b0 = B4.x;
            float b1 = B4.y;
            float b2 = B4.z;
            float b3 = B4.w;

            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float a = A_cur[(ty * TM + m) * AS_STRIDE + k];
                sum[m][0] += a * b0;
                sum[m][1] += a * b1;
                sum[m][2] += a * b2;
                sum[m][3] += a * b3;
            }
        }

        // Wait for next‑tile async loads to complete and swap
        if (has_next) {
            cp_async_wait_all();
            __syncthreads();

            // Swap double‑buffer pointers
            float* tmpA = A_cur; float* tmpB = B_cur;
            A_cur = A_next;   B_cur = B_next;
            A_next = tmpA;    B_next = tmpB;
        }
    }

    // Store results (enforce upper‑triangular)
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int g_row = row_start + ty * TM + m;
        if (g_row >= N) break;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int g_col = col_start + tx * TN + n;
            if (g_col >= N) break;
            if (g_row <= g_col)
                C[g_row * N + g_col] = sum[m][n];
            else
                C[g_row * N + g_col] = 0.0f;
        }
    }
}

// Launcher
extern "C" void triu_matmul_kernel_launcher(
    const float* A, const float* B, float* C, int N)
{
    int M_blk = (N + BM - 1) / BM;
    int num_blocks = M_blk * (M_blk + 1) / 2;

    size_t shared_mem_bytes = 2 * (BM * AS_STRIDE + BK * BS_STRIDE) * sizeof(float);

    dim3 block(16, 16);          // 256 threads
    dim3 grid(num_blocks);

    cudaFuncSetAttribute(
        triu_matmul_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_bytes);

    triu_matmul_kernel<<<grid, block, shared_mem_bytes>>>(A, B, C, N);
}
"""

_cuda_module = load_inline(
    name="triu_matmul_cuda_r090_h2000",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["triu_matmul_launcher"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-arch=sm_90",
        # "--maxrregcount=64"
    ],
)

def _triu_matmul_cuda(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    return _cuda_module.triu_matmul_launcher(A, B)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return _triu_matmul_cuda(A, B)

def get_inputs():
    A = torch.triu(torch.rand(N, N, device="cuda"))
    B = torch.triu(torch.rand(N, N, device="cuda"))
    return [A, B]

def get_init_inputs():
    return []