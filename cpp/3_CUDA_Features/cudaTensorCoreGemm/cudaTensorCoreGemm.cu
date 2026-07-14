/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
//   matrices from global memory to shared memory. After that, all warps in the
//   CTA reuse the A and B data from shared memory, thus reducing the number of
//   data copies from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
//   additional padding (skew) to reduce the number of shared memory access bank
//   conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
//   warp stores its subtiles to shared memory. The CTA then copies the shared
//   memory contents to global memory, again avoiding redundant random global
//   memory  accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
//   utilization, but carefully enough to avoid local memory use.

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK   8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES          (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES           (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES     (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4   // 一个 block 在这一维有 4 个 warp

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2    // 每个 warp 在“列/纵向分组”这一维负责 2 个 16x16 tile

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS) // 整个 block 在这一维共有 2 * 4 = 8 个 16x16 tile，对应实际尺寸：8 * 16 = 128

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 16 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 256-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
// H100 32个bank，每个bank宽度为4 bytes
#define SKEW_HALF 16

#define checkKernelErrors(expr)                                                               \
    do {                                                                                      \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess) {                                                           \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

using namespace nvcuda;

// 在 CPU 端初始化输入矩阵
__host__ void init_host_matrices(half *a, half *b, float *c)
{
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            a[i * K_GLOBAL + j] = (half)(rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++) {
        for (int j = 0; j < K_GLOBAL; j++) {
            b[i * K_GLOBAL + j] = (half)(rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        c[t] = static_cast<float>(rand() % 3);
    }
}

// 核心高性能 Tensor Core kernel
// D = alpha * A * B + beta * C
// 使用 nvcuda::wmma API，也就是 Tensor Core 的 warp-level matrix multiply API
// 一个 CTA/thread block 有 8 个 warp
// 每个 CTA 处理一个大的 128 x 128 输出 tile
// 每个 warp 负责多个 16 x 16 WMMA 小 tile
// 把 A、B、C/D 的 tile 缓存在 shared memory 里
// 用 SKEW_HALF 给 shared memory 做 padding，减少 bank conflict
__global__ void compute_gemm(const half *A, const half *B, const float *C, float *D, float alpha, float beta)
{
    // 动态shared memory
    // 每一行有 CHUNK_K * K + SKEW_HALF （4 * 16 + 16 = 80）个 half，行数由 SHMEM_SZ 决定
    // CHUNK_K * K = 当前 K chunk 的真实数据
    // SKEW_HALF   = padding，用来减少 shared memory bank conflict
    // shared mem 有bank用于并行读
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    // Warp and lane identification.
    // 没有warpM和warpN，只有warpID。一个 warp 算 2 x 4 = 8 个 16x16 D tile。
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    // shmem_idx_b_off = 128，shmem 的前 128 行留给 A，从第 128 行开始放 B
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    // 给当前 warp 做 WMMA load/store C/D fragment 用
    float *shmem_warp_tile_ptr =
        (float *)&shmem[0][0] + (warpId / 2) * SHMEM_STRIDE * K * 2 + (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    // 给当前 warp 从 global memory 连续搬 C 到 shared，以及最后从 shared 连续搬 D 到 global 用
    float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    // 目的：D = alpha * (A * B + beta_adjusted * C)
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    // 每个 block 怎么选择自己要算哪个 128x128 D tile。
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_ROW_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx                 = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            typedef int4 copy_t;

            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2)
                                            : (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2)
                                 ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                                 : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL)
                           + (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t      shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half *tile_ptr    = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t      shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const half *tile_ptr    = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
                // register -> shared mem
                // wmma::store_matrix_sync 可以把 WMMA fragment 写到 shared memory，也可以直接写到 global memory。
                // 关键不在函数名变不变，而在你传进去的指针指向哪里。
                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        // shared memory -> global mem
        // 不直接 store_matrix_sync 到 global，
        // 是因为先把每个 warp 的碎片结果整理进 shared memory，
        // 再由整个 CTA 做连续、合并的 int4 写回，
        // global memory 写入模式更规整。
        for (int i = 0; i < K; i++) {
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
// 简化版 WMMA kernel，主要用于教学和
// 但相比 compute_gemm：
// 1. 不做复杂的 shared memory 分块缓存
// 2.每个 warp 基本负责一个 16 x 16 输出 tile
// 3.直接从 global memory 读取 A、B、C
// 4.用 wmma::load_matrix_sync
// 5.用 wmma::mma_sync
// 6.用 wmma::store_matrix_sync
// 所以它更容易理解 WMMA API 的基本用法，但性能不如 compute_gemm
// A 是 m_ld x k_ld
// B 是 k_ld x n_ld
// C/D 是 m_ld x n_ld
// 这里用 half 做输入，是因为 Tensor Core 擅长处理半精度输入；累加和输出用 float，这样精度更好。
// half  = 16-bit 浮点数，2 字节
// float = 32-bit 浮点数，4 字节
// 4096*4096
/* 高性能的核心原因：
1. compute_gemm 用 shared memory 做 block 内数据缓存，从而显著增加 A/B 数据复用，减少 global memory 重复读取
        simple_wmma_gemm:
            每个 warp 自己从 global memory 读 A/B
            一个 warp 算一个 16x16 D tile
            相邻 warp 之间复用关系没有显式利用
        compute_gemm:
            一个 block 算 128x128 D tile
            block 先把当前 K chunk 的 A/B 搬到 shared memory
            8 个 warp 共同从 shared memory 读取 A/B
            同一份 A/B 被多个 warp、多个输出 tile 反复使用
2. 此外还有几个辅助优化：
        C/D 也通过 shared memory 整理读写
        int4 向量化搬运
        SKEW_HALF 减少 shared memory bank conflict
        每个 warp 计算多个 16x16 tile，提高计算密度
*/
__global__ void
simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld, int n_ld, int k_ld, float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    // 这里的 ld 是 leading dimension，可以先理解成“矩阵在内存里跨一行或一列要跳多少个元素”。
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    // 一个 warp 计算一个 16x16 的tile
    // blockDim.x = 128，blockDim.y = 4;
    // warpM: 第几个 16 行 tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    // warpN: 第几个 16 列 tile，不用“/warpSize”，因为blockDim.x = 128，blockDim.y = 4，本身就是个4*4的warp矩阵
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    // fragment 可以先理解成：Tensor Core / WMMA 使用的一小块矩阵数据，存在 warp 的寄存器状态里。
    // 不是单个线程独占一个完整矩阵，而是 一个 warp 合作持有一个 fragment。每个线程持有 fragment 的一部分，CUDA 帮你安排具体分布。
    // wmma::fragment<> 的第一个模板参数叫 Use，表示这个 fragment 在矩阵乘加里的“角色”，常见就三个
    // a_frag: 这是乘法左边的矩阵 A，元素是 half，内存按 row-major 解释。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    // b_frag: 这是乘法右边的矩阵 B，元素是 half，内存按 col-major 解释。
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    // acc_frag：表示当前输出 D 的中间累加结果，类型是 float。
    // 之后每走一个 K 方向 tile，就累加一次：acc_frag = acc_frag + a_frag * b_frag
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              c_frag;

    // 把累加器清零
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    // 循环次数：4096 / 16 = 256 次
    // acc += A[:, 0:16]    * B[0:16, :]
    // acc += A[:, 16:32]   * B[16:32, :]
    // acc += A[:, 32:48]   * B[32:48, :]
    // ...
    // acc += A[:, 4080:4096] * B[4080:4096, :]
    for (int i = 0; i < k_ld; i += WMMA_K) {
        // 表示当前这次 K 循环，A tile 从哪一列开始
        int aCol = i;
        // 表示当前 warp 要读的 A tile 从哪一行开始
        int aRow = warpM * WMMA_M;
        // 表示当前 warp 要读的 B tile 从哪一列开始。
        int bCol = warpN * N;
        // 表示当前这次 K 循环，B tile 从哪一行开始
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // Load the inputs
            // 没有把数据从 global memory 加载到 shared memory 的步骤。
            // 直接global memory -> WMMA fragment/register
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            // acc_frag = a_frag * b_frag + acc_frag，累加结果
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

        // 对 fragment 里的每个元素做：
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        // register fragment -> global memory D，寄存器直接写回gloabl mem
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

// 这是 CPU 版本的矩阵乘法，用于校验 GPU 结果。
__host__ void matMultiplyOnHost(half  *A,
                                half  *B,
                                float *C,
                                float  alpha,
                                float  beta,
                                int    numARows,
                                int    numAColumns,
                                int    numBRows,
                                int    numBColumns,
                                int    numCRows,
                                int    numCColumns)
{
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            float temp = 0.0;

            for (int k = 0; k < numAColumns; k++) {
                temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
            }

            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}

int main(int argc, char **argv)
{
    printf("Initializing...\n");

    // 选择 CUDA device
    int dev = findCudaDevice(argc, (const char **)argv);

    // 获取 CUDA device 属性
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
    if (deviceProp.major < 7) {
        printf("cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
               "Cores.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    // 声明 CPU 端矩阵
    half  *A_h = NULL;
    half  *B_h = NULL;
    float *C_h = NULL;
#if CPU_DEBUG
    float *result_hD   = NULL;
    float *result_host = NULL;
#endif

    // 分配 CPU 端矩阵内存
    A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
    result_hD   = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

    // 声明 GPU 端矩阵
    half  *A = NULL;
    half  *B = NULL;
    float *C = NULL;
    float *D = NULL;

    // 分配 GPU 端矩阵内存
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D), sizeof(float) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices(A_h, B_h, C_h);

    printf("Preparing data for GPU...\n");

    // 复制数据从 CPU 到 GPU
    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // 计算所需的 shared memory 大小：max(A_tile + B_tile, C_or_D_tile)
    enum {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
        // per-CTA chunks
        // of the A and B matrices. Therefore, the right amount to request is the
        // maximum of those
        // two numbers.
        SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
                       M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };

    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

    const float alpha = 1.1f;
    const float beta  = 1.2f;

    cudaEvent_t start, stop;

    // 创建 CUDA event 用于测量 kernel 执行时间
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // If enough shared memory available on the GPU use high performant kernel
    // 如果 GPU 的 shared memory 足够，使用高性能 kernel compute_gemm
    if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ) {
        printf("Computing... using high performance kernel compute_gemm \n");

        checkCudaErrors(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
        checkKernelErrors(
            (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
#endif
    }
    else {
        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        printf("Computing... using simple_wmma_gemm kernel\n");
        simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
#endif
    }

    // 记录结束时间
    checkCudaErrors(cudaEventRecord(stop));
    // 等待事件完成
    checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");

    memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

    matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
        if (fabs(result_hD[i] - result_host[i]) > 0.1f)
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], result_host[i]);
    }
    free(result_hD);
    free(result_host);
#endif

    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);
    // 计算 TFLOPS
    printf("TFLOPS: %.2f\n",
           static_cast<double>((static_cast<double>(M_GLOBAL) * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.))
               / 1e12);

    free(A_h);
    free(B_h);
    free(C_h);
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

    return 0;
}
