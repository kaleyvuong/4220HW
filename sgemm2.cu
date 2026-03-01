#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Macro for CUDA error checking
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// CPU timer function
double myCPUTimer() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Host function for CPU-only matrix multiplication
void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A_h[i * k + p] * B_h[p * n + j];
            }
            C_h[i * n + j] = sum;
        }
    }
}

// CUDA kernel where each thread computes one element
__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += A_d[row * k + p] * B_d[p * n + col];
        }
        C_d[row * n + col] = sum;
    }
}

// CUDA kernel where a "tiled" version of matrix multiplication is presented 
// which uses dynamically allocated space in shared memory. 
// Each thread calculates one element of the output matrix.
__global__ void matrixMulKernel_tiled(int m, int k, int n, const float *A_d, const float *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz) {
    // Dynamically allocated shared memory
    extern __shared__ float shared_mem[];
    
    // Partition shared memory into two tiles
    float *A_tile = shared_mem;                    // First Adz_sz floats
    float *B_tile = shared_mem + Adz_sz;          // Next Bdz_sz floats
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global row and column indices
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    
    // Tile dimensions
    int TILE_WIDTH = blockDim.x;  // Assuming square tiles
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Loop over tiles of A and B required to compute C element
    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int a_col = t * TILE_WIDTH + tx;
        if (row < m && a_col < k) {
            A_tile[ty * TILE_WIDTH + tx] = A_d[row * k + a_col];
        } else {
            A_tile[ty * TILE_WIDTH + tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = t * TILE_WIDTH + ty;
        if (b_row < k && col < n) {
            B_tile[ty * TILE_WIDTH + tx] = B_d[b_row * n + col];
        } else {
            B_tile[ty * TILE_WIDTH + tx] = 0.0f;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += A_tile[ty * TILE_WIDTH + i] * B_tile[i * TILE_WIDTH + tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < m && col < n) {
        C_d[row * n + col] = sum;
    }
}

// Host function for 1thread1element kernel
void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    float *A_d, *B_d, *C_d;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);
    
    // Allocate device memory
    CHECK(cudaMalloc((void**)&A_d, size_A));
    CHECK(cudaMalloc((void**)&B_d, size_B));
    CHECK(cudaMalloc((void**)&C_d, size_C));
    
    // Copy data from host to device
    CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

// A host function for handling device memory allocation and copy, device query, 
// and dynamically configuring the amount of shared memory and calling the specific 
// CUDA kernel, matrixMulKernel_tiled()
void basicSgemm_d_tiled(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    float *A_d, *B_d, *C_d;
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);
    
    // Allocate device memory
    CHECK(cudaMalloc((void**)&A_d, size_A));
    CHECK(cudaMalloc((void**)&B_d, size_B));
    CHECK(cudaMalloc((void**)&C_d, size_C));
    
    // Copy data from host to device
    CHECK(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));
    
    // Device query to determine optimal configuration
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Configure kernel launch parameters
    int TILE_WIDTH = 16;  // Tile size (can be tuned)
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    
    // Calculate shared memory size needed
    // We need two tiles: one for A (TILE_WIDTH x TILE_WIDTH) and one for B (TILE_WIDTH x TILE_WIDTH)
    unsigned Adz_sz = TILE_WIDTH * TILE_WIDTH;  // Size of A tile in shared memory
    unsigned Bdz_sz = TILE_WIDTH * TILE_WIDTH;  // Size of B tile in shared memory
    size_t sharedMemSize = (Adz_sz + Bdz_sz) * sizeof(float);
    
    // Check if shared memory requirement is within device limits
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("Warning: Requested shared memory (%zu bytes) exceeds device limit (%zu bytes)\n", 
               sharedMemSize, deviceProp.sharedMemPerBlock);
        printf("Reducing tile size may be necessary.\n");
    }
    
    // Launch kernel with dynamic shared memory
    matrixMulKernel_tiled<<<gridDim, blockDim, sharedMemSize>>>(m, k, n, A_d, B_d, C_d, Adz_sz, Bdz_sz);
    CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

// Verification function
bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    const float epsilon = 1e-3f;
    for (unsigned int i = 0; i < nRows * nCols; i++) {
        if (fabs(CPU_Answer[i] - GPU_Answer[i]) > epsilon) {
            printf("Mismatch at index %u: CPU = %f, GPU = %f\n", i, CPU_Answer[i], GPU_Answer[i]);
            return false;
        }
    }
    return true;
}

// Main function
int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <m> <k> <n>\n", argv[0]);
        return 1;
    }
    
    // Parse command-line arguments
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    
    printf("Matrix dimensions: A(%d x %d), B(%d x %d), C(%d x %d)\n", m, k, k, n, m, n);
    
    // Allocate host memory
    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);
    
    float *A_h = (float*)malloc(size_A);
    float *B_h = (float*)malloc(size_B);
    float *C_h_cpu = (float*)malloc(size_C);
    float *C_h_gpu1 = (float*)malloc(size_C);
    float *C_h_gpu2 = (float*)malloc(size_C);
    
    // Initialize matrices with random values
    srand(2024);
    for (int i = 0; i < m * k; i++) {
        A_h[i] = rand() % 100 / 100.0f;
    }
    for (int i = 0; i < k * n; i++) {
        B_h[i] = rand() % 100 / 100.0f;
    }
    
    // CPU computation
    printf("\n=== CPU Computation ===\n");
    double cpu_start = myCPUTimer();
    basicSgemm_h(m, k, n, A_h, B_h, C_h_cpu);
    double cpu_end = myCPUTimer();
    printf("CPU time: %.6f s\n", cpu_end - cpu_start);
    
    // GPU computation - 1 thread 1 element
    printf("\n=== GPU Computation (1 thread 1 element) ===\n");
    double gpu1_start = myCPUTimer();
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_h_gpu1);
    double gpu1_end = myCPUTimer();
    printf("GPU time (1thread1element): %.6f s\n", gpu1_end - gpu1_start);
    
    // GPU computation - tiled with shared memory
    printf("\n=== GPU Computation (Tiled with Shared Memory) ===\n");
    double gpu2_start = myCPUTimer();
    basicSgemm_d_tiled(m, k, n, A_h, B_h, C_h_gpu2);
    double gpu2_end = myCPUTimer();
    printf("GPU time (tiled): %.6f s\n", gpu2_end - gpu2_start);
    
    // Verification
    printf("\n=== Verification ===\n");
    bool result1 = verify(C_h_cpu, C_h_gpu1, m, n);
    printf("Verification (CPU vs 1thread1element): %s\n", result1 ? "PASSED" : "FAILED");
    
    bool result2 = verify(C_h_cpu, C_h_gpu2, m, n);
    printf("Verification (CPU vs tiled): %s\n", result2 ? "PASSED" : "FAILED");
    
    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h_cpu);
    free(C_h_gpu1);
    free(C_h_gpu2);
    
    return 0;
}
