#include <stdio.h>
#include <stdlib.h>
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

// CUDA kernel where each thread computes one row
__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A_d[row * k + p] * B_d[p * n + col];
            }
            C_d[row * n + col] = sum;
        }
    }
}

// CUDA kernel where each thread computes one column
__global__ void matrixMulKernel_1thread1column(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        for (int row = 0; row < m; row++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A_d[row * k + p] * B_d[p * n + col];
            }
            C_d[row * n + col] = sum;
        }
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
    CHECK(cudaMalloc((void**)&C_d, size_C));jj 
    
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

// Host function for 1thread1row kernel
void basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
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
    int blockDim = 256;
    int gridDim = (m + blockDim - 1) / blockDim;
    
    // Launch kernel
    matrixMulKernel_1thread1row<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
}

// Host function for 1thread1column kernel
void basicSgemm_d_1thread1column(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
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
    int blockDim = 256;
    int gridDim = (n + blockDim - 1) / blockDim;
    
    // Launch kernel
    matrixMulKernel_1thread1column<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
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
    float *C_h_gpu3 = (float*)malloc(size_C);
    
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
    
    // GPU computation - 1 thread 1 row
    printf("\n=== GPU Computation (1 thread 1 row) ===\n");
    double gpu2_start = myCPUTimer();
    basicSgemm_d_1thread1row(m, k, n, A_h, B_h, C_h_gpu2);
    double gpu2_end = myCPUTimer();
    printf("GPU time (1thread1row): %.6f s\n", gpu2_end - gpu2_start);
    
    // GPU computation - 1 thread 1 column
    printf("\n=== GPU Computation (1 thread 1 column) ===\n");
    double gpu3_start = myCPUTimer();
    basicSgemm_d_1thread1column(m, k, n, A_h, B_h, C_h_gpu3);
    double gpu3_end = myCPUTimer();
    printf("GPU time (1thread1column): %.6f s\n", gpu3_end - gpu3_start);
    
    // Verification
    printf("\n=== Verification ===\n");
    bool result1 = verify(C_h_cpu, C_h_gpu1, m, n);
    printf("Verification (CPU vs 1thread1element): %s\n", result1 ? "PASSED" : "FAILED");
    
    bool result2 = verify(C_h_cpu, C_h_gpu2, m, n);
    printf("Verification (CPU vs 1thread1row): %s\n", result2 ? "PASSED" : "FAILED");
    
    bool result3 = verify(C_h_cpu, C_h_gpu3, m, n);
    printf("Verification (CPU vs 1thread1column): %s\n", result3 ? "PASSED" : "FAILED");
    
    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h_cpu);
    free(C_h_gpu1);
    free(C_h_gpu2);
    free(C_h_gpu3);
    
    return 0;
}
