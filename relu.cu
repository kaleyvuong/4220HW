#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 16777216

// CPU
void relu_cpu(float *x, float *y, unsigned int n) {
    for (unsigned int i = 0; i < n; i++) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

//GPU
__global__ void relu_gpu(float *x, float *y, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] > 0 ? x[i] : 0;
    }
}

int main() {
    int size = N * sizeof(float);

    printf("Vector size %d\n", N);

    // Allocate memory on host
    float* x_h = (float*) malloc(size);
    float* y_h_cpu = (float*) malloc(size);
    float* y_h_gpu = (float*) malloc(size);

    for (int i = 0; i < N; i++) {
        x_h[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random values between -1 and 1
    }

    // CPU time
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);

    relu_cpu(x_h, y_h_cpu, N);

    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    
    double cpu_time = (cpu_end.tv_sec - cpu_start.tv_sec) + (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e9;
    
    printf("reluAdd on CPU:                                  %f s\n", cpu_time);

    //GPU time
    float *x_d, *y_d;
    cudaEvent_t start, stop;
    float ms = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //cudaMalloc
    cudaEventRecord(start);
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&y_d, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("    cudaMalloc:                                  %f s\n", ms / 1000);

    //cudaMemcpy host --> device
    cudaEventRecord(start);
    cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("    cudaMalloc:                                  %f s\n", ms / 1000);

    //Launch kernel
    int blockSize = 512;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    relu_gpu<<<numBlocks, blockSize>>>(x_d, y_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("    reluKernel:                                  %f s\n", ms / 1000);

    // cudaMemcpy device -> host
    cudaEventRecord(start);
    cudaMemcpy(y_h_gpu, y_d, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("    cudaMemcpy:                                  %f s\n", ms / 1000);

    //Total GPU time
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    float total_ms = 0;

    cudaEventRecord(total_start);
    cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
    relu_gpu<<<numBlocks, blockSize>>>(x_d, y_d, N);
    cudaMemcpy(y_h_gpu, y_d, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_ms, total_start, total_stop);
    printf("reluAdd on GPU:    %f s\n", total_ms / 1000);

    //Clean
    cudaFree(x_d);
    cudaFree(y_d);
    free(x_h);
    free(y_h_cpu);
    free(y_h_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    
    return 0;
}