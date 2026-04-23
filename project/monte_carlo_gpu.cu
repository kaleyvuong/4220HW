#include <stdio.h>
#include <curand_kernel.h>

__global__ void mc_kernel(long n_per_thread, unsigned int *d_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(42, tid, 0, &state);

    unsigned int count = 0;
    for (long i = 0; i < n_per_thread; i++)
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f)
            count++;
    }
    d_counts[tid] = count;
}

int main()
{
    int threads = 256, blocks = 1024;
    int total_threads = threads * blocks; // 262,144 threads
    long n_per_thread = 400;              // ~10^8 total samples

    unsigned int *d_counts;
    cudaMalloc(&d_counts, total_threads * sizeof(unsigned int));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    mc_kernel<<<blocks, threads>>>(n_per_thread, d_counts);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    unsigned int *h_counts = (unsigned int *)malloc(total_threads * sizeof(unsigned int));
    cudaMemcpy(h_counts, d_counts, total_threads * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    long total_inside = 0;
    for (int i = 0; i < total_threads; i++)
        total_inside += h_counts[i];

    long n_total = (long)total_threads * n_per_thread;
    double pi = 4.0 * total_inside / n_total;
    printf("π ≈ %.6f  |  samples: %ld  |  time: %.1fms\n", pi, n_total, ms);

    cudaFree(d_counts);
    free(h_counts);
    return 0;
}
