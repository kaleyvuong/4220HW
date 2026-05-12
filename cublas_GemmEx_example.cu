 #include <cstdio>
 #include <cstdlib>
 #include <vector>
 
 #include <cublas_v2.h>
 #include <cuda_runtime.h>
 
 #include "cublas_utils.h"
 
 using data_type = double;
 
 int main(int argc, char *argv[]) {
     cublasHandle_t cublasH = NULL;
     cudaStream_t stream = NULL;
 
     const int m = 3;   // A is of size m by k
     const int n = 4;   // B is of size k by n
     const int k = 2;

    // In cublasGemmEx (and other cuBLAS routines), lda, ldb, and ldc refer to the leading dimensions of the matrices A, B, and C, respectively.
    // The leading dimension is the distance (in elements) between the starts of consecutive columns in memory.
    // If the matrix is stored in a larger array (for alignment or padding), the leading dimension may be greater than the actual number of rows, for performance, memory alignment, or submatrix access reasons.
     const int lda = 3;
     const int ldb = 2;
     const int ldc = 3;
     /*
      *   A = | 1.0 | 2.0 |
      *       | 3.0 | 4.0 |
     *        | 5.0 | 6.0 |
      *
      *   B = | 7.0 | 8.0 | 9.0 | 10.0 |
      *       | 11.0 | 12.0 | 13.0 | 14.0 |
      */
 
     const std::vector<data_type> A = {1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
     const std::vector<data_type> B = {7.0, 11.0, 8.0, 12.0, 9.0, 13.0, 10.0, 14.0};
     std::vector<data_type> C(m * n);
     const data_type alpha = 1.0;    // C = alpha * A * B + beta * C
     const data_type beta = 0.0;
 
     data_type *d_A = nullptr;
     data_type *d_B = nullptr;
     data_type *d_C = nullptr;
 
     cublasOperation_t transa = CUBLAS_OP_N;
     cublasOperation_t transb = CUBLAS_OP_N;
 
     printf("A\n");
     print_matrix(m, k, A.data(), lda);
     printf("=====\n");
 
     printf("B\n");
     print_matrix(k, n, B.data(), ldb);
     printf("=====\n");
 
     /* step 1: create cublas handle, bind a stream */
     CUBLAS_CHECK(cublasCreate(&cublasH));
 
     CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
     CUBLAS_CHECK(cublasSetStream(cublasH, stream));
 
     /* step 2: copy data to device */
     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
     CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));
 
     CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                                stream));
     CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                                stream));
 
     /* step 3: compute */
     CUBLAS_CHECK(cublasGemmEx(
         cublasH, transa, transb, m, n, k, &alpha, 
         d_A, traits<data_type>::cuda_data_type, lda, 
         d_B, traits<data_type>::cuda_data_type, ldb, 
         &beta, d_C, traits<data_type>::cuda_data_type, ldc,
         CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
 
     /* step 4: copy data to host */
     CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                                stream));
 
     CUDA_CHECK(cudaStreamSynchronize(stream));
 
     /*
      *   C = |  29.0   |   32.0   |   35.0   |   38.0 | 
              |  65.0   |   72.0   |   79.0   |   86.0 | 
              | 101.0   |  112.0   |  123.0   |  134.0 | 
      */
 
     printf("C\n");
     print_matrix(m, n, C.data(), ldc);
     printf("=====\n");
 
     /* free resources */
     CUDA_CHECK(cudaFree(d_A));
     CUDA_CHECK(cudaFree(d_B));
     CUDA_CHECK(cudaFree(d_C));
 
     CUBLAS_CHECK(cublasDestroy(cublasH));
 
     CUDA_CHECK(cudaStreamDestroy(stream));
 
     CUDA_CHECK(cudaDeviceReset());
 
     return EXIT_SUCCESS;
 }