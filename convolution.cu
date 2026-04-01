#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cmath>
#include <sys/time.h>

#define FILTER_RADIUS 2
#define FILTER_SIZE   (2*FILTER_RADIUS+1)   // 5
#define TILE_SIZE     32

// Average 5x5 filter on host
const float F_h[FILTER_SIZE][FILTER_SIZE] = {
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25},
    {1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25, 1.0f/25}
};

// Average 5x5 filter in GPU constant memory
__constant__ float F_d[FILTER_SIZE][FILTER_SIZE];

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


// verify: check two cv::Mat images are close (relative tolerance 1e-2)
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols) {
    const float relativeTolerance = 1e-2f;
    for (int i = 0; i < (int)nRows; i++) {
        for (int j = 0; j < (int)nCols; j++) {
            float relativeError =
                ((float)answer1.at<unsigned char>(i, j) -
                 (float)answer2.at<unsigned char>(i, j)) / 255.0f;
            if (relativeError > relativeTolerance ||
                relativeError < -relativeTolerance) {
                printf("TEST FAILED at (%d, %d) with relativeError: %f\n",
                       i, j, relativeError);
                printf("    answer1.at<unsigned char>(%d, %d): %u\n",
                       i, j, answer1.at<unsigned char>(i, j));
                printf("    answer2.at<unsigned char>(%d, %d): %u\n\n",
                       i, j, answer2.at<unsigned char>(i, j));
                return false;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return true;
}

// A. CPU-only convolution (average blur)
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h,
                 unsigned int nRows, unsigned int nCols) {
    for (unsigned int row = 0; row < nRows; row++) {
        for (unsigned int col = 0; col < nCols; col++) {
            float pixVal = 0.0f;
            for (int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++) {
                for (int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++) {
                    int inRow = (int)row + fRow;
                    int inCol = (int)col + fCol;
                    if (inRow >= 0 && inRow < (int)nRows &&
                        inCol >= 0 && inCol < (int)nCols) {
                        pixVal += F_h[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS] *
                                  (float)Pin_Mat_h.at<unsigned char>(inRow, inCol);
                    }
                    // boundary: treat out-of-bounds as 0 (zero-padding)
                }
            }
            Pout_Mat_h.at<unsigned char>(row, col) =
                (unsigned char)fminf(fmaxf(pixVal, 0.0f), 255.0f);
        }
    }
}

// B. GPU kernel — simple (no tiling), filter from constant memory
__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin,
                                 unsigned int width, unsigned int height) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float pixVal = 0.0f;
        for (int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++) {
            for (int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++) {
                int inRow = (int)row + fRow;
                int inCol = (int)col + fCol;
                if (inRow >= 0 && inRow < (int)height &&
                    inCol >= 0 && inCol < (int)width) {
                    pixVal += F_d[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS] *
                              (float)Pin[inRow * width + inCol];
                }
            }
        }
        Pout[row * width + col] =
            (unsigned char)fminf(fmaxf(pixVal, 0.0f), 255.0f);
    }
}

// Host wrapper for blurImage_Kernel
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h,
                 unsigned int nRows, unsigned int nCols) {
    size_t imgBytes = (size_t)nRows * nCols * sizeof(unsigned char);

    unsigned char *d_Pin, *d_Pout;
    CHECK(cudaMalloc((void**)&d_Pin,  imgBytes));
    CHECK(cudaMalloc((void**)&d_Pout, imgBytes));

    CHECK(cudaMemcpy(d_Pin, Pin_Mat_h.data, imgBytes, cudaMemcpyHostToDevice));

    // Copy filter to constant memory
    CHECK(cudaMemcpyToSymbol(F_d, F_h, sizeof(F_h)));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((nCols + TILE_SIZE - 1) / TILE_SIZE,
                 (nRows + TILE_SIZE - 1) / TILE_SIZE);

    blurImage_Kernel<<<gridDim, blockDim>>>(d_Pout, d_Pin, nCols, nRows);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(Pout_Mat_h.data, d_Pout, imgBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_Pin));
    CHECK(cudaFree(d_Pout));
}

// C. GPU kernel — tiled convolution using shared memory + constant memory
//  Each block loads a (TILE_SIZE + 2*FILTER_RADIUS)^2 "halo" tile into
//  shared memory, then every thread computes its output pixel from shared mem.
#define SHARED_SIZE (TILE_SIZE + 2*FILTER_RADIUS)

__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin,
                                       unsigned int width, unsigned int height) {
    // Shared memory tile including halo
    __shared__ float s_tile[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Top-left corner of the shared tile in global coordinates
    int tileStartCol = (int)(blockIdx.x * TILE_SIZE) - FILTER_RADIUS;
    int tileStartRow = (int)(blockIdx.y * TILE_SIZE) - FILTER_RADIUS;

    // Each thread may need to load more than one element into shared memory
    // because SHARED_SIZE > TILE_SIZE. We use a loop.
    for (int r = ty; r < SHARED_SIZE; r += TILE_SIZE) {
        for (int c = tx; c < SHARED_SIZE; c += TILE_SIZE) {
            int inRow = tileStartRow + r;
            int inCol = tileStartCol + c;
            if (inRow >= 0 && inRow < (int)height &&
                inCol >= 0 && inCol < (int)width) {
                s_tile[r][c] = (float)Pin[inRow * width + inCol];
            } else {
                s_tile[r][c] = 0.0f;   // zero-padding for boundaries
            }
        }
    }
    __syncthreads();

    // Output pixel this thread is responsible for
    unsigned int outCol = blockIdx.x * TILE_SIZE + tx;
    unsigned int outRow = blockIdx.y * TILE_SIZE + ty;

    if (outCol < width && outRow < height) {
        float pixVal = 0.0f;
        // In shared memory, this thread's data starts at (ty+FILTER_RADIUS, tx+FILTER_RADIUS)
        for (int fRow = 0; fRow < FILTER_SIZE; fRow++) {
            for (int fCol = 0; fCol < FILTER_SIZE; fCol++) {
                pixVal += F_d[fRow][fCol] *
                          s_tile[ty + fRow][tx + fCol];
            }
        }
        Pout[outRow * width + outCol] =
            (unsigned char)fminf(fmaxf(pixVal, 0.0f), 255.0f);
    }
}

// Host wrapper for blurImage_tiled_Kernel
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h,
                       unsigned int nRows, unsigned int nCols) {
    size_t imgBytes = (size_t)nRows * nCols * sizeof(unsigned char);

    unsigned char *d_Pin, *d_Pout;
    CHECK(cudaMalloc((void**)&d_Pin,  imgBytes));
    CHECK(cudaMalloc((void**)&d_Pout, imgBytes));

    CHECK(cudaMemcpy(d_Pin, Pin_Mat_h.data, imgBytes, cudaMemcpyHostToDevice));

    // Copy filter to constant memory (idempotent if already done)
    CHECK(cudaMemcpyToSymbol(F_d, F_h, sizeof(F_h)));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((nCols + TILE_SIZE - 1) / TILE_SIZE,
                 (nRows + TILE_SIZE - 1) / TILE_SIZE);

    blurImage_tiled_Kernel<<<gridDim, blockDim>>>(d_Pout, d_Pin, nCols, nRows);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(Pout_Mat_h.data, d_Pout, imgBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_Pin));
    CHECK(cudaFree(d_Pout));
}

int main(int argc, char** argv) {
    cudaDeviceSynchronize();

    if (argc < 2) {
        printf("Usage: %s <input_image.jpg>\n", argv[0]);
        return -1;
    }

    double startTime, endTime;

    // Load grayscale image
    cv::Mat grayImg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (grayImg.empty()) {
        printf("Error: could not load image '%s'\n", argv[1]);
        return -1;
    }

    unsigned int nRows      = grayImg.rows;
    unsigned int nCols      = grayImg.cols;
    unsigned int nChannels  = grayImg.channels();
    printf("Image: %u rows x %u cols x %u channels\n\n",
           nRows, nCols, nChannels);

    // ----- OpenCV reference -----
    cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    cv::blur(grayImg, blurredImg_opencv,
             cv::Size(2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1),
             cv::Point(-1, -1), cv::BORDER_CONSTANT);
    endTime = myCPUTimer();
    printf("OpenCV's blur (CPU):              %f s\n\n", endTime - startTime);

    // ----- CPU implementation -----
    cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on CPU:                 %f s\n\n", endTime - startTime);

    // ----- GPU simple kernel -----
    cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on GPU:                 %f s\n\n", endTime - startTime);

    // ----- GPU tiled kernel -----
    cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_tiled_d(blurredImg_tiled_gpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("(tiled) blurImage on GPU:         %f s\n\n", endTime - startTime);

    // ----- Save output images -----
    bool check;
    check = cv::imwrite("./blurredImg_opencv.jpg",     blurredImg_opencv);
    if (!check) { printf("error writing blurredImg_opencv.jpg\n"); return -1; }

    check = cv::imwrite("./blurredImg_cpu.jpg",        blurredImg_cpu);
    if (!check) { printf("error writing blurredImg_cpu.jpg\n"); return -1; }

    check = cv::imwrite("./blurredImg_gpu.jpg",        blurredImg_gpu);
    if (!check) { printf("error writing blurredImg_gpu.jpg\n"); return -1; }

    check = cv::imwrite("./blurredImg_tiled_gpu.jpg",  blurredImg_tiled_gpu);
    if (!check) { printf("error writing blurredImg_tiled_gpu.jpg\n"); return -1; }

    // ----- Verification -----
    printf("Verifying blurImage_h vs OpenCV:\n");
    verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);

    printf("Verifying blurImage_d vs OpenCV:\n");
    verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);

    printf("Verifying blurImage_tiled_d vs OpenCV:\n");
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

    return 0;
}
