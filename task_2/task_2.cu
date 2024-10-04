/**
 * TASK: Matrix Multiplication Using CUDA
 * NOTE: Implemented and tested on Windows 10 with NVIDIA GTX 1050 (laptop) card
 * RESULTS: (for matrices A(1500; 2000); B(2000; 3000) and result matrix (1500;3000) with -O3 compilation flag)
 *  - CPU multiplication: ~4000 ms
 *  - GPU multiplication (simple kernel matMulKernel): ~200 ms (~20 times faster)
 *  - GPU multiplication (tiled matrix kernel matMulTiledKernel): ~87 ms (~45 times faster)
 *  - GPU data preparation (time to allocate GPU buffers and host to device data copy): ~380 ms
 **/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <malloc.h>
#include <iostream>
#include <chrono>

constexpr int32_t MAT_DIM_N = 1500; // rows of the matrix A
constexpr int32_t MAT_DIM_M = 2000; // cols of the matrix A and rows of the matrix B
constexpr int32_t MAT_DIM_K = 3000; // cols of the matrix B
constexpr int32_t MAT_A_SIZE = MAT_DIM_N * MAT_DIM_M;
constexpr int32_t MAT_B_SIZE = MAT_DIM_M * MAT_DIM_K;
constexpr int32_t MAT_RES_SIZE = MAT_DIM_N * MAT_DIM_K;
constexpr int32_t ALIGNMENT = 16;
constexpr float MAT_A_OFFSET = 0.5f;
constexpr float MAT_B_OFFSET = 1.3f;
constexpr bool PRINT_MAT = false;

// CUDA specific
constexpr int32_t TILE_WIDTH = 16;                                          // size of the matrix tile 16x16
constexpr int32_t TILES_AMOUNT = (MAT_DIM_M + TILE_WIDTH - 1) / TILE_WIDTH; // amount of tiles to cover all matrix elements
const dim3 CUDA_BLOCK_SIZE(TILE_WIDTH, TILE_WIDTH);                         // 256 threads per block. Should have the same dimension as matrix tile for efficient processing
const dim3 CUDA_GRID_SIZE((MAT_DIM_K + CUDA_BLOCK_SIZE.x - 1) / CUDA_BLOCK_SIZE.x, (MAT_DIM_N + CUDA_BLOCK_SIZE.y - 1) / CUDA_BLOCK_SIZE.y); // blocks to cover the matrix

// Helpers
void initData(float* pMatA, float* pMatB, float* pMatRes);
void resetRes(float* pMatRes);
void printMat(float* pMat, int32_t rows, int32_t cols);
void cleanup(float* pMatA, float* pMatB, float* pMatRes, float* pDevMatA, float* pDevMatB, float* pDevMatRes);
void matMul(float* pMatA, float* pMatB, float* pMatRes);
void matMulCuda(float* pDevMatA, float* pDevMatB, float* pDevMatRes, float* pMatRes);

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA devices

    if (!deviceCount) {
        std::cout << "CUDA-capable GPU isn't found.\n";
        return 0;
    }

    float* pMatA = static_cast<float*>(_aligned_malloc(MAT_A_SIZE * sizeof(float), ALIGNMENT));
    if (!pMatA) {
        std::cerr << "Failed to allocate memory for matrix A." << std::endl;
        return 1;
    }

    float* pMatB = static_cast<float*>(_aligned_malloc(MAT_B_SIZE * sizeof(float), ALIGNMENT));
    if (!pMatB) {
        _aligned_free(pMatA);
        std::cerr << "Failed to allocate memory for matrix B." << std::endl;
        return 1;
    }

    float* pMatRes = static_cast<float*>(_aligned_malloc(MAT_RES_SIZE * sizeof(float), ALIGNMENT));
    if (!pMatRes) {
        _aligned_free(pMatA);
        _aligned_free(pMatB);
        std::cerr << "Failed to allocate memory for result matrix." << std::endl;
        return 1;
    }

    initData(pMatA, pMatB, pMatRes);

    if (PRINT_MAT) {
        std::cout << "Mat A:\n";
        printMat(pMatA, MAT_DIM_N, MAT_DIM_M);

        std::cout << "Mat B:\n";
        printMat(pMatB, MAT_DIM_M, MAT_DIM_K);
    }

    resetRes(pMatRes);
    matMul(pMatA, pMatB, pMatRes);

    std::cout << "===== GPU Matrix Multiplication =====\n";
    resetRes(pMatRes);

    // CUDA buffers
    float* pDevMatA = nullptr;
    float* pDevMatB = nullptr;
    float* pDevMatRes = nullptr;

    const auto startTimePoint = std::chrono::high_resolution_clock::now();

    // Allocate GPU buffers for three vectors
    cudaError_t cudaStatus = cudaMalloc((void**)&pDevMatA, MAT_A_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for vector A!\n";
        cleanup(pMatA, pMatB, pMatRes, pDevMatA, pDevMatB, pDevMatRes);
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&pDevMatB, MAT_B_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for vector B!\n";
        cleanup(pMatA, pMatB, pMatRes, pDevMatA, pDevMatB, pDevMatRes);
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&pDevMatRes, MAT_RES_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for result vector!\n";
        cleanup(pMatA, pMatB, pMatRes, pDevMatA, pDevMatB, pDevMatRes);
        return 1;
    }

    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(pDevMatA, pMatA, MAT_A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed for vector A!\n";
        cleanup(pMatA, pMatB, pMatRes, pDevMatA, pDevMatB, pDevMatRes);
        return 1;
    }

    cudaStatus = cudaMemcpy(pDevMatB, pMatB, MAT_B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed for vector B!\n";
        cleanup(pMatA, pMatB, pMatRes, pDevMatA, pDevMatB, pDevMatRes);
        return 1;
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "GPU data preparation (buffers allocation and host to device data copy) time: " << duration.count() << " ms.\n";

    matMulCuda(pDevMatA, pDevMatB, pDevMatRes, pMatRes);

    cleanup(pMatA, pMatB, pMatRes, pDevMatA, pDevMatB, pDevMatRes);

    return 0;
}

void initData(float* pMatA, float* pMatB, float* pMatRes)
{
    for (int i = 0; i < MAT_A_SIZE; ++i) {
        pMatA[i] = static_cast<float>(i) + MAT_A_OFFSET;
    }
    for (int i = 0; i < MAT_B_SIZE; ++i) {
        pMatB[i] = static_cast<float>(i) + MAT_B_OFFSET;
    }
}

void resetRes(float* pMatRes)
{
    memset(pMatRes, 0, MAT_DIM_N * MAT_DIM_K * sizeof(float));
}

void printMat(float* pMat, int32_t rows, int32_t cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << pMat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

void cleanup(float* pMatA, float* pMatB, float* pMatRes, float* pDevMatA, float* pDevMatB, float* pDevMatRes)
{
    cudaFree(pDevMatRes);
    cudaFree(pDevMatB);
    cudaFree(pDevMatA);
    _aligned_free(pMatRes);
    _aligned_free(pMatB);
    _aligned_free(pMatA);
}


void matMul(float* pMatA, float* pMatB, float* pMatRes)
{
    std::cout << "===== CPU Matrix Multiplication =====\n";

    // result matrix dim - [N;K]
    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int row = 0; row < MAT_DIM_N; ++row) {
        for (int col = 0; col < MAT_DIM_K; ++col) {
            for (int i = 0; i < MAT_DIM_M; ++i) {
                pMatRes[row * MAT_DIM_K + col] += pMatA[row * MAT_DIM_M + i] * pMatB[i * MAT_DIM_K + col];
            }
        }
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes, MAT_DIM_N, MAT_DIM_K);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

__global__ void matMulKernel(float* pMatA, float* pMatB, float* pMatRes, int matDimN, int matDimM, int matDimK)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matDimN && col < matDimK) {           // bounding checks
        float sum = 0.0f;
        for (int i = 0; i < matDimM; ++i) {
            sum += pMatA[row * matDimM + i] * pMatB[i * matDimK + col];
        }
        pMatRes[row * matDimK + col] = sum;
    }
}

__global__ void matMulTiledKernel(float* pMatA, float* pMatB, float* pMatRes, int matDimN, int matDimM, int matDimK, int tilesAmount)
{
    // Allocate shared memory for sub-matrices A and B tiles for faster access
    // Shared for all threads in one thread block
    __shared__ float sharedMatA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedMatB[TILE_WIDTH][TILE_WIDTH];

    // Calculate element row and column within the output result matrix to be processed by thread
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Accumulator for the result element
    float elementValue = 0.0f;

    // One thread block calculates result values for one tile of the result matrix (256 threads in parallel)
    // Loop over tiles of matrices A and B (one tile per one iteration to sum up to element value)
    for (int t = 0; t < tilesAmount; ++t) {
        // Load elements of A and B into shared memory
        const int elementTileAIndex = t * TILE_WIDTH + threadIdx.x; // each thread process one element of the tile from matrix A
        if (row < matDimN && (elementTileAIndex) < matDimM) {
            sharedMatA[threadIdx.y][threadIdx.x] = pMatA[row * matDimM + elementTileAIndex]; 
        } else {
            sharedMatA[threadIdx.y][threadIdx.x] = 0.0f; // Padding for out of bounds threads
        }

        const int elementTileBIndex = t * TILE_WIDTH + threadIdx.y; // each thread process one element of the tile from matrix B
        if (col < matDimK && elementTileBIndex < matDimM) {
            sharedMatB[threadIdx.y][threadIdx.x] = pMatB[elementTileBIndex * matDimK + col];
        } else {
            sharedMatB[threadIdx.y][threadIdx.x] = 0.0f; // Padding for out of bounds threads
        }

        // Synchronize to make sure all data is loaded before computation
        __syncthreads();

        // Perform computation on the tile
        for (int i = 0; i < TILE_WIDTH; ++i) {
            elementValue += sharedMatA[threadIdx.y][i] * sharedMatB[i][threadIdx.x];
        }

        // Synchronize to make sure all threads are done before loading the next tile
        __syncthreads();
    }

    // Write the result back to the output matrix C
    if (row < matDimN && col < matDimK) {
        pMatRes[row * matDimK + col] = elementValue;
    }
}

void matMulCuda(float* pDevMatA, float* pDevMatB, float* pDevMatRes, float* pMatRes)
{
    cudaEvent_t startKernelEvent, stopKernelEvent; // events to measure kernel execution time
    cudaEventCreate(&startKernelEvent);
    cudaEventCreate(&stopKernelEvent);

    cudaEventRecord(startKernelEvent, 0);
    matMulTiledKernel<<<CUDA_GRID_SIZE, CUDA_BLOCK_SIZE>>>(pDevMatA, pDevMatB, pDevMatRes, MAT_DIM_N, MAT_DIM_M, MAT_DIM_K, TILES_AMOUNT);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "matMulKernel launch failed:" << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    cudaEventRecord(stopKernelEvent, 0);
    cudaEventSynchronize(stopKernelEvent);

    float kernelTimeMs = 0.0f;
    cudaEventElapsedTime(&kernelTimeMs, startKernelEvent, stopKernelEvent);
    cudaEventDestroy(startKernelEvent);
    cudaEventDestroy(stopKernelEvent);

    cudaStatus = cudaDeviceSynchronize();

    // Any errors encountered during the launch
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error code" << cudaStatus << std::endl;
        return;
    }

    // Copy the result matrix back to the host
    cudaStatus = cudaMemcpy(pMatRes, pDevMatRes, MAT_RES_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!";
        return;
    }

    if (PRINT_MAT) {
        std::cout << "Result matrix:\n";
        printMat(pMatRes, MAT_DIM_N, MAT_DIM_K);
    }

    std::cout << "Execution time (kernel): " << kernelTimeMs << " ms.\n";
}