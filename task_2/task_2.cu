/**
 * TASK: Matrix Multiplication Using CUDA
 * RESULTS: (for matrices A(1500; 2000); B(2000; 3000) and result matrix (1500;3000) with -O3 compilation flag)
 *  - CPU multiplication: ~4300 ms
 *  - GPU multiplication: ~540 ms (~8 times faster)
 * TODO:
 *  - tiled matrices
 *  - transpose matrix B for sequential access
 **/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <iostream>
#include <malloc.h>

constexpr int32_t MAT_DIM_N = 1500; // rows of the matrix A
constexpr int32_t MAT_DIM_M = 2000; // cols of the matrix A and rows of the matrix B
constexpr int32_t MAT_DIM_K = 3000; // cols of the matrix B
constexpr int32_t MAT_A_SIZE = MAT_DIM_N * MAT_DIM_M;
constexpr int32_t MAT_B_SIZE = MAT_DIM_M * MAT_DIM_K;
constexpr int32_t MAT_RES_SIZE = MAT_DIM_N * MAT_DIM_K;
constexpr int32_t ALIGNMENT = 16;
constexpr bool PRINT_MAT = false;
constexpr float MAT_A_OFFSET = 0.5f;
constexpr float MAT_B_OFFSET = 1.3f;

const dim3 THREADS_PER_BLOCK(16, 16);                                       // 16x16 threads per block
const dim3 BLOCKS_PER_GRID((MAT_DIM_K + 15) / 16, (MAT_DIM_N + 15) / 16);   // blocks to cover the matrix

void initData(float* pMatA, float* pMatB, float* pMatRes);
void resetRes(float* pMatRes);
void printMat(float* pMat, int32_t rows, int32_t cols);

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

    resetRes(pMatRes);

    // CUDA buffers
    float* pDevMatA = nullptr;
    float* pDevMatB = nullptr;
    float* pDevMatRes = nullptr;

    // Allocate GPU buffers for three vectors
    cudaError_t cudaStatus = cudaMalloc((void**)&pDevMatA, MAT_A_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for vector A!\n";
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&pDevMatB, MAT_B_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for vector B!\n";
        cudaFree(pDevMatA);
        return 1;
    }

    cudaStatus = cudaMalloc((void**)&pDevMatRes, MAT_RES_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for result vector!\n";
        cudaFree(pDevMatA);
        cudaFree(pDevMatB);
        return 1;
    }

    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(pDevMatA, pMatA, MAT_A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed for vector A!\n";
    }

    cudaStatus = cudaMemcpy(pDevMatB, pMatB, MAT_B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed for vector B!\n";
    }

    matMulCuda(pDevMatA, pDevMatB, pDevMatRes, pMatRes);

    cudaFree(pDevMatRes);
    cudaFree(pDevMatB);
    cudaFree(pDevMatA);

    _aligned_free(pMatRes);
    _aligned_free(pMatB);
    _aligned_free(pMatA);

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

__global__ void matMulKernel(float* pMatA, float* pMatB, float* pMatRes, int MAT_DIM_N, int MAT_DIM_M, int MAT_DIM_K)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MAT_DIM_N && col < MAT_DIM_K) {           // bounding checks
        for (int i = 0; i < MAT_DIM_M; ++i) {
            pMatRes[row * MAT_DIM_K + col] += pMatA[row * MAT_DIM_M + i] * pMatB[i * MAT_DIM_K + col];
        }
    }
}

void matMulCuda(float* pDevMatA, float* pDevMatB, float* pDevMatRes, float* pMatRes)
{
    std::cout << "===== GPU Matrix Multiplication =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    matMulKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(pDevMatA, pDevMatB, pDevMatRes, MAT_DIM_N, MAT_DIM_M, MAT_DIM_K);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "matMulKernel launch failed:" << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    cudaStatus = cudaDeviceSynchronize();
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

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

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}