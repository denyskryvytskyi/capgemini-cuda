/**
 * TASK: Addition of two vectors
 * RESULTS: (For vectors with the size = 100'000'000 with -O3 compilation flag)
 *  - CPU addition: ~110 ms
 *  - GPU addition: ~13 ms
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <malloc.h>
#include <iostream>
#include <chrono>

constexpr int32_t VEC_SIZE = 100'000'000;
constexpr int32_t ALIGNMENT = 16;
constexpr float VEC_A_OFFSET = 0.2f;
constexpr float VEC_B_OFFSET = 1.3f;
constexpr bool PRINT_VEC = false;

constexpr int32_t CUDA_BLOCK_SIZE = 512;                                                     // amount of threads per threads block
constexpr int32_t CUDA_BLOCKS_PER_GRID = (VEC_SIZE + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE; // amount of thread block

void printVec(float* pVec);
void initData(float* pVecA, float* pVecB);
void cleanup(float* pVecA, float* pVecB, float* pVecRes, float* pDevVecA, float* pDevVecB, float* pDevVecRes);
void add(float* pVecA, float* pVecB, float* pVecRes);
void addWithCuda(float* pDevVecA, float* pDevVecB, float* pDevVecRes, float* pVecRes);

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA devices

    if (!deviceCount) {
        std::cout << "CUDA-capable GPU isn't found.\n";
        return 0;
    }

    float* pVecA = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecA) {
        std::cerr << "Memory allocation failed for vector A." << std::endl;
        return 1;
    }
    
    float* pVecB = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecB) {
        _aligned_free(pVecA);
        std::cerr << "Memory allocation failed for vector B." << std::endl;
        return 1;
    }
    
    float* pVecRes = static_cast<float*>(_aligned_malloc(VEC_SIZE * sizeof(float), ALIGNMENT));
    if (!pVecRes) {
        _aligned_free(pVecA);
        _aligned_free(pVecB);
        std::cerr << "Memory allocation failed for vector B." << std::endl;
        return 1;
    }

    initData(pVecA, pVecB);

    if (PRINT_VEC) {
        std::cout << "Vector A: ";
        printVec(pVecA);
        
        std::cout << "Vector B: ";
        printVec(pVecB);
    }

    add(pVecA, pVecB, pVecRes);    // CPU addition

    // CUDA buffers
    float* pDevVecA = nullptr;
    float* pDevVecB = nullptr;
    float* pDevVecRes = nullptr;

    // Allocate GPU buffers for three vectors
    cudaError_t cudaStatus = cudaMalloc(&pDevVecA, VEC_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for vector A!\n";
        cleanup(pVecA, pVecB, pVecRes, pDevVecA, pDevVecB, pDevVecRes);
        return 1;
    }

    cudaStatus = cudaMalloc(&pDevVecB, VEC_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for vector B!\n";
        cleanup(pVecA, pVecB, pVecRes, pDevVecA, pDevVecB, pDevVecRes);
        return 1;
    }

    cudaStatus = cudaMalloc(&pDevVecRes, VEC_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for result vector!\n";
        cleanup(pVecA, pVecB, pVecRes, pDevVecA, pDevVecB, pDevVecRes);
        return 1;
    }

    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(pDevVecA, pVecA, VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed for vector A!\n";
        cleanup(pVecA, pVecB, pVecRes, pDevVecA, pDevVecB, pDevVecRes);
        return 1;
    }

    cudaStatus = cudaMemcpy(pDevVecB, pVecB, VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed for vector B!\n";
        cleanup(pVecA, pVecB, pVecRes, pDevVecA, pDevVecB, pDevVecRes);
        return 1;
    }

    addWithCuda(pDevVecA, pDevVecB, pDevVecRes, pVecRes);

    cleanup(pVecA, pVecB, pVecRes, pDevVecA, pDevVecB, pDevVecRes);

    return 0;
}

void printVec(float* pVec)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

void initData(float* pVecA, float* pVecB)
{
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecA[i] = static_cast<float>(i) + VEC_A_OFFSET;
        pVecB[i] = static_cast<float>(i) + VEC_B_OFFSET;
    }
}

void cleanup(float* pVecA, float* pVecB, float* pVecRes, float* pDevVecA, float* pDevVecB, float* pDevVecRes)
{
    cudaFree(pDevVecRes);
    cudaFree(pDevVecB);
    cudaFree(pDevVecA);

    _aligned_free(pVecRes);
    _aligned_free(pVecB);
    _aligned_free(pVecA);
}

void add(float* pVecA, float* pVecB, float* pVecRes)
{
    std::cout << "===== CPU Addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < VEC_SIZE; ++i) {
        pVecRes[i] = pVecA[i] + pVecB[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_VEC) {
        std::cout << "A + B: ";
        printVec(pVecRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

__global__ void addKernel(float* pVecA, float* pVecB, float* pVecRes)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = index; i < VEC_SIZE; i += stride)
        pVecRes[i] = pVecA[i] + pVecB[i];
}

void addWithCuda(float* pDevVecA, float* pDevVecB, float* pDevVecRes, float* pVecRes)
{
    std::cout << "===== GPU Addition =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    addKernel<<<CUDA_BLOCKS_PER_GRID, CUDA_BLOCK_SIZE>>>(pDevVecA, pDevVecB, pDevVecRes);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "addKernel launch failed:" << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // Any errors encountered during the launch
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error code" << cudaStatus << std::endl;
        return;
    }

    // Copy output vector from GPU buffer to host memory
    cudaStatus = cudaMemcpy(pVecRes, pDevVecRes, VEC_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!";
        return;
    }

    if (PRINT_VEC) {
        std::cout << "A + B: ";
        printVec(pVecRes);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}
