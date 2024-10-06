/**
 * TASK: Reduction Operations Using CUDA
 * NOTE: Implemented and tested on Windows 10 with NVIDIA GTX 1050 (laptop) card
 * RESULTS: (For array with the size = 100'000'000)
 *  - CPU reduction:  ms
 *  - GPU reduction: ~ ms
 *  - GPU data preparation overhead (allocation and host to device copy): ~ms
 * 
 * Tests (100'000'000):
 * reduce1: ~45 ms
 * reduce2: ~23.5 ms
 * reduce3: ~18 ms
 * reduce4: ~9 ms
 * reduce4: ~5.8 ms
 */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <iostream>
#include <malloc.h>

//constexpr int32_t ARR_SIZE = 100'000'000'000;
constexpr int32_t ARR_SIZE = 100'000'000;
constexpr int32_t ALIGNMENT = 16;
constexpr float OFFSET = 1.0f;
constexpr bool PRINT_ARR = false;

// CUDA specific
constexpr int32_t CUDA_BLOCK_SIZE = 128; // amount of threads per thread block(
constexpr int32_t CUDA_GRID_SIZE = ((ARR_SIZE / 2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE); // amount of thread blocks per grid

// Helpers
void printArr(float* pArr);
void initData(float* pArr);

void reduce(float* pArr);
void reduceWithCuda(float* pDevArr, float* pDevArrOut, float* pArrOut);

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA devices

    if (!deviceCount) {
        std::cout << "CUDA-capable GPU isn't found.\n";
        return 0;
    }

    float* pArr = static_cast<float*>(_aligned_malloc(ARR_SIZE * sizeof(float), ALIGNMENT));
    if (!pArr) {
        std::cerr << "Memory allocation failed for array." << std::endl;
        return 1;
    }

    initData(pArr);

    if (PRINT_ARR) {
        std::cout << "Array: ";
        printArr(pArr);
    }

    reduce(pArr); // CPU addition

    // Partial reduction output array after parallel execution on GPU
    float* pArrOut = static_cast<float*>(_aligned_malloc(CUDA_GRID_SIZE * sizeof(float), ALIGNMENT));
    if (!pArrOut) {
        std::cerr << "Memory allocation failed for array." << std::endl;
        _aligned_free(pArr);
        return 1;
    }

    // GPU buffers
    float* pDevArr = nullptr; 
    float* pDevArrOut = nullptr; 


    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    // Allocate GPU buffer for array
    cudaError_t cudaStatus = cudaMalloc(&pDevArr, ARR_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed!\n";
        _aligned_free(pArr);
        return 1;
    }
    
    cudaStatus = cudaMalloc(&pDevArrOut, CUDA_GRID_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMalloc failed for the output array!\n";
        cudaFree(pDevArr);
        _aligned_free(pArr);
        return 1;
    }

    // Copy input array from host memory to GPU buffers
    cudaStatus = cudaMemcpy(pDevArr, pArr, ARR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!\n";
        _aligned_free(pArr);
        return 1;
    }

    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "===== GPU Reduction =====\n";
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "GPU data preparation (buffers allocation and host to device data copy) time: " << duration.count() << " ms.\n";

    reduceWithCuda(pDevArr, pDevArrOut, pArrOut);

    cudaFree(pDevArr);
    _aligned_free(pArrOut);
    _aligned_free(pArr);

    return 0;
}

void printArr(float* pVec)
{
    for (int i = 0; i < ARR_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

void initData(float* pArr)
{
    for (int i = 0; i < ARR_SIZE; ++i) {
        pArr[i] = static_cast<float>(i);
    }
}

void reduce(float* pArr)
{
    std::cout << "===== CPU Reduction =====\n";

    double sum = 0.0f; // double to avoid float precision issue

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARR_SIZE; ++i) {
        sum += pArr[i];
    }
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    std::cout << "Reduction (sum): " << sum << std::endl;

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

__global__ void reduceKernel1(float* pArr, float* pArrOut)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = pArr[i];
    __syncthreads();

    for (size_t s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        pArrOut[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceKernel2(float* pArr, float* pArrOut)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = pArr[i];
    __syncthreads();

    for (size_t s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        pArrOut[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceKernel3(float* pArr, float* pArrOut)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = pArr[i];
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        pArrOut[blockIdx.x] = sdata[0];
    }
}

__global__ void reduceKernel4(float* pArr, float* pArrOut)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = pArr[i] + pArr[i + blockDim.x];
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        pArrOut[blockIdx.x] = sdata[0];
    }
}

__device__ void warpReduce(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceKernel5(float* pArr, float* pArrOut)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = pArr[i] + pArr[i + blockDim.x];
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        pArrOut[blockIdx.x] = sdata[0];
    }
}

void reduceWithCuda(float* pDevArr, float* pDevArrOut, float* pArrOut)
{
    cudaEvent_t startKernelEvent, stopKernelEvent; // events to measure kernel execution time
    cudaEventCreate(&startKernelEvent);
    cudaEventCreate(&stopKernelEvent);

    cudaEventRecord(startKernelEvent, 0);

    reduceKernel5<<<CUDA_GRID_SIZE, CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE * sizeof(float)>>>(pDevArr, pDevArrOut);

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "addKernel launch failed:" << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    cudaEventRecord(stopKernelEvent, 0);
    cudaEventSynchronize(stopKernelEvent);

    float kernelTimeMs = 0.0f;
    cudaEventElapsedTime(&kernelTimeMs, startKernelEvent, stopKernelEvent);
    cudaEventDestroy(startKernelEvent);
    cudaEventDestroy(stopKernelEvent);

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    // Any errors encountered during the launch
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error code" << cudaStatus << std::endl;
        return;
    }

    // Copy output vector from GPU buffer to host memory
    const auto fstartTimePoint = std::chrono::high_resolution_clock::now();
    cudaStatus = cudaMemcpy(pArrOut, pDevArrOut, CUDA_GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!";
        return;
    }

    double sum = 0.0f; // double to avoid float precision issue
    for (int i = 0; i < CUDA_GRID_SIZE; ++i) {
        sum += pArrOut[i];
    }
    const auto fendTimePoint = std::chrono::high_resolution_clock::now();

    const auto fduration = std::chrono::duration_cast<std::chrono::milliseconds>(fendTimePoint - fstartTimePoint);

    std::cout << "Reduction (sum): " << sum << std::endl;
    std::cout << "Execution time (kernel): " << kernelTimeMs << " ms.\n";
    std::cout << "Execution time (final): " << fduration.count() << " ms.\n";
}