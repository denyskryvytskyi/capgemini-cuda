/**
 * TASK: Reduction Operations Using CUDA
 * NOTE: Implemented and tested on Windows 10 with NVIDIA GTX 1050 (laptop) card
 * RESULTS: (For array with the size = 100'000'000)
 * Windows results (my laptop with Nvidia GTX 1050 and Intel Core i7-7700HQ):
 *  - CPU reduction: ~125 ms
 *  - GPU reduction: ~5.8 ms
 *      - CPU final computation time based on GPU computation results: ~1 ms
 *      - Overall results: ~6.8 ms (x18 faster)
 *  - GPU data preparation overhead (allocation and host to device copy): ~580 ms
 * Linux results (this PC with Nvidia Tesla m60 and Intel Xeon CPU E5-2686):
 *  - CPU reduction: ~118 ms
 *  - GPU reduction: ~3.5 ms
 *      - CPU final computation based on GPU computation results: ~1 ms
 *      - Overall results: ~5.5 ms (x21 faster)
 *  - GPU data preparation overhead (allocation and host to device copy): ~174 ms
 **/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <iostream>
#include <malloc.h>

constexpr int32_t ARR_SIZE = 100'000'000;
constexpr int32_t ALIGNMENT = 16;
constexpr bool PRINT_ARR = false;

// CUDA specific
constexpr int32_t CUDA_BLOCK_SIZE = 128; // Amount of threads per thread block(
constexpr int32_t CUDA_GRID_SIZE = ((ARR_SIZE / 2 + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE); // Amount of thread blocks per grid
constexpr int32_t CUDA_WARP_SIZE = 32;

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

    reduce(pArr);

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

// Last warp unrolling
__device__ void warpReduce(volatile float* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

/**
* Main kernel for reduction operation.
* We make the first add while loading to the shared memory,
* that's why the final thread blocks amount (CUDA_GRID_SIZE) is divided by 2.
* Then we make a tree-based sum up per thread block.
* For the final warp we make warp unrolling for the final result.
*/
__global__ void reduceKernel(float* pArr, float* pArrOut)
{
    extern __shared__ float sdata[];                                // Shared data accessible by all threads in a thread block

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;   // Index of the processing element from the array
    sdata[tid] = pArr[i] + pArr[i + blockDim.x];                    // Make first add while loading to shared data
    __syncthreads();                                                // Wait till all threads in the block finish loading to shared data

    // Tree-based sum up
    for (unsigned int s = blockDim.x / 2; s > CUDA_WARP_SIZE; s >>= 1) {
        // Threads in block sequentially access elements and sum up with elements by stride
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();    // Wait till all threads in block finish instructions
    }

    // For the last warp we don't need (tid < s) check and sync because all threads in warp process simultaneously
    if (tid < CUDA_WARP_SIZE) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        pArrOut[blockIdx.x] = sdata[0]; // Final sum of the elements in thread block are in the first element of the sdata
    }
}

void reduceWithCuda(float* pDevArr, float* pDevArrOut, float* pArrOut)
{
    cudaEvent_t startKernelEvent, stopKernelEvent; // Events to measure kernel execution time
    cudaEventCreate(&startKernelEvent);
    cudaEventCreate(&stopKernelEvent);

    cudaEventRecord(startKernelEvent, 0);

    reduceKernel<<<CUDA_GRID_SIZE, CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE * sizeof(float)>>>(pDevArr, pDevArrOut);

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
