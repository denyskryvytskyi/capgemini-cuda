/**
 * TASK: Sorting with Thrust
 * NOTE: Implemented and tested on Windows 10 with NVIDIA GTX 1050 (laptop) card
 * RESULTS: ()
 *  - CPU sort: ~2880 ms
 *  - GPU (Thrust) sort: ~133 ms
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <iostream>
#include <malloc.h>
#include <random>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

constexpr int32_t ARR_SIZE = 100'000'000;
constexpr int32_t ARR_MIN_VAL = -100;
constexpr int32_t ARR_MAX_VAL = 100;
constexpr int32_t ALIGNMENT = 16;
constexpr bool PRINT_ARR = false;

// Helpers
void initData(int32_t* pArr);
void printArr(int32_t* pArr);

void sort(int32_t* pArr);
void sortWithCuda(thrust::device_vector<int32_t>& devArr);

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get the number of CUDA devices

    if (!deviceCount) {
        std::cout << "CUDA-capable GPU isn't found.\n";
        return 0;
    }

    int32_t* pArr = static_cast<int32_t*>(_aligned_malloc(ARR_SIZE * sizeof(int32_t), ALIGNMENT));
    if (!pArr) {
        std::cerr << "Memory allocation failed for array." << std::endl;
        return 1;
    }

    initData(pArr);

    thrust::device_vector<int32_t> devArr(pArr, pArr + ARR_SIZE);   // init thrust array

    if (PRINT_ARR) {
        std::cout << "Array: ";
        printArr(pArr);
    }

    sort(pArr);
    _aligned_free(pArr);

    sortWithCuda(devArr);

    return 0;
}

void initData(int32_t* pArr)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(ARR_MIN_VAL, ARR_MAX_VAL); // range [min, max]

    for (int i = 0; i < ARR_SIZE; ++i) {
        pArr[i] = distr(gen);
    }
}

void printArr(int32_t* pVec)
{
    for (int i = 0; i < ARR_SIZE; ++i) {
        std::cout << pVec[i] << " ";
    }

    std::cout << std::endl;
}

void sort(int32_t* pArr)
{
    std::cout << "===== CPU Sorting =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    std::sort(pArr, pArr + ARR_SIZE);
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARR) {
        std::cout << "Sorted array: ";
        printArr(pArr);
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}

void sortWithCuda(thrust::device_vector<int32_t>& devArr)
{
    std::cout << "===== GPU Sorting =====\n";

    const auto startTimePoint = std::chrono::high_resolution_clock::now();
    thrust::sort(devArr.begin(), devArr.end());
    const auto endTimePoint = std::chrono::high_resolution_clock::now();

    if (PRINT_ARR) {
        std::cout << "Sorted array: ";
        thrust::copy(devArr.begin(), devArr.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTimePoint - startTimePoint);
    std::cout << "Execution time: " << duration.count() << " ms.\n";
}