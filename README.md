# Capgemini CUDA tasks
Tasks were implemented and tested in Windows 10 with the Visual Studio CUDA Integration tool and NVCC compiler.

Tasks list:
- vectors addition;
- matrix multiplication: with tiled shared memory usage and matrix transposition;
- reduction (sum) with a custom kernel and Nvidia Thrust library for performance comparison;
- sorting using the Nvidia Thrust library.

## Getting Started
- Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
- [Windows] Install Visual Studio 2022 (we need MSVC compiler to compile host code and link with device code for the final executable).
- [Linux] Install gcc compiler.

Compile code using **nvcc** compiler:

`nvcc <program_name>.cu -o <program_name> -O3`

Run:
`./<program_name>`
