# Capgemini CUDA tasks
Tasks were implemented and tested in Windows 10 with the Visual Studio CUDA Integration tool and nvcc compiler.

## Getting Started
- Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
- [Windows] Install Visual Studio 2022 (we need msvc compiler to compile host code and link with device code for final executable).
- [Linux] Install gcc compiler.

Compile code using **nvcc** compiler:

`nvcc task_<n>.cu -o task_<n> -O3`