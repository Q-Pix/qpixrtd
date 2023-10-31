#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void addArrays(int* a, int* b, int* c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

extern "C" void launch_add_arrays(int* a, int* b, int* c, int size) {
    int* d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Cuda Result..\n";
    for (int i = 0; i < size; i++) {
        std::cout << "a: " << a[i] << " " << ", b: " << b[i] << "\n";
    }
    std::cout << std::endl;

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "adding blocks: " << blocksPerGrid << "\n";
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy the result from device to host
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Cuda Result: ";
    for (int i = 0; i < size; i++) {
        std::cout << c[i] << std::endl;
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
