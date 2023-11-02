#include "RTDCuda.h"

#include <iostream>
#include <numeric>
#include <iomanip>

#include <stdlib.h>


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


__global__ void addDiffArrays(double* point, double* step, Qpix::ION * dest, int* count, int size, int nHits)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        for(int i=0; i<nHits; ++i){
            if(count[i] >= tid + 1){
                dest[tid].x = point[i] - step[i] * (tid + 1 - count[i]);
                return;
            }
        }
        dest[tid].x = -41;
        return;
    }

};

extern "C" void launch_add_diff_arrays(double* start, double* step, Qpix::ION * dest, int* con, int size, int nHits)
{
    double* d_a, *d_b;
    Qpix::ION *d_c;
    int* d_con;

    // Allocate device memory
    cudaMalloc(&d_c, size * sizeof(Qpix::ION));

    cudaMalloc(&d_a, nHits * sizeof(double));
    cudaMalloc(&d_b, nHits * sizeof(double));
    cudaMalloc(&d_con, nHits * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_a, start, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, step, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_con, con, nHits * sizeof(int), cudaMemcpyHostToDevice);

    // std::cout << "Cuda Diff Input..\n";
    // for (int i = 0; i < nHits; i++) {
    //     std::cout << "a: " << a[i] << " " << ", b: " << b[i] << ", con: " << con[i] <<"\n";
    // }
    // std::cout << std::endl;

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addDiffArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, d_con, size, nHits);

    // Copy the result from device to host
    cudaMemcpy(dest, d_c, size * sizeof(Qpix::ION), cudaMemcpyDeviceToHost);

    // std::cout << std::fixed;
    // std::cout << std::setprecision(4);
    // std::cout << "Add Size of Diff Cuda Result: " << size << "\n";
    for (int i = 0; i < size; i++) {
        if(dest[i].x == 0)
            std::cout << "warning val at index: " << i << "\n";
    }
    // std::cout << std::endl;
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_con);
};

__global__ void makeElectron(double* start_x, double* start_y, double* start_z, double* start_t,
                             double* step_x, double* step_y, double* step_z, double* step_t,
                              double* ionx,
                            //  Qpix::ION* ions,
                             int* nElectrons, int maxElec, int size_step)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < maxElec) {
        for(int i=0; i<size_step; ++i){
            if(nElectrons[i] >= tid + 1){
                // dest[tid] = point[i] + step[i] * (tid + 1 - nElectrons[i]);
                ionx[tid] = start_x[i] - step_x[i] * (tid + 1 - nElectrons[i]);
                // ions[tid].x = start_x[i] - step_x[i] * (tid + 1 - nElectrons[i]);
                // ions[tid].y = start_y[i] - step_y[i] * (tid + 1 - nElectrons[i]);
                // ions[tid].z = start_z[i] - step_z[i] * (tid + 1 - nElectrons[i]);
                // ions[tid].t = start_t[i] - step_t[i] * (tid + 1 - nElectrons[i]);
                return;
            }
        }
        ionx[tid] = -41;
        // ions[tid].x = -41;
        // ions[tid].y = -42;
        // ions[tid].z = -42;
        // ions[tid].t = -42;
    }
}

extern "C" void makeElectrons(double* start_x, double* start_y, double* start_z, double* start_t,
                              double* step_x, double* step_y, double* step_z, double* step_t,
                              Qpix::ION* ions,
                              int* nElec, int nElecSize)
{
    // allocate memory for all of the electrons
    // Qpix::ION* d_ions;

    double* d_ions_x;
    int nElectrons = nElec[nElecSize -1];
    // cudaMalloc((void**)&d_ions, nElectrons * sizeof(Qpix::ION));  
    cudaMalloc((void**)&d_ions_x, nElectrons * sizeof(double));  

    // allocate and copy memory for the step information on each hit
    double *d_sx;
    double *d_stepx;
    // double *d_sx, *d_sy, *d_sz, *d_st;
    // double *d_stepx, *d_stepy, *d_stepz, *d_stept;
    int* d_nElec;

    // allocate the starting points
    cudaMalloc((void**)&d_sx, nElecSize * sizeof(double));  
    // cudaMalloc((void**)&d_sy, nElecSize * sizeof(double));  
    // cudaMalloc((void**)&d_sz, nElecSize * sizeof(double));  
    // cudaMalloc((void**)&d_st, nElecSize * sizeof(double));  
    cudaMalloc((void**)&d_stepx, nElecSize * sizeof(double));  
    // cudaMalloc((void**)&d_stepy, nElecSize * sizeof(double));  
    // cudaMalloc((void**)&d_stepz, nElecSize * sizeof(double));  
    // cudaMalloc((void**)&d_stept, nElecSize * sizeof(double));  
    cudaMalloc((void**)&d_nElec, nElecSize * sizeof(int));  

    // copy the starting points
    cudaMemcpy(start_x, d_sx, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(start_y, d_sy, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(start_z, d_sz, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(start_t, d_st, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(step_x, d_stepx, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(step_y, d_stepy, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(step_z, d_stepz, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(step_t, d_stept, nElecSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(nElec, d_nElec, nElecSize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(256,1,1);
    dim3 gridDim((nElectrons + blockDim.x - 1) / blockDim.x, 1, 1);
    // std::cout << "producing n blocks: " << gridDim.x << "\n";
    // makeElectron<<<gridDim, blockDim>>>(d_sx, d_sy, d_sz, d_st,
    //                                     d_stepx, d_stepy, d_stepz, d_stept,
    //                                     d_ions_x,
    //                                     // d_ions,
    //                                     d_nElec, nElectrons, nElecSize);

    int threadsPerBlock = 256;
    int blocksPerGrid = (nElectrons + threadsPerBlock - 1) / threadsPerBlock;
    // addDiffArrays<<<blocksPerGrid, threadsPerBlock>>>(d_sx, d_stepx, d_ions_x, d_nElec, nElectrons, nElecSize);

    // cudaMemcpy(ions, d_ions, nElectrons * sizeof(Qpix::ION), cudaMemcpyDeviceToHost);
    std::vector<double> vx(nElectrons);
    cudaMemcpy(vx.data(), d_ions_x, nElectrons * sizeof(double), cudaMemcpyDeviceToHost);

    // if(ions[0].x != -41)
    if(vx[0] != -41)
        std::cout << "found n ions: " << nElectrons << ", electron.x: " << vx[0] << std::endl;

    // free bird
    // cudaFree(d_ions);
    cudaFree(d_ions_x);
    cudaFree(d_sx);
    // cudaFree(d_sy);
    // cudaFree(d_sz);
    // cudaFree(d_st);
    cudaFree(d_stepx);
    // cudaFree(d_stepy);
    // cudaFree(d_stepz);
    // cudaFree(d_stept);
    cudaFree(d_nElec);
};