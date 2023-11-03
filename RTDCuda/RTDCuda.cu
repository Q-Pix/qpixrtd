#include "RTDCuda.h"

#include <iostream>
#include <numeric>
#include <iomanip>

#include <stdlib.h>

// #define MAX_DEPTH 2048
#define INSERTION_SORT 32

// reimplemented from Qpix::Functions
__device__ inline int ID_Encoder(const int& pix_x, const int& pix_y)
{
    return (int)(pix_x*10000+pix_y);
}

__device__ void DiffuseIon(Qpix::ION* qp_ion, Qpix::Qpix_Paramaters *Qpix_params, double& rand_x, double& rand_y, double& rand_z)
{
    double T_drift = qp_ion->z / Qpix_params->E_vel;
    // diffuse the electrons position
    double sigma_T = sqrt(2*Qpix_params->DiffusionT*T_drift);
    double sigma_L = sqrt(2*Qpix_params->DiffusionL*T_drift);

    double px = qp_ion->x + sigma_T * rand_x; 
    double py = qp_ion->y + sigma_T * rand_y; 
    double pz = rand_z + sigma_L * rand_x; 

    // convert the electrons x,y to a pixel index
    int Pix_Xloc = (int) ceil(px / Qpix_params->Pix_Size);
    int Pix_Yloc = (int) ceil(py / Qpix_params->Pix_Size);

    qp_ion->Pix_ID = ID_Encoder(Pix_Xloc, Pix_Yloc);
    qp_ion->time = qp_ion->t + ( pz / Qpix_params->E_vel );
}

__global__ void makeQPixIons(double* start_x, double* step_x, double *start_y, double *step_y,
                             double* start_z, double* step_z, double *start_t, double *step_t, 
                             Qpix::ION * dest, int* count, int size, int nHits,
                             Qpix::Qpix_Paramaters qp_params,
                             curandState* state)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;


    if (tid < size) {

        curandState localState = state[tid]; 
        double rand_x = curand_normal_double(&localState);
        double rand_y = curand_normal_double(&localState);
        double rand_z = curand_normal_double(&localState);

        for(int i=0; i<nHits; ++i){
            if(count[i] >= tid + 1){
                dest[tid].x = start_x[i] - step_x[i] * (tid + 1 - count[i]);
                dest[tid].y = start_y[i] - step_y[i] * (tid + 1 - count[i]);
                dest[tid].z = start_z[i] - step_z[i] * (tid + 1 - count[i]);
                dest[tid].t = start_t[i] - step_t[i] * (tid + 1 - count[i]);

                DiffuseIon(&dest[tid], &qp_params, rand_x, rand_y, rand_z);
                return;
            }
        }
        dest[tid].x = -41;
        dest[tid].y = -41;
        dest[tid].z = -41;
        dest[tid].t = -41;
        return;
    }

};

extern "C" void Launch_Make_QPixIons(double* start_x, double* step_x, double *start_y, double *step_y, 
                                     double* start_z, double* step_z, double *start_t, double *step_t, 
                                     Qpix::ION * dest_ions, int* con, int size, int nHits,
                                     Qpix::Qpix_Paramaters qp_params, int seed)
{
    // std::cout << "running with diffusion: " << qp_params->DiffusionT << "\n";
    // exit(-1);

    double *d_start_x, *d_step_x;
    double *d_start_y, *d_step_y;
    double *d_start_z, *d_step_z;
    double *d_start_t, *d_step_t;
    Qpix::ION *d_c;

    int *d_con;

    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // random number generation
    curandState *d_devStates;
    cudaMalloc((void **)&d_devStates, size * sizeof(curandState));
    setup_normal_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_devStates, size, seed);

    // Allocate device memory for ION destination
    auto err = cudaMalloc(&d_c, size * sizeof(Qpix::ION));
    // if(err != 0){std::cout << "error.\n"; exit(-1);};

    cudaMalloc(&d_con, nHits * sizeof(int));
    cudaMalloc(&d_start_x, nHits * sizeof(double));
    cudaMalloc(&d_step_x, nHits * sizeof(double));
    cudaMalloc(&d_start_y, nHits * sizeof(double));
    cudaMalloc(&d_step_y, nHits * sizeof(double));
    cudaMalloc(&d_start_z, nHits * sizeof(double));
    cudaMalloc(&d_step_z, nHits * sizeof(double));
    cudaMalloc(&d_start_t, nHits * sizeof(double));
    cudaMalloc(&d_step_t, nHits * sizeof(double));


    // Copy input data from host to device
    cudaMemcpy(d_con, con, nHits * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_x, start_x, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step_x, step_x, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_y, start_y, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step_y, step_y, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_z, start_z, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step_z, step_z, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_t, start_t, nHits * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step_t, step_t, nHits * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the working kernel
    makeQPixIons<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_start_x, d_step_x, d_start_y, d_step_y, 
                                                       d_start_z, d_step_z, d_start_t, d_step_t, 
                                                       d_c, d_con, size, nHits,
                                                       qp_params,
                                                       d_devStates);

    // Copy the result from device to host
    cudaMemcpy(dest_ions, d_c, size * sizeof(Qpix::ION), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < size; i++) {
    //     if(dest_ions[i].x == 0)
    //         std::cout << "warning val at index: " << i << "\n";
    // }
    // Free device memory
    cudaFree(d_devStates);

    cudaFree(d_start_x);
    cudaFree(d_step_x);
    cudaFree(d_c);

    cudaFree(d_start_y);
    cudaFree(d_step_y);
    cudaFree(d_start_z);
    cudaFree(d_step_z);
    cudaFree(d_start_t);
    cudaFree(d_step_t);

    cudaFree(d_con);
};

/* modified from: https://docs.nvidia.com/cuda/curand/device-api-overview.html */
__global__ void setup_normal_kernel(curandState* state,
                                    int nElectrons, int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread uses 3 different dimensions */
    // curand_init(0, id, 0, &state[id]);
    // curand(&state[id]);
    if(id < nElectrons){
        curand_init(seed, 0, 0, &state[id]);
    }
}

// prototyping the sort function
extern "C" void Launch_QuickSort(unsigned int* h_input_keys, unsigned int* h_output_keys, const int size, const int max_depth)
{
    std::cout << "kernel launch from host with size: " << size << "\n";
    unsigned int *d_input_keys;
    cudaMalloc((void**)&d_input_keys, size * sizeof(unsigned int));

    cudaMemcpy(d_input_keys, h_input_keys, size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch the working kernel
    cdp_simple_quicksort<<<1, 1>>>(d_input_keys, 0, size-1, 0, max_depth);

    cudaMemcpy(h_output_keys, d_input_keys, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_input_keys);
}

/* Below working examples taking from cuda-samples */
////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right) {
  for (int i = left; i <= right; ++i) {
    unsigned min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for (int j = i + 1; j <= right; ++j) {
      unsigned val_j = data[j];

      if (val_j < min_val) {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if (i != min_idx) {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
                                     int depth, const int max_depth) {
  // If we're too deep or there are few elements left, we use an insertion
  // sort...
  if (depth >= max_depth || right - left <= INSERTION_SORT) {
    selection_sort(data, left, right);
    return;
  }

  unsigned int *lptr = data + left;
  unsigned int *rptr = data + right;
  unsigned int pivot = data[(left + right) / 2];

  // Do the partitioning.
  while (lptr <= rptr) {
    // Find the next left- and right-hand values to swap
    unsigned int lval = *lptr;
    unsigned int rval = *rptr;

    // Move the left pointer as long as the pointed element is smaller than the
    // pivot.
    while (lval < pivot) {
      lptr++;
      lval = *lptr;
    }

    // Move the right pointer as long as the pointed element is larger than the
    // pivot.
    while (rval > pivot) {
      rptr--;
      rval = *rptr;
    }

    // If the swap points are valid, do the swap!
    if (lptr <= rptr) {
      *lptr++ = rval;
      *rptr-- = lval;
    }
  }

  // Now the recursive part
  int nright = rptr - data;
  int nleft = lptr - data;

  // Launch a new block to sort the left part.
  if (left < (rptr - data)) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1, max_depth);
    cudaStreamDestroy(s);
  }

  // Launch a new block to sort the right part.
  if ((lptr - data) < right) {
    cudaStream_t s1;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1, max_depth);
    cudaStreamDestroy(s1);
  }
}