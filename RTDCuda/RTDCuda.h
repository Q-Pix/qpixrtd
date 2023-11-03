// list of CUDA functions to make things go fast
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// random things
#include "curand.h"
#include <curand_kernel.h>

#include "Structures.h"

#define THREADS_PER_BLOCK 512

__device__ inline int ID_Encoder(const int& pix_x, const int& pix_y);

__device__ void DiffuseIon(Qpix::ION* qp_ion, Qpix::Qpix_Paramaters *Qpix_params, double& rand_x, double& rand_y,
                           double& rand_z);

// RTD Things to make life go fast
__global__ void makeQPixIons(double* start_x, double* step_x, double *start_y, double *step_y,
                             double* start_z, double* step_z, double *start_t, double *step_t, 
                             Qpix::ION* dest, int* count, int size, int nHits,
                             Qpix::Qpix_Paramaters qp_params,
                             curandState* state);

extern "C" void Launch_Make_QPixIons(double* start_x, double* step_x, double *start_y, double *step_y, 
                                     double* start_z, double* step_z, double *start_t, double *step_t, 
                                     Qpix::ION* dest, int* con, int size, int nHits,
                                     Qpix::Qpix_Paramaters qp_params, int seed);


// once each thread makes an ion, it can call device functions to put these ions where we want them to go
__global__ void setup_normal_kernel(curandState *state,
                                    int nElectrons, int seed);

// prototyping the sort function
extern "C" void Launch_QuickSort(unsigned int* h_input_keys, unsigned int* h_output_keys, const int size, const int max_depth);

/* examples taken from cuda-examples github */
__device__ void selection_sort(unsigned int *data, int left, int right);
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right,
                                     int depth, const int max_depth);