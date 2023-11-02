// list of CUDA functions to make things go fast
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Structures.h"

__global__ void addDiffArrays(double* point, double* step, double *start_y, double *step_y,
                              double* start_z, double* step_z, double *start_t, double *step_t, 
                              Qpix::ION* dest, int* count, int size, int nHits);
extern "C" void launch_add_diff_arrays(double* start, double* step, double *start_y, double *step_y, 
                                       double* start_z, double* step_z, double *start_t, double *step_t, 
                                       Qpix::ION* dest, int* con, int size, int nHits);


// RTD Things to make life go fast
__global__ void makeElectron(double* start_x, double* start_y, double* start_z, double* start_t,
                             double* step_x, double* step_y, double* step_z, double* step_t,
                             Qpix::ION* ions,
                             int* nElectrons, int maxElec, int size_step);
extern "C" void makeElectrons(double* start_x, double* start_y, double* start_z, double* start_t,
                              double* step_x, double* step_y, double* step_z, double* step_t,
                              Qpix::ION* ions,
                              int* nElec, int nElecSize);