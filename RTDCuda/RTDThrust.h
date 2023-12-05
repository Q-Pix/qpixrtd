#ifndef __RTDThrust
#define __RTDThrust

#include "Structures.h"
#include "RTDStructures.h"

// extra thrust includes
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/unique.h>
#include <thrust/iterator/reverse_iterator.h>
#include "thrust/device_malloc.h"
#include "thrust/device_free.h"

// testing functions
extern "C" void Launch_ThrustSort(unsigned int* a, int size);
void ThrustSort(unsigned int* h_a, unsigned int* d_a, int size);

// extern "C" void Launch_ThrustSortStruct(pix* p, int size);
// void ThrustSortStruct(pix* h_p, pix* d_p, int size);

// working functions for sorting the qpix ions in place
void ThrustQSort(int* d_Pix_ID, int* d_Trk_ID, double* time, int* count, int nHits, int nElectrons);

__host__ __device__
void sID_Decoder(const int& ID, int& Xcurr, int& Ycurr);

// working function to make the combined output we need
thrust::device_vector<Pixel_Current> ThrustQMerge(int* d_Pix_ID, int* d_Trk_ID, double* time, int* count, int size);

extern "C" void Launch_ThrustTestSort();
void ThrustTestSort();

#endif