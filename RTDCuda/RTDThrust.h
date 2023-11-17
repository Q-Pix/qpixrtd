#ifndef __RTDThrust
#define __RTDThrust

#include "Structures.h"

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

// working functions for sorting the qpix ions in place
void ThrustQSort(Qpix::ION* qion, int size);

// used for time sorting Qpix::IONs
struct compIon{
  __host__ __device__
  bool operator()(const Qpix::ION& a, const Qpix::ION& b) 
  {
    // sort by increasing pixel, and then by increasing time
    if(a.Pix_ID == b.Pix_ID) 
      return a.t < b.t;
    else
      return a.Pix_ID < b.Pix_ID;
  };
};

// used for pixel counting after sort
struct countPixels{
  __host__ __device__
  bool operator()(const Qpix::ION& a, const Qpix::ION& b) 
  {
    // pixels are the same iff they have same the ID
    return a.Pix_ID == b.Pix_ID;
  }
};


__host__ __device__
void sID_Decoder(const int& ID, int& Xcurr, int& Ycurr);

// after the electrons (photons) are created via ionization (scintillation) they are drifted to a pixel
// this structor forms the sparse matrix where the elements are the time bin and number of elements in time bin
struct Pixel_Current
{
    Pixel_Current() = default;
    __host__ __device__
    Pixel_Current(const Qpix::ION& qion, const double& timeBinSize);

    // pixel identifiers
    int X_Pix, Y_Pix;
    int ID;

    long unsigned int elecTime;
    unsigned int nElec;
    double t;

    double GetTimeBinSize() const {return _timeBinSize;};

    private:
        double _timeBinSize;

};

// unaryOP for the ThrustQMerge
struct single_pixelCurrent{
  single_pixelCurrent() = delete;
  single_pixelCurrent(const double& time) : _timeBin(time) {};

  // convert each ION into a pixel_current time bin
  __host__ __device__
  Pixel_Current operator()(const Qpix::ION& a) 
  {
    return Pixel_Current(a, _timeBin);
  }

  double _timeBin;
};

struct pixelCurrentSum{
  __host__ __device__
  Pixel_Current operator()(Pixel_Current lhs, Pixel_Current rhs)
  {
      // only add the electron counts if it's the same pixel
      if(lhs.ID == rhs.ID){
        // they're in the same time bin
        if(lhs.elecTime == rhs.elecTime)
        {
            rhs.nElec += lhs.nElec;
        }
      }
    return rhs;
  }
};

// used for finding the next iterator in the sorted sequence
struct nextPixel{
  nextPixel() = delete;
  nextPixel(thrust::device_ptr<Pixel_Current> qi) : _qion(qi) {};
  __host__ __device__
  bool operator()(const Pixel_Current& a) 
  {
    // we've found the next pixel is the two IDs are different
    return a.ID != _qion.get()->ID;
  }
  thrust::device_ptr<Pixel_Current> _qion;
};

// used for comparing the pixel ID and the time window
struct nextPixelTime{
  __host__ __device__
  bool operator()(const Pixel_Current& a, const Pixel_Current& b) 
  {
    return a.ID == b.ID && a.elecTime == b.elecTime;
  }
};

// working function to make the combined output we need
thrust::device_vector<Pixel_Current> ThrustQMerge(Qpix::ION* qion, int size);

#endif