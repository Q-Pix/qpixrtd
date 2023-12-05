#ifndef __RTDStructures
#define __RTDStructures

#include "Structures.h"
#include <thrust/device_ptr.h>
#include <thrust/zip_function.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>

/*
struct compIon{
  __host__ __device__
  bool operator()(const Qpix::ION& a, const Qpix::ION& b) ;
};

// used when sorting in place
struct compTrk{
  __host__ __device__
  bool operator()(const Qpix::ION& a, const Qpix::ION& b) ;
};

// used for pixel counting after sort
struct countPixels{
  __host__ __device__
  bool operator()(const Qpix::ION& a, const Qpix::ION& b);
};
*/

/*
after the electrons (photons) are created via ionization (scintillation) they
are drifted to a pixel this structor forms the sparse matrix where the
elements are the time bin and number of elements in time bin
*/
struct Pixel_Current
{
    Pixel_Current() = default;
    __host__ __device__
    Pixel_Current(const int& pix_id, const int& trk_id, const double& time, const double& timeBinSize);

    // pixel identifiers
    int X_Pix, Y_Pix;
    int ID;

    // source trk ID for verification
    int Trk_ID;

    long unsigned int elecTime;
    unsigned int nElec;
    double t;

    double GetTimeBinSize() const {return _timeBinSize;};

    private:
        double _timeBinSize;

};

// relevant type naming for the merge
typedef thrust::device_vector<int>::iterator IntIterator;
typedef thrust::device_vector<double>::iterator DoubleIterator;
typedef thrust::tuple<IntIterator, IntIterator, DoubleIterator> Iterator3Tuple;
typedef thrust::zip_iterator<Iterator3Tuple> Zip3Iterator;

// unaryOP for the ThrustQMerge
struct single_pixelCurrent{
  single_pixelCurrent() = delete;
  single_pixelCurrent(const double& time) : _timeBin(time) {};

  // convert each ION into a pixel_current time bin
  __host__ __device__
  Pixel_Current operator()(const int& pix_id, const int& trk_id, const double& time) 
  // Pixel_Current operator()(Iterator3Tuple zip) 
  {
    return Pixel_Current(pix_id, trk_id, time, _timeBin);
  }

  double _timeBin;
};

struct pixelCurrentSum{
  __host__ __device__
  Pixel_Current operator()(Pixel_Current lhs, Pixel_Current rhs);
};

// used for finding the next iterator in the sorted sequence
struct nextPixel{
  nextPixel() = delete;
  nextPixel(thrust::device_ptr<Pixel_Current> qi) : _qion(qi) {};
  __host__ __device__
  bool operator()(const Pixel_Current& a) ;

  // member
  thrust::device_ptr<Pixel_Current> _qion;
};

// used for comparing the pixel ID and the time window
struct nextPixelTime{
  __host__ __device__
  bool operator()(const Pixel_Current& a, const Pixel_Current& b);
};

struct pix
{
    unsigned int a, b, c;
};

struct d_compPix
{
    __host__ __device__
    bool operator()(const pix& lhs, const pix& rhs);
};


#endif