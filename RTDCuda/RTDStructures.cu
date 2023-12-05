#include "RTDStructures.h"

/*
// used for time sorting Qpix::IONs
__host__ __device__
bool compIon::operator()(const Qpix::ION& a, const Qpix::ION& b) 
{
  // sort by increasing pixel, then increasing track ID, and finally by increasing time
  if(a.Pix_ID != b.Pix_ID)
    return a.Pix_ID < b.Pix_ID;
  else if(a.Trk_ID != b.Trk_ID)
    return a.Trk_ID < b.Trk_ID;
  else
    return a.t < b.t;
};

bool compTrk::operator()(const Qpix::ION& a, const Qpix::ION& b) 
{
  // sort by increasing pixel, then increasing track ID, and finally by increasing time
  if(a.Pix_ID != b.Pix_ID)
    return a.Pix_ID < b.Pix_ID;
  else
    return a.t < b.t;
};

__host__ __device__
bool countPixels::operator()(const Qpix::ION& a, const Qpix::ION& b) 
{
  // pixels are the same iff they have same the ID
  return a.Pix_ID == b.Pix_ID;
}

*/

// used in inclusive scan to quickly add electrons from same track / same pixel / same time bin
__host__ __device__
Pixel_Current pixelCurrentSum::operator()(Pixel_Current lhs, Pixel_Current rhs)
{
    // only add the electron counts if it's the same pixel from the same trackID
    if(lhs.ID == rhs.ID && lhs.Trk_ID == rhs.Trk_ID){
      // they're in the same time bin
      if(lhs.elecTime == rhs.elecTime)
      {
          rhs.nElec += lhs.nElec;
      }
    }
  return rhs;
}

__host__ __device__
bool nextPixel::operator()(const Pixel_Current& a) 
{
  // we've found the next pixel is the two IDs are different
  return a.ID != _qion.get()->ID;
}

// unique operator of sorting looks at pixel ID / parent track ID / electron bin time
__host__ __device__
bool nextPixelTime::operator()(const Pixel_Current& a, const Pixel_Current& b)
{
  return a.ID == b.ID && a.elecTime == b.elecTime && a.Trk_ID == b.Trk_ID;
}


/// Pixel Current Definitions
// pair decoders and encoders
__host__ __device__
void sID_Decoder(const int& ID, int& Xcurr, int& Ycurr){
    double PixID = ID/10000.0, fractpart, intpart;
    fractpart = modf (PixID , &intpart);
    Xcurr = (int) round(intpart);
    Ycurr = (int) round(fractpart*10000); 
}

__host__ __device__
Pixel_Current::Pixel_Current(const int& pix_id, const int& trk_id, const double& time, const double& timeBinSize) : _timeBinSize(timeBinSize) 
{
    ID = pix_id;
    t = time;
    Trk_ID = trk_id;
    sID_Decoder(ID, X_Pix, Y_Pix);
    elecTime = (long unsigned int)( time / _timeBinSize);
    nElec = 1;
};

__host__ __device__
bool d_compPix::operator()(const pix& lhs, const pix& rhs)
{return lhs.a > rhs.a;};
