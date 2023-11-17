#include "RTDThrust.h"

#include <cstdlib>
#include <chrono>

extern "C" void Launch_ThrustSort(unsigned int* a, int size)
{
    unsigned int* d_vals;
    cudaMalloc((void**)&d_vals, size * sizeof(unsigned int));
    cudaMemcpy(d_vals, a, size * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // kernel launch
    ThrustSort(a, d_vals, size);
}

void ThrustSort(unsigned int* h_a, unsigned int* d_a, int size)
{
    auto start = std::chrono::high_resolution_clock::now();
    thrust::device_ptr<unsigned int> td_vals(d_a);
    thrust::sort(td_vals, td_vals+size);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "launch thrust call time: " << duration.count() << "\n";

    cudaMemcpy(h_a, d_a, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // print the sorted result
    // if(err != 0)std::cout << "err.\n";
    // std::cout << "out keys: { ";
    // for(int i=0; i<size; i++) std::cout << h_a[i] << " ";
    // std::cout << "}\n";
}


// working qpix sorting functions, assume that the ion memory is already allocated, and we just
// need to make the memory a thrust vector
void ThrustQSort(Qpix::ION* qion, int size)
{
    thrust::device_ptr<Qpix::ION> d_qion_ptr(qion);
    thrust::sort(d_qion_ptr, d_qion_ptr+size, compIon());
}

thrust::device_vector<Pixel_Current> ThrustQMerge(Qpix::ION* qion, int size)
{
    thrust::device_ptr<Qpix::ION> d_qion_ptr(qion);
    // int count = thrust::unique_count(d_qion_ptr, d_qion_ptr+size, countPixels());

    // hard code 10 ns, for now
    double binSize = 10e-9;
    // thrust::device_vector<Pixel_Current> d_pixel_current = thrust::device_malloc<Pixel_Current>(size);
    thrust::device_vector<Pixel_Current> d_pixel_current(size);
    thrust::transform(d_qion_ptr, d_qion_ptr+size, d_pixel_current.begin(), single_pixelCurrent(binSize));
    thrust::inclusive_scan(d_pixel_current.begin(), d_pixel_current.end(), d_pixel_current.begin(), pixelCurrentSum());

    // down select the highest unique values in this list! 
    // go in reverse since the maximum values are on the 'right'
    auto d_uniq_pixel_current = thrust::unique(d_pixel_current.rbegin(), d_pixel_current.rend(), nextPixelTime());
    int uniq_length = thrust::distance(d_pixel_current.rbegin(), d_uniq_pixel_current);
    std::cout << "uniq len: " << uniq_length << "\n";

    thrust::device_vector<Pixel_Current> uniq_data(uniq_length);
    thrust::copy(d_pixel_current.rbegin(), d_uniq_pixel_current, uniq_data.begin());

    // int nSum = 0;
    // int nPixels = 0;
    // auto curItr = d_pixel_current;
    // while(nPixels < count){
    //     auto nextItr = thrust::find_if(curItr, d_pixel_current+size, nextPixel(curItr));
    //     // sum the whole pixel here
    //     nPixels += 1;
    //     nSum += thrust::distance(curItr, nextItr);
    //     thrust::inclusive_scan(curItr, nextItr, curItr, pixelCurrentSum());
    //     curItr = nextItr;
    // }

    return uniq_data;
}

// pair decoders and encoders
__host__ __device__
void sID_Decoder(const int& ID, int& Xcurr, int& Ycurr){
    double PixID = ID/10000.0, fractpart, intpart;
    fractpart = modf (PixID , &intpart);
    Xcurr = (int) round(intpart);
    Ycurr = (int) round(fractpart*10000); 
}

__host__ __device__
Pixel_Current::Pixel_Current(const Qpix::ION& qion, const double& timeBinSize) : _timeBinSize(timeBinSize) 
{
    ID = qion.Pix_ID;
    t = qion.time;
    sID_Decoder(qion.Pix_ID, X_Pix, Y_Pix);
    elecTime = (long unsigned int)( qion.t / _timeBinSize);
    nElec = 1;
};
