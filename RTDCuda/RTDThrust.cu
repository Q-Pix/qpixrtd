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
    /* 5613 us for 13e6 keys */
    // auto start = std::chrono::high_resolution_clock::now();
    thrust::device_ptr<unsigned int> td_vals(d_a);
    thrust::sort(td_vals, td_vals+size);
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "launch thrust call time: " << duration.count() << "\n";

    /* 5197 us for 13e6 keys */
    // start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_a, d_a, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // stop = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "thrust copy time: " << duration.count() << "\n";

    // print the sorted result
    // if(err != 0)std::cout << "err.\n";
    // std::cout << "out keys: { ";
    // for(int i=0; i<size; i++) std::cout << h_a[i] << " ";
    // std::cout << "}\n";
}

// extern "C" void Launch_ThrustSortStruct(pix* a, int size)
// {
//     pix* d_vals;
//     cudaMalloc((void**)&d_vals, size * sizeof(pix));
//     cudaMemcpy(d_vals, a, size * sizeof(pix), cudaMemcpyHostToDevice);
//     ThrustSortStruct(a, d_vals, size);
//     cudaFree(d_vals);
// }

// void ThrustSortStruct(pix* h_p, pix* d_p, int size)
// {
//     auto start = std::chrono::high_resolution_clock::now();
//     thrust::device_ptr<pix> td_vals(d_p);
//     thrust::sort(td_vals, td_vals+size, d_compPix());
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//     std::cout << "launch thrust sort struct call time: " << duration.count() << "\n";
//     cudaMemcpy(h_p, d_p, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
// }

typedef thrust::tuple<int, double> PixTuple;
typedef thrust::device_vector<PixTuple>::iterator TupleIterator;

typedef thrust::device_vector<int>::iterator IntIterator;
typedef thrust::device_vector<double>::iterator DoubleIterator;
// typedef thrust::tuple<IntIterator, DoubleIterator> IteratorTuple;
typedef thrust::tuple<int, int> IntTuple;
typedef thrust::tuple<IntIterator, IntIterator> IteratorTuple;
typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

struct TupleComp
{
    __host__ __device__
    inline bool operator()(const IntTuple& t1, const IntTuple& t2)
    {
        if(thrust::get<0>(t1) < thrust::get<0>(t2))
            return true;
        if(thrust::get<0>(t1) > thrust::get<0>(t2))
            return false;
        return thrust::get<1>(t1) < thrust::get<1>(t2);
    }
};
typedef thrust::tuple<int, int, double> threetup;
struct TupleDoubleComp
{
    __host__ __device__
    inline bool operator()(const threetup& t1, const threetup& t2)
    {
        // sort by pixel
        if(thrust::get<0>(t1) != thrust::get<0>(t2))
            return thrust::get<0>(t1) < thrust::get<0>(t2);
        // sort by trk_id
        if(thrust::get<1>(t1) != thrust::get<1>(t2))
            return thrust::get<1>(t1) < thrust::get<1>(t2);
        // sort by time
        else
            return thrust::get<2>(t1) < thrust::get<2>(t2);
    };
};

struct s
{
    int a,b;
};

struct compVec
{
    __host__ __device__
    inline bool operator()(const s &a, const s &b){
        if(a.a < b.a) return true;
        if(a.a > b.a) return false;
        if(a.b < b.b) return true;
        return a.b > b.b;
    };
};

// helping function to sort the index
struct arb_sort_functor
{
  ZipIterator data;
  // take in index from
//   template <typename Tup>
  __host__ __device__
  void operator()(const int& start, const int& fin)
  {
    // int start = thrust::get<0>(T);
    // int fin = thrust::get<1>(T);
    thrust::sort(thrust::device, data+start, data+fin-1, TupleComp());
  }
};

// working qpix sorting functions, assume that the ion memory is already allocated, and we just
// need to make the memory a thrust vector
void ThrustQSort(int* d_Pix_ID, int* d_Trk_ID, double* time, int* hits, int nHits, int nElectrons)
{

    thrust::device_vector<int> d_pixid_ptr(d_Pix_ID, d_Pix_ID+nElectrons-1); // gives the range of the sort
    thrust::device_vector<int> d_trkid_ptr(d_Trk_ID, d_Trk_ID+nElectrons-1); 
    thrust::device_vector<double> d_time_ptr(time, time+nElectrons-1); 

    auto pix_begin = thrust::make_zip_iterator(thrust::make_tuple(d_pixid_ptr.begin(), d_trkid_ptr.begin(), d_time_ptr.begin()));
    auto pix_end = thrust::make_zip_iterator(thrust::make_tuple(d_pixid_ptr.end(), d_trkid_ptr.end(), d_time_ptr.end()));
    thrust::sort(pix_begin, pix_end, TupleDoubleComp());

    // allocating
    // std::cout << "allocating nHits: " << nHits << "\n";

    // for loop sort
    // thrust::device_ptr<int> d_hit_ptr(hits); // gives the range of the sort
    // std::cout << "looping..\n";
    // for(int i=0; i<nHits; ++i){
    //     int start;
    //     if(i == 0)
    //         start = 0;
    //     else
    //         start = d_hit_ptr[i-1];
    //     int fin = d_hit_ptr[i];
    //     std::cout << "start: " << start << ", fin: " << fin << "\n";
    //     thrust::sort(thrust::device, zip_itr+start, zip_itr+fin-1, TupleComp());
    // }

    // memory allocation issues here
    // thrust::device_vector<int> d_idxs(hits, hits+nHits-1); // total number of hits at this index
    // thrust::device_vector<int> d_idxs_1(nHits, 0); // total number of hits at previous index
    // thrust::copy(d_idxs.begin(), d_idxs.end()-1, d_idxs_1.begin()+1);

    // ZipIterator zip_itr = thrust::make_zip_iterator(thrust::make_tuple(d_pixid_ptr.begin(), d_time_ptr.begin()));
    // arb_sort_functor f = {zip_itr};

    // auto tup_begin = thrust::make_zip_iterator(thrust::make_tuple(d_idxs.begin(), d_idxs_1.begin()));
    // auto tup_end = thrust::make_zip_iterator(thrust::make_tuple(d_idxs.end(), d_idxs_1.end()));
    // thrust::for_each(tup_begin, tup_end, f);

}

thrust::device_vector<Pixel_Current> ThrustQMerge(int* d_Pix_ID, int* d_Trk_ID, double* time, int* count, int size)
{
    // hard code 30 ns, for now
    double binSize = 30e-9;
    thrust::device_vector<Pixel_Current> d_pixel_current(size);

    // thrust::device_vector<Pixel_Current> d_pixel_current = thrust::device_malloc<Pixel_Current>(size);
    thrust::device_vector<int> d_pixid_ptr(d_Pix_ID, d_Pix_ID+size-1); 
    thrust::device_vector<int> d_trkid_ptr(d_Trk_ID, d_Trk_ID+size-1); 
    thrust::device_vector<double> d_time_ptr(time, time+size-1); 

    // zip and rip
    std::cout << "transforming..\n";
    auto zip_start = thrust::make_zip_iterator(thrust::make_tuple(d_pixid_ptr.begin(), d_trkid_ptr.begin(), d_time_ptr.begin()));
    auto zip_fin = thrust::make_zip_iterator(thrust::make_tuple(d_pixid_ptr.end(), d_trkid_ptr.end(), d_time_ptr.end())); 
    thrust::transform(zip_start, zip_fin, d_pixel_current.begin(), thrust::make_zip_function(single_pixelCurrent(binSize)));
    std::cout << "scanning..\n";
    thrust::inclusive_scan(d_pixel_current.begin(), d_pixel_current.end(), d_pixel_current.begin(), pixelCurrentSum());

    // down select the highest unique values in this list! 
    // go in reverse since the maximum values are on the 'right'
    auto d_uniq_pixel_current = thrust::unique(d_pixel_current.rbegin(), d_pixel_current.rend(), nextPixelTime());
    int uniq_length = thrust::distance(d_pixel_current.rbegin(), d_uniq_pixel_current);

    thrust::device_vector<Pixel_Current> uniq_data(uniq_length);
    thrust::copy(d_pixel_current.rbegin(), d_uniq_pixel_current, uniq_data.begin());

    return uniq_data;
}

#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cstdlib>

extern "C" void Launch_ThrustTestSort(){ThrustTestSort();};
void ThrustTestSort(){
    std::cout << "testing sort..\n";
    int nHits = 3000000;
    int nSorts = 5000;

    // random keys
    thrust::device_vector<int> k2(nSorts);
    thrust::device_vector<int> k1(k2.size(), 0);
    thrust::sequence(k2.begin(), k2.end(), nHits/nSorts, nHits/nSorts);
    thrust::copy(k2.begin(), k2.end()-1, k1.begin()+1);

    // host vector verification
    thrust::host_vector<int> bla1 = k1;
    thrust::host_vector<int> bla2 = k2;
    int i=0;
    // for(auto c : bla1)
    //     std::cout << "i: " << ++i << " - " << c << "\n";
    // for(auto c : bla2)
    //     std::cout << "i2: " << ++i << " - " << c << "\n";

    auto tup_begin = thrust::make_zip_iterator(thrust::make_tuple(k1.begin(), k2.begin()));
    auto tup_end = thrust::make_zip_iterator(thrust::make_tuple(k1.end(), k2.end()));

    // values to sort on
    thrust::host_vector<int> h_v1(nHits);
    thrust::host_vector<int> h_v2(nHits);
    srand(13);
    thrust::generate(thrust::host, h_v1.begin(), h_v1.end(), rand);
    thrust::generate(thrust::host, h_v2.begin(), h_v2.end(), rand);

    i = 0;
    for(auto& c : h_v1){
        c = c%2 + 1;
    }
    for(auto& c : h_v2){
        c = c%10 + 1;
    }
    // std::cout << "presorted...*************************************\n";
    // for(int i=0; i<nHits; ++i){
    //     std::cout << "h1: " << h_v1[i] << " - " << h_v2[i] << "\n";
    // }
    // std::cout << "*************************************************\n";

    thrust::device_vector<int> d_v1 = h_v1;
    thrust::device_vector<int> d_v2 = h_v2;

    ZipIterator zip_itr = thrust::make_zip_iterator(thrust::make_tuple(d_v1.begin(), d_v2.begin()));
    arb_sort_functor f = {zip_itr};

    std::vector<s> bla(nHits);
    for(int i=0; i<bla.size(); ++i){
        bla[i].a= rand();
        bla[i].b= rand();
    }

    thrust::device_vector<s> d_bla = bla;

    auto start = std::chrono::high_resolution_clock::now();
    // thrust::for_each(tup_begin, tup_end, thrust::make_zip_function(f));
    thrust::sort(d_bla.begin(), d_bla.end(), compVec());
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "rand: " << nHits << " @ test sort time: " << duration.count() << "\n";
    h_v1 = d_v1;
    h_v2 = d_v2;

    // std::cout << "post-sorted...*************************************\n";
    // for(int i=0; i<nHits; ++i){
    //     std::cout << "i: " << i << " h1: " << h_v1[i] << " hv2 " << h_v2[i] << "\n";
    // }
    // std::cout << "*************************************************\n";

    start = std::chrono::high_resolution_clock::now();
    std::sort(bla.begin(), bla.end(), compVec());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "std::sort time comparison: " << duration.count() << "\n";
}