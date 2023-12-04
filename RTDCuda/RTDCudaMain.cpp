// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <map>

// for rand
#include <cstdlib>
#include <chrono>
#include <algorithm>

// Qpix includes
#include "Random.h"
#include "Structures.h"
#include "PixelResponse.h"

// RTDCuda includes
#include "RTDCudaFileManager.h"
#include "RTDCuda.h"
#include "RTDThrust.h"

// ROOT includes
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

// cuda includes
#include <cuda_runtime.h>

template <typename T> 
void printArray(T* a, const unsigned int& size)
{
    // print the input result
    std::cout << "int keys: { ";
    for(int i=0; i<size; ++i) std::cout << a[i] << " ";
    std::cout << "}\n";
}

/* use the input and output file to create a set of ions using CUDA */
void makeEvents(std::string& input_file, std::string& output_file)
{
    // begin the main parsing function to rip into the G4 Code
    // changing the seed for the random numbergenerator and generating the noise vector 
    constexpr std::uint64_t Seed = 41;
    Qpix::Random_Set_Seed(Seed);

    // Qpix paramaters 
    Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
    // Qpix_params->Buffer_time = 100e3;
    // neutrino events happen quickly
    Qpix_params->Buffer_time = 1;

    // root file manager
    Qpix::RTDCudaFileManager* rfm = new Qpix::RTDCudaFileManager(input_file, output_file);
    rfm->AddMetadata(Qpix_params);  // add parameters to metadata
    rfm->EventReset();

    // save the pixel current information tree
    TFile* tf = new TFile(output_file.c_str(), "RECREATE");
    TTree* tt = new TTree("tt", "ttree");
    std::vector<int> pix_x, pix_y;
    std::vector<double> reset_time;
    std::vector<int> trk_ids, nElec;
    tt->Branch("pix_x", &pix_x);
    tt->Branch("pix_y", &pix_y);
    tt->Branch("trk_id", &trk_ids);
    tt->Branch("nElec", &nElec);

    // cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL/64);

    // std::vector<Qpix::ION>* b_ions = 0;
    // tt->Branch("ions", &b_ions);
    // Loop though the hit events in the geant file

    std::cout << "CUDA RTD begin with entries: " << rfm->NumberEntries() << std::endl;
    int curEvent = 0;
    std::vector<Qpix::ION> event_ions;
    while (rfm->GetCurrentEntry() < rfm->NumberEntries())
    {
        // create true electron image  2P3T * E -> N_e * P3T

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Pixel_Current> event_current = rfm->Get_Event(curEvent++, event_ions);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "evt: " << curEvent << " @ make time: " << duration.count() << "\n";
        // assign tree values from the pixel currents here
        // for(const auto& pc : event_current){
        //     pix_x.push_back(pc.X_Pix);
        //     pix_y.push_back(pc.Y_Pix);
        //     trk_ids.push_back(pc.Trk_ID);
        //     nElec.push_back(pc.nElec);
        // }

        // std::vector<int> pix_ids;
        // std::vector<double> pix_resets;
        // std::vector<std::vector<int>> pix_trk_ids;
        // Launch_Make_QResets(event_current, pix_ids.data(), pix_resets.data(), pix_trk_ids);

        // save the relevant entries into the relevant trees
        // b_ions = &event_ions;
        // tt->Fill();

        // emtpy the vectors after fill
        pix_x = {};
        pix_y = {};
        trk_ids = {};
        nElec = {};
    }

    tf->Write();
    tf->Close();
    // std::cout << "read a total of " << rfm.GetCurrentEntry() << ", entries.\n";
    // std::cout << "created a total of " << nElectrons << std::endl;

    // save and close generated rtd file
    // rfm.Save();

    delete rfm;
}


struct compPix
{
    bool operator()(const pix& lhs, const pix& rhs)
    {return lhs.a > rhs.a;};
};

const int N_keys = 1e6;
void testThrustSortStruct(){
    std::vector<pix> v_pix;
    v_pix.reserve(N_keys);

    // std sorting
    // fill
    srand(1);
    pix first_pivot;
    for(int i=0; i<N_keys; ++i){
        first_pivot.a = (((double)rand() / (RAND_MAX)) * N_keys);
        first_pivot.b = 42;
        first_pivot.c = 42;
        v_pix.push_back(first_pivot);
    }

    // thrust sorting struct
    auto v_pix_copy = v_pix;
    auto start = std::chrono::high_resolution_clock::now();
    Launch_ThrustSortStruct(v_pix_copy.data(), N_keys);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "thrust sort struct alloc and call time: " << duration.count() << "\n";

    // std sorting struct
    start = std::chrono::high_resolution_clock::now();
    std::sort(v_pix.begin(), v_pix.end(), compPix());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "std sort struct call time: " << duration.count() << "\n";
}

void testThrustSort(){
    // prototype the sorting function
    std::vector<unsigned int> h_keys;
    h_keys.reserve(N_keys);

    // fill
    srand(1);
    unsigned int first_pivot;
    std::vector<unsigned int> vals;
    for(int i=0; i<N_keys; ++i){
        first_pivot = (((double)rand() / (RAND_MAX)) * N_keys);
        vals.push_back(first_pivot);
    }

    // sorting kernel call here
    auto start = std::chrono::high_resolution_clock::now();
    Launch_ThrustSort(vals.data(), vals.size());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "launch thrust alloc and call time: " << duration.count() << "\n";

    vals.clear();
    for(int i=0; i<N_keys; ++i){
        first_pivot = (((double)rand() / (RAND_MAX)) * N_keys);
        vals.push_back(first_pivot);
    }
    start = std::chrono::high_resolution_clock::now();
    std::sort(vals.begin(), vals.end());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "std sort call time: " << duration.count() << "\n";

}

int main(int argc, char** argv) {

    int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) {
        deviceCount = 0;
        std::cout << "received device err code: " << cudaResultCode << std::endl;
    }

    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) /* 9999 means emulation only */
            ++gpuDeviceCount;
    }
    printf("%d GPU CUDA device(s) found\n", gpuDeviceCount);

    // run the CUDA core on the input file and make the output ROOT file
    // In and out files
    std::string file_in = argv[1];
    std::string file_out = argv[2];
    clock_t time_req;
    time_req = clock();
    double time;
    makeEvents(file_in, file_out);

    // testThrustSort();
    // testThrustSortStruct();

    std::cout << "done" << std::endl;
    time_req = clock() - time_req;
    time = (float)time_req/CLOCKS_PER_SEC;
    std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;

    return 0;
}