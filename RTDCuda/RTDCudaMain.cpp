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

    // we can make the pixel map once since we know everything about the detector
    // from the meta data
    std::unordered_map<int, Qpix::Pixel_Info> mPixelInfo = rfm->MakePixelInfoMap(); // ~870k pixels

    // TFile* tf = new TFile("test_cuda.root", "RECREATE");
    // TTree* tt = new TTree("tt", "ttree");
    // std::vector<Qpix::ION>* b_ions = 0;
    // tt->Branch("ions", &b_ions);
    // Loop though the hit events in the geant file

    std::cout << "CUDA RTD begin with entries: " << rfm->NumberEntries() << std::endl;
    // Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    int curEvent = 0;
    while (rfm->GetCurrentEntry() < rfm->NumberEntries())
    {
        // create true electron image  2P3T * E -> N_e * P3T
        std::vector<Qpix::ION> event_ions = rfm->Get_Event(curEvent++);
        // b_ions = &event_ions;
        // tt->Fill();
    }

    // tf->Write();
    // tf->Close();
    // std::cout << "read a total of " << rfm.GetCurrentEntry() << ", entries.\n";
    // std::cout << "created a total of " << nElectrons << std::endl;

    // save and close generated rtd file
    // rfm.Save();

    delete rfm;
}

void testThrustSort(){
    // prototype the sorting function
    int N_keys = 1e7;
    std::vector<unsigned int> h_keys;
    h_keys.reserve(N_keys);

    // srand(1);
    // unsigned int first_pivot;
    // for(int i=0; i<N_keys; ++i){
    //     first_pivot = (((double)rand() / (RAND_MAX)) * N_keys);
    //     h_keys.push_back(first_pivot);
    // }

    // sorting kernel call here
    srand(1);
    unsigned int first_pivot;
    std::vector<unsigned int> vals;
    for(int i=0; i<N_keys; ++i){
        first_pivot = (((double)rand() / (RAND_MAX)) * N_keys);
        vals.push_back(first_pivot);
    }

    // print the input result
    // printArray<unsigned int>(vals.data(), vals.size());

    auto start = std::chrono::high_resolution_clock::now();
    Launch_ThrustSort(vals.data(), vals.size());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "launch thrust alloc and call time: " << duration.count() << "\n";
    // std::cout << "input arr: {";
    // for(int i=0; i<5; ++i)std::cout << vals[i] << " ";
    // std::cout << "}\n";

    vals.clear();
    for(int i=0; i<N_keys; ++i){
        first_pivot = (((double)rand() / (RAND_MAX)) * N_keys);
        vals.push_back(first_pivot);
    }
    start = std::chrono::high_resolution_clock::now();
    std::sort(vals.begin(), vals.end());
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "quicksort call time: " << duration.count() << "\n";
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
    makeEvents(file_in, file_out);

    return 0;
}