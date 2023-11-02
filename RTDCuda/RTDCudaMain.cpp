// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <map>

// Qpix includes
#include "Random.h"
#include "Structures.h"
#include "PixelResponse.h"

// RTDCuda includes
#include "RTDCudaFileManager.h"

// ROOT includes
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

// cuda includes
#include <cuda_runtime.h>

// Prototype for the CUDA function
// extern "C" void launch_add_arrays(int* a, int* b, int* c, int size);

#include "RTDCuda.h"
// extern "C" void launch_add_diff_arrays(int* a, int* b, int* c, int size);

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

    // attempt the diff array size add
    int points = 500;
    std::vector<double> h_point, h_step;
    std::vector<int> h_nsum;

    int run_sum = 100;
    for(int i=0; i<points; ++i){
        h_point.push_back(i+1);
        h_step.push_back(0.0001);
        run_sum += 100;
        h_nsum.push_back(run_sum);
    }

    std::vector<double> h_elec(h_nsum.back());
    std::cout << "hstep / size: " << points << ", " << h_elec.size() << "\n";

    // launch_add_diff_arrays(h_point.data(), h_step.data(), h_elec.data(), h_nsum.data(), h_elec.size(), h_nsum.size());
    // return 0;

    // begin the main parsing function to rip into the G4 Code
    // changing the seed for the random numbergenerator and generating the noise vector 
    constexpr std::uint64_t Seed = 41;
    Qpix::Random_Set_Seed(Seed);
    // std::vector<double> Gaussian_Noise = Qpix::Make_Gaussian_Noise(0, (int) 1e7);

    // In and out files
    std::string file_in = argv[1];
    std::string file_out = argv[2];

    // Qpix paramaters 
    Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
    set_Qpix_Paramaters(Qpix_params);
    // Qpix_params->Buffer_time = 100e3;
    // neutrino events happen quickly
    Qpix_params->Buffer_time = 1;

    // root file manager
    Qpix::RTDCudaFileManager rfm = Qpix::RTDCudaFileManager(file_in, file_out);
    rfm.AddMetadata(Qpix_params);  // add parameters to metadata
    rfm.SetQPixParams(*Qpix_params);
    rfm.EventReset();

    // we can make the pixel map once since we know everything about the detector
    // from the meta data
    std::unordered_map<int, Qpix::Pixel_Info> mPixelInfo = rfm.MakePixelInfoMap(); // ~870k pixels

    // Loop though the hit events in the geant file
    std::cout << "CUDA RTD begin with entries: " << rfm.NumberEntries() << std::endl;
    // Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    long unsigned int nElectrons = 0;
    int curEvent = 0;
    while (rfm.GetCurrentEntry() < rfm.NumberEntries())
    {
        // create true electron image  2P3T * E -> N_e * P3T
        std::vector<Qpix::ION> event_ions = rfm.Get_Event(curEvent++);
        nElectrons += event_ions.size();

        // Pixelize the electrons P3T -> P2T
        // std::set<int> hit_pixels = PixFunc.Pixelize_Event(hit_e, mPixelInfo);

        // the reset function -> P2T
        // PixFunc.Reset_Fast(Qpix_params, hit_pixels, mPixelInfo);

        // // fill output to tree
        // rfm.AddEvent(hit_pixels, mPixelInfo);
        // rfm.EventFill();
        // rfm.EventReset();

    }
    // std::cout << "read a total of " << rfm.GetCurrentEntry() << ", entries.\n";
    // std::cout << "created a total of " << nElectrons << std::endl;

    // save and close generated rtd file
    // rfm.Save();

    return 0;
}