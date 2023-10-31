// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <map>

// Qpix includes
#include "Random.h"
#include "ROOTFileManager.h"
#include "Structures.h"
#include "PixelResponse.h"

// ROOT includes
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

// cuda includes
#include <cuda_runtime.h>

// Prototype for the CUDA function
extern "C" void launch_add_arrays(int* a, int* b, int* c, int size);

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
    int size = 5;

    // Initialize input arrays a and b
    std::vector<int> a(size);
    std::vector<int> b(size);
    
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Initialize the output array c
    std::vector<int> c(size);

    // Call the CUDA function
    launch_add_arrays(a.data(), b.data(), c.data(), size);

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
    int number_entries = -1;
    Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
    rfm.AddMetadata(Qpix_params);  // add parameters to metadata
    number_entries = rfm.NumberEntries();
    rfm.EventReset();

    // we can make the pixel map once since we know everything about the detector
    // from the meta data
    std::unordered_map<int, Qpix::Pixel_Info> mPixelInfo = rfm.MakePixelInfoMap(); // ~870k pixels

    // Loop though the hit events in the geant file
    std::cout << "CUDA RTD begin with entries: " << number_entries << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    long unsigned int nElectrons = 0;
    for (int evt = 0; evt < number_entries; evt++)
    {
        // create true electron image  2P3T * E -> N_e * P3T
        std::vector<Qpix::ELECTRON> hit_e;
        rfm.Get_Event(evt, Qpix_params, hit_e, false); // sort by time, not index

        nElectrons += hit_e.size();

        // Pixelize the electrons P3T -> P2T
        std::set<int> hit_pixels = PixFunc.Pixelize_Event(hit_e, mPixelInfo);

        // the reset function -> P2T
        PixFunc.Reset_Fast(Qpix_params, hit_pixels, mPixelInfo);

        // fill output to tree
        rfm.AddEvent(hit_pixels, mPixelInfo);
        rfm.EventFill();
        rfm.EventReset();
    }
    std::cout << "created a total of " << nElectrons << std::endl;

    // save and close generated rtd file
    rfm.Save();

    return 0;
}