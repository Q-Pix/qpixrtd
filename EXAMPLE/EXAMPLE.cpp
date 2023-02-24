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

//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main(int argc, char** argv)
{

  clock_t time_req;
  time_req = clock();
  double time;

  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 777;
  Qpix::Random_Set_Seed(Seed);
  std::vector<double> Gaussian_Noise = Qpix::Make_Gaussian_Noise(0, (int) 1e7);

  // In and out files
  std::string file_in = argv[1];
  std::string file_out = argv[2];

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);
  Qpix_params->Buffer_time = 100e3;
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  rfm.AddMetadata(Qpix_params);  // add parameters to metadata
  number_entries = rfm.NumberEntries();
  rfm.EventReset();

  // we can make the pixel map once since we know everything about the detector
  // from the meta data
  std::map<int, Qpix::Pixel_Info> mPixelInfo = rfm.MakePixelInfoMap();

  // Loop though the events in the file
  for (int evt = 0; evt < number_entries; evt++)
  {
    std::cout << "*********************************************" << std::endl;
    std::cout << "Starting on event " << evt << std::endl;

    std::cout << "Getting the event" << std::endl;
    // turn the Geant4 hits into electrons
    std::vector<Qpix::ELECTRON> hit_e;
    rfm.Get_Event(evt, Qpix_params, hit_e);
    std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    // Pixelize the electrons 
    // std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    std::set<int> hit_pixels = PixFunc.Pixelize_Event(hit_e, mPixelInfo);
    // std::vector<Qpix::Pixel_Info> Pixel;
    // PixFunc.Pixelize_Event(hit_e, Pixel);

    std::cout << "Running the resets" << std::endl;
    // the reset function
    // PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);
    PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, hit_pixels, mPixelInfo);
    // PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);

    for(auto id : hit_pixels){
      if(mPixelInfo.find(id) == mPixelInfo.end())
        std::cout << "WARNING index " << id << ", not in hit map\n";
    }

    rfm.AddEvent(hit_pixels, mPixelInfo);
    // rfm.AddEvent(Pixel);
    rfm.EventFill();
    rfm.EventReset();
  }

  // save and close
  rfm.Save();




  std::cout << "done" << std::endl;
  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
