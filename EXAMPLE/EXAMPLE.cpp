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

void avg(std::vector<clock_t> start_times, std::vector<clock_t> stop_times, std::string msg){
  int size = start_times.size();
  double avgs;
  for(int i=0; i<start_times.size(); ++i){
    clock_t time_req = stop_times[i] - start_times[i];
    avgs += (float)time_req/CLOCKS_PER_SEC;
  }

  std::cout << msg << avgs/size << std::endl;
}


//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 41;
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
  std::unordered_map<int, Qpix::Pixel_Info> mPixelInfo = rfm.MakePixelInfoMap();

  clock_t time_req;
  time_req = clock();
  double time;

  std::vector<clock_t> start_pixelize_times, start_reset_times;
  std::vector<clock_t> stop_pixelize_times, stop_reset_times;

  // Loop though the events in the file
  for (int evt = 0; evt < number_entries; evt++)
  {
    // std::cout << "*********************************************" << std::endl;
    // std::cout << "Starting on event " << evt << std::endl;

    if(evt%1000 == 0){
      std::cout << "Getting the event: " << evt << std::endl;
      avg(start_pixelize_times, stop_pixelize_times, "pixel times average: ");
      avg(start_reset_times, stop_reset_times, "reset times average: ");
      start_reset_times.clear();
      stop_reset_times.clear();
      start_pixelize_times.clear();
      stop_pixelize_times.clear();
    }
    // turn the Geant4 hits into electrons
    std::vector<Qpix::ELECTRON> hit_e;
    // rfm.Get_Event(evt, Qpix_params, hit_e, true); // sort by index
    rfm.Get_Event(evt, Qpix_params, hit_e, false); // sort by time, not index
    // std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    // Pixelize the electrons 
    // std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    std::vector<Qpix::Pixel_Info> Pixel;

    start_pixelize_times.push_back(clock());
    // PixFunc.Pixelize_Event(hit_e, Pixel);
    std::set<int> hit_pixels = PixFunc.Pixelize_Event(hit_e, mPixelInfo);
    stop_pixelize_times.push_back(clock());

    // for(int hid : hit_pixels)
    //   Pixel.push_back(mPixelInfo[hid]);

    // std::cout << "Running the resets" << std::endl;
    // the reset function
    start_reset_times.push_back(clock());
    // PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);
    PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, hit_pixels, mPixelInfo);
    stop_reset_times.push_back(clock());

    // rfm.AddEvent(Pixel);
    rfm.AddEvent(hit_pixels, mPixelInfo);

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
