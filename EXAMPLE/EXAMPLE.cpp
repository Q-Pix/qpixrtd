// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <map>

// Qpix includes
#include "Random.h"
#include "ROOTFileManager.h"
#include "ROOT/RDataFrame.hxx"
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

void getSize(std::unordered_map<int, Qpix::Pixel_Info>& pixel_umap)
{
  int reset_size = 0;
  int tslr_size = 0;
  int trkid_size = 0;
  int resetI_size = 0;
  int resetW_size = 0;
  int time_size = 0;
  float mpid_size = 0;
  float nCounts = 0;
  float charge = 0;
  for(auto int_pix : pixel_umap){
    auto pixel = int_pix.second;
    if(pixel.time.size() > 0){
      time_size += pixel.time.size();
      trkid_size += pixel.Trk_ID.size();
      // mpid_size += pixel.mPids.size();
      charge += pixel.charge;
      ++nCounts;
    }
    if(nCounts == 0) nCounts = 1;
    // zero
    // reset_size += pixel.RESET.size();
    // tslr_size += pixel.TSLR.size();
    // resetI_size += pixel.RESET_TRUTH_ID.size();
    // resetW_size += pixel.RESET_TRUTH_W.size();
  }
  std::cout << "map size:\ntime: " << time_size  << ", avg: " << time_size / nCounts
            << "\ntrkid_size: " << trkid_size << ", avg: " << trkid_size / nCounts
            << "\nmap size: " << mpid_size << ", avg: " << mpid_size / nCounts
            << "\ncharge: " << charge << ", avg: " << charge / nCounts
            << "\ncounts: " << nCounts 
            // << "\nRESET: " << reset_size
            // << "\ntslr_size: " << tslr_size
            // << "\nresetI_size: " << resetI_size
            // << "\nresetW_size: " << resetW_size
            << std::endl;
}

//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 41;
  Qpix::Random_Set_Seed(Seed);
  // std::vector<double> Gaussian_Noise = Qpix::Make_Gaussian_Noise(0, (int) 1e7);

  // In and out files
  std::string file_in = argv[1];
  std::string file_out = argv[2];

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  // Qpix_params->Buffer_time = 100e3;
  // neutrino events happen quickly
  Qpix_params->Buffer_time = 1;
  // print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  rfm.AddMetadata(Qpix_params);  // add parameters to metadata
  number_entries = rfm.NumberEntries();
  rfm.EventReset();

  // we can make the pixel map once since we know everything about the detector
  // from the meta data
  std::unordered_map<int, Qpix::Pixel_Info> mPixelInfo = rfm.MakePixelInfoMap(); // ~870k pixels

  clock_t time_req;
  time_req = clock();
  double time;

  std::vector<clock_t> start_pixelize_times, start_reset_times;
  std::vector<clock_t> stop_pixelize_times, stop_reset_times;
  
  // Loop though the events in the file
  std::cout << "RTD begin with entries: " << number_entries << std::endl;
  for (int evt = 0; evt < number_entries; evt++)
  {
    // std::cout << "*********************************************" << std::endl;
    // std::cout << "Starting on event " << evt << std::endl;

    if(evt%100000 == 0){
      std::cout << "Getting the event: " << evt << std::endl;
      // avg(start_pixelize_times, stop_pixelize_times, "pixel times average: ");
      // avg(start_reset_times, stop_reset_times, "reset times average: ");
      // start_reset_times.clear();
      // stop_reset_times.clear();
      // start_pixelize_times.clear();
      // stop_pixelize_times.clear();
      // getSize(mPixelInfo);
    }
    // turn the Geant4 hits into electrons
    std::vector<Qpix::ELECTRON> hit_e;
    // rfm.Get_Event(evt, Qpix_params, hit_e, true); // sort by index
    rfm.Get_Event(evt, Qpix_params, hit_e, false); // sort by time, not index
    // std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    // Pixelize the electrons 
    // std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    // std::vector<Qpix::Pixel_Info> Pixel;

    // start_pixelize_times.push_back(clock());
    // PixFunc.Pixelize_Event(hit_e, Pixel);
    std::set<int> hit_pixels = PixFunc.Pixelize_Event(hit_e, mPixelInfo);
    // stop_pixelize_times.push_back(clock());

    // std::cout << "Running the resets" << std::endl;
    // the reset function
    // start_reset_times.push_back(clock());
    // PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);
    PixFunc.Reset_Fast(Qpix_params, hit_pixels, mPixelInfo);
    // stop_reset_times.push_back(clock());

    // rfm.AddEvent(Pixel);
    rfm.AddEvent(hit_pixels, mPixelInfo);

    rfm.EventFill();
    rfm.EventReset();
  }

  // save and close generated rtd file
  rfm.Save();

  // find the maximum entry for this bin
  // auto rdf = ROOT::RDataFrame("event_tree", file_out.c_str());
  // auto h2 = rdf.Histo2D({"h", "h", 575, 0.5, 575.5, 1500, 0.5, 1500.5}, "pixel_x", "pixel_y");
  // int pix_x_width = 8;
  // int pix_y_width = 8;
  // int pix_x_max;
  // int pix_y_max;
  // int z;

  // h2->Rebin2D(pix_x_width, pix_y_width);
  // int maxBin = h2->GetMaximumBin();
  // int size = h2->GetBinContent(maxBin);
  // h2->GetBinXYZ(maxBin, pix_x_max, pix_y_max, z);
  // std::cout << "Max Hits:" << size << "\n";
  // std::cout << "found maximum pixel_x: " << pix_x_max*pix_x_width << ", maximum pixel_y: " << pix_y_max*pix_y_width << std::endl;

  std::cout << "done" << std::endl;
  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
