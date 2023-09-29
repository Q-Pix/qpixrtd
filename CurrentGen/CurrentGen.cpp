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

#include <fstream>

// #define pix_x_start 408
// #define pix_y_start 824
#define pix_x_width 8
#define pix_y_width 8

// save the text file that we want
void saveTxtFile(std::set<int> hit_pixels, std::unordered_map<int, Qpix::Pixel_Info> hit_map){

  // find the largest stopping time for each of the pixel
  double cur_time=0;
  double timeStep = 1e-9;
  double stop_time=1e-3;

  // create the file and the file header
  std::ofstream output_txt_file;
  output_txt_file.open("test_cur_gen.txt");
  output_txt_file << "time(ns)";
  output_txt_file << "\t";
  for(auto pix : hit_pixels) {
    std::string pix_name = "(" + std::to_string(hit_map[pix].X_Pix) + "," + std::to_string(hit_map[pix].Y_Pix) + ")";
    output_txt_file << pix_name + "\t";
  }
  output_txt_file << "\n";

  // keep track of index within each of the hit_maps;
  std::vector<int> nelectrons(hit_pixels.size());

  // increment time
  while(cur_time < stop_time){
    std::string input = std::to_string((int)(cur_time*1e9)) + "\t";

    // count the electrons in each pixel at the current time
    int nPix = 0;
    for(auto pix : hit_pixels){

      int count = 0;
      while(nelectrons[nPix] < hit_map[pix].time.size() && hit_map[pix].time[nelectrons[nPix]] < cur_time){
        ++nelectrons[nPix];
        count += 1;
      }
      // running sum
      // int pixel_nelectrons = nelectrons[nPix];
      // count for this
      int pixel_nelectrons = count;
      input = input + std::to_string(pixel_nelectrons);
      // go to next pixel
      nPix++;
      if(nPix < hit_pixels.size())
        input = input + "\t";
    }

    cur_time += timeStep;
    input = input + "\n";
    output_txt_file << input;
  }
  int totElectrons(0);
  for(auto nelec : nelectrons)
    totElectrons += nelec;
  std::cout << "found total electrons: " << totElectrons << ", for " << totElectrons/6250 << " total resets.\n";
}

int main(int argc, char** argv)
{
  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 41;
  Qpix::Random_Set_Seed(Seed);

  if(argc != 4){
    std::cout << "must provide input file, pixel_x, and pixel_y for testing\n";
    return -1;
  }

  // In and out files
  std::string file_in = argv[1];
  int pix_x_start = std::stoi(argv[2]);
  int pix_y_start = std::stoi(argv[3]);
  std::cout << "looking for pixels beginning at (" << pix_x_start << "," << pix_y_start << ")" << std::endl;
  std::string file_out = "cur_gen_poop.root";

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);
  Qpix_params->Buffer_time = 100e3; // useful for radiogenic data
  // neutrino events happen quickly
  // Qpix_params->Buffer_time = 1;
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

  // build the good_pixel set that we want
  std::set<int> good_pixels;
  for(int good_pix_x=pix_x_start; good_pix_x<pix_x_start+pix_x_width; good_pix_x++)
    for(int good_pix_y=pix_y_start; good_pix_y<pix_y_start+pix_y_width; good_pix_y++)
      good_pixels.insert(Qpix::ID_Encoder(good_pix_x, good_pix_y));


  // Pixelize the electrons 
  Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();

  // Loop though the events in the file
  std::cout << "CurrentGen begin with entries: " << number_entries << std::endl;
  for (int evt = 0; evt < number_entries; evt++)
  {
    if(evt%100000 == 0){
      std::cout << "Getting the event: " << evt << std::endl;
    }

    std::vector<Qpix::ELECTRON> hit_e;
    rfm.Get_Event(evt, Qpix_params, hit_e, false); // sort by time, not index

    // remove the pixels that aren't in the set that we want to digitize
    std::set<int> hit_pixels = PixFunc.Pixelize_Event(hit_e, mPixelInfo, good_pixels);
  }

  std::cout << "generating current profile..\n";
  saveTxtFile(good_pixels, mPixelInfo);

  return 0;

} // main()
