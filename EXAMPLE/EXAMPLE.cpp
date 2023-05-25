// C++ includes
#include <iostream>
// #include <numeric>
#include <string>
#include <vector>
#include <ctime>

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
  Qpix_params->Buffer_time = 0.01;
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  rfm.AddMetadata(Qpix_params);  // add parameters to metadata
  number_entries = rfm.NumberEntries();
  rfm.EventReset();

  // std::vector< int > v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // auto start = v.begin() + 1;
  // auto stop = start + 2;
  // // auto stop = v.begin() + 3;
  // int sum = accumulate(start, stop, 0);
  // std::cout << "sum: " << sum << std::endl;

  // Loop though the events in the file
  for (int evt = 0; evt < number_entries; evt++)
  {
    // break;
    std::cout << "*********************************************" << std::endl;
    std::cout << "Starting on event " << evt << std::endl;

    std::cout << "Getting the event" << std::endl;
    std::vector<Qpix::ELECTRON> hit_e;
    // turn the Geant4 hits into electrons
    rfm.Get_Event( evt, Qpix_params, hit_e);
    std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    std::vector<Qpix::Pixel_Info> Pixel;
    // Pixelize the electrons 
    PixFunc.Pixelize_Event( hit_e, Pixel );

    std::cout << "hit_e.size(): " << hit_e.size() << std::endl;

    std::cout << "Running the resets" << std::endl;
    // the reset function
    PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);
    // PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);
    // PixFunc.reset_th1_test(Qpix_params, Gaussian_Noise, Pixel);
    // PixFunc.reset_thnsparse_test(Qpix_params, Gaussian_Noise, Pixel);
    // PixFunc.reset_thnsparse_noise_test(Qpix_params, Gaussian_Noise, Pixel);
    // PixFunc.Reset_THnSparse(Qpix_params, Gaussian_Noise, Pixel);

    rfm.AddEvent( Pixel );
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
