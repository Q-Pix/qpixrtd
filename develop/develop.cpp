// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>


// Qpix includes
#include "Qpix/Random.h"
#include "Qpix/ROOTFileManager.h"
#include "Qpix/Structures.h"
#include "Qpix/PixelResponse.h"


//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main()
{

  clock_t time_req;
  time_req = clock();
  double time;

  // changing the seed for the random numbergenerator and generating the noise vector 
  constexpr std::uint64_t Seed = 777;
  Qpix::Random_Set_Seed(Seed);
  std::vector<double> Gaussian_Noise = Qpix::Make_Gaussian_Noise(2, (int) 1e7);
  // std::vector<double> Gaussian_Noise(1000, 0.0);

  // In and out files
  std::string file_in = "/Users/austinmcdonald/projects/QPIX/Nu_e-78.root";
  // std::string file_in = "/Users/austinmcdonald/projects/QPIX/Ar39-999.root";
  std::string file_out = "../out_example.root";

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);
  long long num1 = 1e10;
  Qpix_params->Buffer_time = num1;
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  number_entries = rfm.NumberEntries();
  rfm.EventReset();


  int ct=0;
  // Loop though the events in the file
  for (int evt = 0; evt < 100; evt++)
  {
    std::cout << "*********************************************" << std::endl;
    std::cout << "Starting on event " << evt << std::endl;

    std::cout << "Getting the event" << std::endl;
    std::vector<Qpix::ELECTRON> hit_e;
    // turn the Geant4 hits into electrons
    rfm.Get_Event( evt, Qpix_params, hit_e);
    std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    if (hit_e.size() < 100){std::cout << "[WARNING] not enough hits" << std::endl; continue;}

    std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    std::vector<Qpix::Pixel_Info> Pixel;
    // Pixelize the electrons 
    PixFunc.Pixelize_Event( hit_e, Pixel );

    std::cout << "Running the resets" << std::endl;
    // the reset function
    // PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);
    PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);

    // int ct=0;
    for (int pixx = 0; pixx < Pixel.size(); pixx++)
    {
      ct += Pixel[pixx].RESET.size();
    } 
    std::cout << "Number of resets = " << ct << std::endl;

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
