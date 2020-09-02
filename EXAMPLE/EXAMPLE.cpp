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

  // In and out files
  // std::string file_in = "MARLEY_100_Events.root";
  std::string file_in  = "/Users/austinmcdonald/projects/QPIX/Q_PIX_RTD/EXAMPLE/test_muon.root";
  std::string file_out = "/Users/austinmcdonald/projects/QPIX/Q_PIX_RTD/EXAMPLE/out_example.root";

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  number_entries = rfm.NumberEntries();
  rfm.EventReset();

  // Loop though the events in the file
  // for (int evt = 0; evt < number_entries; evt++)
  for (int evt = 1; evt < 2; evt++) 
  {
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

    std::cout << "Running the resets" << std::endl;
    // the reset function
    PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);


    // for (int i = 0; i < Pixel.size(); i++)
    // {
    //   std::cout << "SIZE " << i << "\t" << Pixel[i].RESET.size() << std::endl;
    // }


    // std::cout << "SIZE " << Pixel[20].RESET.size() << std::endl;

    for (int i = 0; i < Pixel[3].RESET.size(); i++)
    {
      std::cout << Pixel[3].RESET[i] << "\t" << Pixel[3].TSLR[i] << std::endl;

    }

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
