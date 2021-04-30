// C++ includes
#include <iostream>
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
  // std::vector<double> Gaussian_Noise(1000, 0.0);

  // In and out files
  std::string file_in  = argv[1];
  std::string file_out = argv[2];

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  set_Qpix_Paramaters(Qpix_params);
  // long long num1 = 1e10;
  // long long num1 = 1e9;
  // Qpix_params->Buffer_time = num1;
  Qpix_params->Reset = 650;
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  rfm.AddMetadata(Qpix_params);  // add parameters to metadata
  number_entries = rfm.NumberEntries();
  rfm.EventReset();


  int ct=0;
  // Loop though the events in the file
  // Event: 4
	//  max hit 9.674661926631305
	//  mean z 333.1369690397591
  for (int evt = 4; evt < 5; evt++)
  {
    std::cout << "*********************************************" << std::endl;
    std::cout << "Starting on event " << evt << std::endl;

    std::cout << "Getting the event" << std::endl;
    std::vector<Qpix::ELECTRON> hit_e;
    // turn the Geant4 hits into electrons
    rfm.Get_Event( evt, Qpix_params, hit_e);
    std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    if (hit_e.size() < 100){std::cout << "[WARNING] not enough hits" << std::endl;}

    std::cout << "Pixelizing the event" << std::endl;
    Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
    std::vector<Qpix::Pixel_Info> Pixel;
    // Pixelize the electrons 
    PixFunc.Pixelize_Event( hit_e, Pixel );

    std::cout << "Running the resets" << std::endl;
    // the reset function
    PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);
    // PixFunc.Reset_Fast(Qpix_params, Gaussian_Noise, Pixel);

    // int ct=0;
    for (int pixx = 0; pixx < Pixel.size(); pixx++)
    {
      ct += Pixel[pixx].RESET.size();
    } 
    std::cout << "Number of resets = " << ct << std::endl;

    for (int pixx = 0; pixx < Pixel.size(); pixx++)
    {
      std::cout 
      << "X " << Pixel[pixx].X_Pix << '\t'
      << "Y " << Pixel[pixx].Y_Pix << std::endl;
      for (int pixx2 = 0; pixx2 < Pixel[pixx].RESET.size(); pixx2++)
      {
        std::cout 
        << "resets = " << Pixel[pixx].RESET[pixx2] << std::endl;
        // std::cout << Pixel[pixx].RESET_TRUTH_ID.size() << std::endl;
        for (int pixx3 = 0; pixx3 < Pixel[pixx].RESET_TRUTH_ID[pixx2].size(); pixx3++)
        {
          std::cout 
          << Pixel[pixx].RESET_TRUTH_ID[pixx2][pixx3] << '\t'
          << Pixel[pixx].RESET_TRUTH_W[pixx2][pixx3] << std::endl;
        }
      }
      std::cout << "end" << std::endl;
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
