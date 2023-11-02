// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include <string>     // std::string, std::to_string


// Qpix includes
#include "Random.h"
#include "ROOTFileManager.h"
#include "Structures.h"
#include "PixelResponse.h"

#include "Electronics.h"


// class Snip
// {
// private:
//   double const ElectronCharge_ = 1.60217662e-19;


// public:

//   void Snipped_RTD( std::vector<Qpix::Pixel_Info> Pixel,  std::string File_Name)
//   {

//     int Hot_index = 0;
//     int Hot_size = 0;
//     for (int pixx = 0; pixx < Pixel.size(); pixx++)
//     {
//       if (Pixel[pixx].time.size() > Hot_size)
//       {
//         Hot_size = Pixel[pixx].time.size();
//         Hot_index = pixx;
//       }
//     }
//     std::cout << "Hot_index = " << Hot_index << std::endl;
//     std::cout << "Hot Pixel info = " << Pixel[Hot_index].X_Pix << " , "<< Pixel[Hot_index].Y_Pix << std::endl;

//     std::cout << "X Pix   " << "Y Pix" << std::endl;

//     std::ofstream Reset_File;
//     Reset_File.open(File_Name);

//     int X_dex = Pixel[Hot_index].X_Pix - 5;
//     int Y_dex = Pixel[Hot_index].Y_Pix - 5;

//     for (int row = 0; row < 11; row++)
//     {
//       for (int col = 0; col < 11; col++)
//       {
//         for (int pixx = 0; pixx < Pixel.size(); pixx++)
//         {
//           if ( Y_dex == Pixel[pixx].Y_Pix && X_dex == Pixel[pixx].X_Pix )
//           {
//             std::cout << "Hot Pixel info = " << Pixel[pixx].X_Pix << " , "<< Pixel[pixx].Y_Pix << std::endl;
//             std::cout << "Size  = " << Pixel[pixx].RESET.size() << std::endl;
//             for (int T = 0; T < Pixel[pixx].RESET.size(); T++)
//             {
//               Reset_File << X_dex << "," << Y_dex << "," << Pixel[pixx].RESET[T] << "\n";
//             }

//           }
//         }
//         X_dex += 1;

//       }
//       Y_dex += 1;
//       X_dex = Pixel[Hot_index].X_Pix - 5;

//     }

//     Reset_File.close();
//     Reset_File.clear();

//   }

// };




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
  // std::string file_in = "../mitch_10cm.root";
  // std::string file_in = "../mitch_100cm.root";
  std::string file_in = "../mitch_proton_10cm.root";
  // std::string file_in = "/Users/austinmcdonald/projects/QPIX/Nu_e-78.root";
  // std::string file_in = "../../../muon.root";
  std::string file_out = "../out_example.root";

  // Qpix paramaters 
  Qpix::Qpix_Paramaters * Qpix_params = new Qpix::Qpix_Paramaters();
  // Qpix_params->Buffer_time = 1e10;
  print_Qpix_Paramaters(Qpix_params);

  // root file manager
  int number_entries = -1;
  Qpix::ROOTFileManager rfm = Qpix::ROOTFileManager(file_in, file_out);
  number_entries = rfm.NumberEntries();
  rfm.EventReset();

  // Loop though the events in the file
  // int evt = 24;
  // std::cout << "*********************************************" << std::endl;
  // std::cout << "Starting on event " << evt << std::endl;

  // std::cout << "Getting the event" << std::endl;
  // std::vector<Qpix::ELECTRON> hit_e;
  // // turn the Geant4 hits into electrons
  // rfm.Get_Event( evt, Qpix_params, hit_e);
  // std::cout << "size of hit_e = " << hit_e.size() << std::endl;

  // std::cout << "Pixelizing the event" << std::endl;
  // Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
  // std::vector<Qpix::Pixel_Info> Pixel;
  // // Pixelize the electrons 
  // PixFunc.Pixelize_Event( hit_e, Pixel );

  // std::cout << "Running the resets" << std::endl;
  // // the reset function
  // PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);

  // // std::string current_f = "current_electron_3MeV.csv";
  // // std::string reset_f   = "reset_electron_3MeV.csv";
  // // Qpix::Current_Profile Current = Qpix::Current_Profile();
  // // Current.Get_Hot_Current( Qpix_params, Pixel, Gaussian_Noise, current_f, reset_f);

  // std::string ASIC_file   = "Muon_0.txt";
  // Snip ASIC_snip = Snip();
  // // ASIC_snip.Snipped_RTD( Pixel, ASIC_file);


  // rfm.AddEvent( Pixel );
  // rfm.EventFill();
  // rfm.EventReset();


  // Loop though the events in the file
  for (int evt = 0; evt < 25; evt++)
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
    PixFunc.Reset(Qpix_params, Gaussian_Noise, Pixel);

    std::string ASIC_file   = "../Out_files/Protons/Proton_"+std::to_string(evt)+".txt";
    
    Qpix::Snip ASIC_snip = Qpix::Snip();
    ASIC_snip.Snipped_RTD( Pixel, ASIC_file);


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
