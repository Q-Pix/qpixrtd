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


class Current
{
private:
  double const ElectronCharge_ = 1.60217662e-19;


public:

  void Find_Hot_Pixel( std::vector<Qpix::Pixel_Info> Pixel, std::vector<double> Gaussian_Noise, std::string Current_F, std::string Reset_F)
  {

    int Hot_index = 0;
    int Hot_size = 0;
    for (int pixx = 0; pixx < Pixel.size(); pixx++)
    {
      if (Pixel[pixx].time.size() > Hot_size)
      {
        Hot_size = Pixel[pixx].time.size();
        Hot_index = pixx;
      }
    }

    int charge = 0;
    int pix_size = Pixel[Hot_index].time.size();
    int pix_dex = 0;
    int current_time = 0;
    int Noise_index = 0;
    int Noise_Vector_Size = Gaussian_Noise.size();

    // int OFFSET = Pixel[Hot_index].time[0] - 100;
    int OFFSET = 0;

    int pix_time = Pixel[Hot_index].time[pix_dex] - OFFSET;

    // int End_Time = Pixel[Hot_index].time[pix_size -1] - OFFSET;
    int End_Time = 20000;

    std::ofstream Current_File;
    Current_File.open(Current_F);

    // for each pixel loop through the buffer time
    while (current_time <= End_Time)
    {
      // setting the "time"
      // current_time += Qpix_params->Sample_time;
      current_time += 10;

      // adding noise from the noise vector
      charge += Gaussian_Noise[Noise_index];
      Noise_index += 1;
      if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

      // main loop to add electrons to the counter
      if ( current_time > pix_time && pix_dex < pix_size)
      {
        // this adds the electrons that are in the step
        while( current_time > pix_time )
        {
          charge += 1;
          pix_dex += 1;
          if (pix_dex >= pix_size){break; }
          pix_time = Pixel[Hot_index].time[pix_dex] - OFFSET;
        }

      }

      // Current_File  << (charge*ElectronCharge_/10e-9)*1e9 << "," << current_time << ",";
      // Current_File  << current_time << "," << (charge*ElectronCharge_/10e-9)*1e9 << "\n";
      Current_File  << current_time << "," << (charge) << "\n";

    }
    Current_File.close();
    Current_File.clear();


    int ct=0;
    std::ofstream Reset_File;
    Reset_File.open(Reset_F);
    Reset_File << ct << "," << 0 << "\n";
    for (int pixx = 0; pixx < Pixel[Hot_index].RESET.size(); pixx++)
    {
      ct+=1;
      // Reset_File << ct << "," << Pixel[Hot_index].RESET[pixx] << ",";
      Reset_File << ct << "," << Pixel[Hot_index].RESET[pixx] << "\n";
    }
    Reset_File << ct << "," << End_Time ;
    Reset_File.close();
    Reset_File.clear();

    for (int i = 0; i < 20; i++ ) 
    {
      std::cout << Pixel[Hot_index].time[i] << std::endl;

    }

  }

};




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
  std::string file_in = "../mitch.root";
  std::string file_out = "../out_example.root";

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
 
  int evt = 1;
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

  std::string current_f = "current_electron_3MeV.csv";
  std::string reset_f   = "reset_electron_3MeV.csv";
  Current curr = Current();
  curr.Find_Hot_Pixel( Pixel, Gaussian_Noise, current_f, reset_f);


  rfm.AddEvent( Pixel );
  rfm.EventFill();
  rfm.EventReset();


  // save and close
  rfm.Save();

  


  std::cout << "done" << std::endl;
  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
