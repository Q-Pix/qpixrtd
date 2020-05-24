#include "Qpix/Random.h"
#include "Qpix/ReadG4txt.h"
#include "Qpix/ElectronHandler.h"
#include "Qpix/PixelResponse.h"

#include <algorithm>
#include <ctime>


int main() 
{  
   clock_t time_req;
   time_req = clock();
   
   double Wvalue, E_vel, DiffusionL, DiffusionT, Life_Time;
   Wvalue = 23.6; // in eV
   //E_vel = 0.1648; //cm/mus
   E_vel = 1.648; //mm/mus
   DiffusionL = 682.23/1e6;  //mm**2/mus
   DiffusionT = 1315.86/1e6; //mm**2/mus
   Life_Time = 10000; // in mus

   //std::string Input_File  = "Muon_100.txt";
   std::string Input_File  = "Ar39_10k.txt";
   //std::string Input_File  = "Proton_100.txt";
   std::string Output_File = "Example01_Output.txt";

   // Read out plane size in mm
   int Readout_Dim = 1000;
   int Pix_Size = 4;

   // Number of electrons for reset
   int Reset = 6000;
   //int Reset = 1000;


   // Makes the Pixel plane with the params set above
   std::cout << "*********************************************" << std::endl;
   std::cout << "Making the readout plane" << std::endl;
   std::vector<std::vector<int>> data2d;
   data2d = Qpix::Make_Readout_plane(Readout_Dim, Pix_Size, Reset);


   // makes a noise vector to add to the pixels
   std::cout << "*********************************************" << std::endl;
   std::cout << "Making the noise vector" << std::endl;
   std::vector<double> Gaussian_Noise;
   int Noise_Vector_Size = 10000;
   Gaussian_Noise = Qpix::Make_Gaussian_Noise(30, Noise_Vector_Size);
   //int Noise_index = 0;


   // Loads the Geant4 output
   std::cout << "*********************************************" << std::endl;
   std::cout << "Loading the G4 data" << std::endl;
   // here x,y,z are in mm and edep is in MeV
   std::vector<std::vector<double>> Raw_Data_Vector;
   std::vector<int> Event_Lengths;
   Qpix::DataFileParser2(Input_File, Raw_Data_Vector, Event_Lengths);

   // Selects a specific event in the geant4 output 
   int Event = 0;
   std::vector<std::vector<double>> Specific_Event;
   Specific_Event =  Qpix::GetEventVector(Event ,  Event_Lengths,  Raw_Data_Vector);


   // making the diffuseld and projected electron cloud
   std::cout << "*********************************************" << std::endl;
   std::cout << "Making the diffused electron cloud" << std::endl;
   std::vector<Qpix::HIT> Electron_Event_Vector;
   Electron_Event_Vector = Qpix::Diffuser( Wvalue, E_vel, Life_Time, DiffusionL, DiffusionT, Specific_Event);

   int Event_Length = Electron_Event_Vector.size();
   std::cout << "The cloud contains " << Event_Length << " electrons" << std::endl;

   
   // finds the unique pixels hit 
   // vect of hit pixels 
   std::vector<std::vector<int>> Pixels_Hit;
   Pixels_Hit = Qpix::Find_Unique_Pixels(Pix_Size, Event_Length, Electron_Event_Vector);

   int Pixels_Hit_Len = Pixels_Hit.size();
   std::cout << "Which hit "<< Pixels_Hit_Len << " unique pixels" << std::endl;



   // Projecting the electrons in time onto the pixels 
   std::cout << "*********************************************" << std::endl;
   std::cout << "Starting the Qpix response" << std::endl;
   // if using the deat time you need this guy
   //std::vector<std::vector<double>> Dead;
   //Dead = Qpix::Make_Dead_Time(Readout_Dim, Pix_Size, Reset);


   std::vector<std::vector<double>> RTD;
   // event time window in microseconds
   int Start_Time = 0, End_Time = 3200;
   RTD = Qpix::Make_Reset_Response(Reset, Pix_Size, E_vel, Event_Length, Pixels_Hit_Len, Noise_Vector_Size, Start_Time, End_Time,
                               Gaussian_Noise, Pixels_Hit, data2d, Electron_Event_Vector);
   //RTD = Qpix::Make_Reset_Response_DeadTime(Reset, Pix_Size, E_vel, Event_Length, Pixels_Hit_Len, Noise_Vector_Size, Start_Time, End_Time,
   //                            Gaussian_Noise, Pixels_Hit, data2d, Electron_Event_Vector, Dead);



   std::cout << "*********************************************" << std::endl;
   std::cout << "Writing the output file" << std::endl;
   // writes the reset as Xpix, Ypix, reset time, time since last reset
   Qpix::Write_Reset_Data( Output_File, Event, RTD);


   std::cout << "done" << std::endl;
   time_req = clock() - time_req;
   double time = (float)time_req/CLOCKS_PER_SEC;
   std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
   return 0;
}
