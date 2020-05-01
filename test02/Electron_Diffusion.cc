
//#include <iostream>
//#include <vector>
//#include <fstream>
//#include <math.h>

#include "Qpix/Random.h"
#include "Qpix/ReadG4txt.h"
#include "Qpix/ElectronHandler.h"

#include <ctime>

//#include <iostream>
//#include <fstream>


int main() 
{  
   clock_t time_req;
   time_req = clock();


   double Wvalue, E_vel, DiffusionL, DiffusionT;
   Wvalue     = 23.6; // in eV
   E_vel      = 1.648; //mm/mus
   DiffusionL = 682.23/1e6;  //mm**2/mus
   DiffusionT = 1315.86/1e6; //mm**2/mus

   // here x,y,z are in mm and edep is in MeV
   std::vector<std::vector<double>> RawDataVector2;
   std::vector<int> EventLengths2;
   Qpix::DataFileParser2("test10MeV.txt", RawDataVector2, EventLengths2);
   //Qpix::DataFileParser2("test10MeV.txt", RawDataVector2, EventLengths2);
   int Event = 0;
   std::vector<std::vector<double>> eventt2;
   eventt2 =  Qpix::GetEventVector(Event ,  EventLengths2,  RawDataVector2);

   std::vector<Qpix::HIT> Electron_Event_Vector;
   Electron_Event_Vector = Qpix::DiffuserTest2( Wvalue, E_vel, DiffusionL, DiffusionT, eventt2);

   std::ofstream rawdata;
   rawdata.open ("Diffused_Electron_Position.txt");

   for (int i = 0; i < Electron_Event_Vector.size(); i++) 
   { 
      std::cout<< Electron_Event_Vector[i].x_pos << " "; 
      std::cout<< Electron_Event_Vector[i].y_pos << " "; 
      std::cout<< Electron_Event_Vector[i].z_pos << " "; 
      std::cout<< "\n"; 

      rawdata << Electron_Event_Vector[i].x_pos << " " << Electron_Event_Vector[i].y_pos << " " << Electron_Event_Vector[i].z_pos << std::endl;
   } 
   rawdata.close();




   std::cout << "done" << std::endl;

   time_req = clock() - time_req;
   double time = (float)time_req/CLOCKS_PER_SEC;
   std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
   return 0;
}