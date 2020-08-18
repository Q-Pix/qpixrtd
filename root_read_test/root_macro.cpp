// -----------------------------------------------------------------------------
//  root_macro.c
//
//  Example of a macro for reading the ROOT files produced from the
//  Q_PIX_GEANT4 program.
//   * Author: Everybody is an author!
//   * Creation date: 12 August 2020
// -----------------------------------------------------------------------------

// C++ includes
#include <iostream>
#include <string>
#include <vector>

// ROOT includes
// #include "TBranch.h"
// #include "TChain.h"
// #include "TFile.h"
// #include "TROOT.h"

// Qpix includes
#include "Qpix/ReadG4root.h"
#include "Qpix/PixelResponse.h"


#include <ctime>



//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main()
{
  clock_t time_req;
  time_req = clock();

  std::vector< std::string > file_list_ = 
  {
    "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test.root",
    "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test2.root"
  };


  Qpix::Liquid_Argon_Paramaters * LAr_params = new Qpix::Liquid_Argon_Paramaters();
  set_Liquid_Argon_Paramaters(LAr_params);
  print_Liquid_Argon_Paramaters(LAr_params);


  int evt=1;
  //std::vector<double> hit_e;
  std::vector<Qpix::ELECTRON> hit_e;
  Qpix::ROOT_READ(file_list_, evt, LAr_params, hit_e);



  std::cout << "size of hit_e = " << hit_e.size() << std::endl;


  std::cout << "*********************************************" << std::endl;
  std::cout << "Making the readout plane" << std::endl;
  std::vector<std::vector<int>> data2d;
  bool Make_Noise=false;
  data2d = Qpix::Setup_Readout_Plane(LAr_params, Make_Noise);



  std::vector<std::vector<int>> Pixels_Hit;
  Pixels_Hit = Qpix::Find_Unique_Pixels(LAr_params, hit_e);

  int Pixels_Hit_Len = Pixels_Hit.size();
  std::cout << "Which hit "<< Pixels_Hit_Len << " unique pixels" << std::endl;


  std::cout << "*********************************************" << std::endl;
  std::cout << "Making the noise vector" << std::endl;
  std::vector<double> Gaussian_Noise;
  int Noise_Vector_Size = 10000;
  Gaussian_Noise = Qpix::Make_Gaussian_Noise(30, Noise_Vector_Size);


  std::cout << "*********************************************" << std::endl;
  std::cout << "Starting the Qpix response" << std::endl;
  std::vector<std::vector<double>> RTD;
  RTD = Qpix::Reset_Response(LAr_params, Gaussian_Noise, Pixels_Hit, data2d, hit_e);



  for (int i = 0; i < RTD.size(); i++) 
  {
    std::cout << RTD[i][0] << "\t" << RTD[i][1] << "\t" << RTD[i][2] << std::endl;
  }


  std::cout << RTD.size() << std::endl;



  // for (int i = 0; i < hit_e.size(); i++) 
  // {
  //   std::cout << hit_e[i].x_pos << "\t"
  //             << hit_e[i].y_pos << "\t"
  //             << hit_e[i].z_pos << "\t"
  //             << hit_e[i].t_pos << "\t"
  //             << std::endl;

  // }

  



  std::cout << "done" << std::endl;

    time_req = clock() - time_req;
    double time = (float)time_req/CLOCKS_PER_SEC;
    std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
    return 0;

} // main()
