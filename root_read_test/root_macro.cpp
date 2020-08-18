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
