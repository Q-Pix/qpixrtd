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


#include "Qpix/ReadG4root.h"


//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main()
{

  std::vector< std::string > file_list_ = 
  {
    "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test.root",
    "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test2.root"
  };
  int evt=1;
  Qpix::ROOT_READ(file_list_, evt);


  //std::cout << "size of hit_end_x_ = " << hit_end_x_.size() << std::endl;
  return 0;
} // main()
