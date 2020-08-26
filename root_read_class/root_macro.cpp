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

// Qpix includes
#include "Qpix/ReadG4root.h"
#include "Qpix/Random.h"
#include "Qpix/Structures.h"
#include "Qpix/PixelResponse.h"

#include <ctime>




// ROOT includes
#include "TBranch.h"
#include "TChain.h"
#include "TFile.h"
#include "TROOT.h"



//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main()
{
  clock_t time_req;
  time_req = clock();
  double time;

  // changing the seed for the random numbergenerator 
  constexpr std::uint64_t Seed = 777;
  Qpix::Random_Set_Seed(Seed);

  // std::vector< std::string > file_list_ = 
  // {
  //     "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test.root",
  //     "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test2.root"
  // };

  // std::vector< std::string > file_list_ = 
  // {
  //   "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test_muon.root"
  // };

  std::string file_ = "/n/home02/jh/repos/Q_PIX_RTD/root_read_class/test.root";
  std::string file_path = "/n/home02/jh/repos/Q_PIX_RTD/root_read_class/out.root";

  Qpix::Liquid_Argon_Paramaters * LAr_params = new Qpix::Liquid_Argon_Paramaters();
  set_Liquid_Argon_Paramaters(LAr_params);
  //LAr_params->charge_loss = true;
  LAr_params->Buffer_time = 1e8;
  print_Liquid_Argon_Paramaters(LAr_params);


  Qpix::READ_G4_ROOT reader = Qpix::READ_G4_ROOT();
  reader.Open_File(file_);
  // reader.Open_File(file_list_);

  std::vector<Qpix::ELECTRON> hit_e;
  int evt=1;
  reader.Get_Event( evt, LAr_params, hit_e);
  std::cout << "size of hit_e = " << hit_e.size() << std::endl;

  // for (int i = 0; i < 20; i++) 
  // {
  //   std::vector<Qpix::ELECTRON> hit_e;
  //   reader.Get_Event( i, LAr_params, hit_e);
  //   std::cout << "size of hit_e = " << hit_e.size() << std::endl;
  // }


  Qpix::Pixel_Functions PixFunc = Qpix::Pixel_Functions();
  std::vector<Qpix::Pixel_Info> Pixel;
  PixFunc.Pixelize_Event( hit_e, Pixel );

  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  std::cout<< Pixel.size() <<std::endl;


  std::cout << "*********************************************" << std::endl;
  std::cout << "Making the noise vector" << std::endl;
  std::vector<double> Gaussian_Noise;
  // int Noise_Vector_Size = (int) 1e6;
  int Noise_Vector_Size = (int) 1e5;
  Gaussian_Noise = Qpix::Make_Gaussian_Noise(2, Noise_Vector_Size);

  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;


  // the reset function
  PixFunc.Reset(LAr_params, Gaussian_Noise, Pixel);



  // root file stuff
  int run_ = 1;
  int event_ = 1;

  // ROOT objects
  TFile * tfile_;
  TTree * ttree_;

  tfile_ = new TFile(file_path.data(), "recreate", "q_pix_geant");

  // GEANT4 event tree
  ttree_ = new TTree("RTD_event_tree", "RTD event tree");

  ttree_->Branch("run",   &run_,   "run/I");
  ttree_->Branch("event", &event_, "event/I");

  // vectors for storing pixel information
  std::vector< int > x_;
  std::vector< int > y_;
  std::vector< std::vector < double > > reset_;
  // std::vector< std::vector < int > > reset_;

  ttree_->Branch("x", &x_);
  ttree_->Branch("y", &y_);
  ttree_->Branch("reset", &reset_);

  // prints the pixel info
  for (unsigned int i=0; i<Pixel.size() ; i++)
  {
      std::cout << "NEW" << std::endl;
      //std::cout << Pixel[i].ID    << "\t" ;

      // vector of resets as double instead of int for workaround
      std::vector< double > reset_double;

      for (unsigned int j=0; j<Pixel[i].RESET.size(); j++)
      {
      std::cout << Pixel[i].ID    << "\t"
                  << Pixel[i].X_Pix << "\t"
                  << Pixel[i].Y_Pix << "\t"
                  << Pixel[i].RESET[j] << std::endl;

          // cast from int to double
          reset_double.push_back(static_cast<double>(Pixel[i].RESET[j]));

      }

      // add to tree vectors
      x_.push_back(Pixel[i].X_Pix);
      y_.push_back(Pixel[i].Y_Pix);
      // reset_.push_back(Pixel[i].RESET);
      reset_.push_back(reset_double);

  }

  // set run and event numbers
  run_ = 1;
  event_ = 1;

  // fill tree
  ttree_->Fill();

  // reset variables
  run_ = -1;
  event_ = -1;
  x_.clear();
  y_.clear();
  reset_.clear();

  // get G4 and MARLEY trees from ROOT file

  TFile read_file(file_.data());

  TTree * read_g4_event_tree;
  TTree * read_marley_event_tree;

  read_file.GetObject("g4_event_tree", read_g4_event_tree);
  read_file.GetObject("marley_event_tree", read_marley_event_tree);

  // go into output file
  tfile_->cd();

  // clone G4 and MARLEY trees
  auto write_g4_event_tree = read_g4_event_tree->CloneTree();
  auto write_marley_event_tree = read_marley_event_tree->CloneTree();

  if (write_g4_event_tree == 0 or write_marley_event_tree)
  {
      // maybe throw an exception here
      std::cout << "Could not copy trees to output file!" << std::endl;
  }

  // write_g4_event_tree->Print();
  // write_marley_event_tree->Print();

  // write to output file
  tfile_->Write();
  tfile_->Close();

  std::cout << "done" << std::endl;

  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
