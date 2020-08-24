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



//----------------------------------------------------------------------
// main function
//----------------------------------------------------------------------
int main()
{
    clock_t time_req;
    time_req = clock();
    double time;

    std::vector< std::string > file_list_ = 
    {
        "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test.root",
        "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test2.root"
    };

  // std::vector< std::string > file_list_ = 
  // {
  //   "/Users/austinmcdonald/projects/Q_PIX_NEW/Q_PIX_RTD/root_read_test/test_muon.root"
  // };


    Qpix::Liquid_Argon_Paramaters * LAr_params = new Qpix::Liquid_Argon_Paramaters();
    set_Liquid_Argon_Paramaters(LAr_params);
    //LAr_params->charge_loss = true;
    LAr_params->Buffer_time = 1e8;
    print_Liquid_Argon_Paramaters(LAr_params);


    Qpix::READ_G4_ROOT reader = Qpix::READ_G4_ROOT();
    reader.Open_File(file_list_);

    std::vector<Qpix::ELECTRON> hit_e;
    int evt=1;
    reader.Get_Event( evt, LAr_params, hit_e);
    std::cout << "size of hit_e = " << hit_e.size() << std::endl;

    for (int i = 0; i < 20; i++) 
    {
        std::vector<Qpix::ELECTRON> hit_e;
        reader.Get_Event( i, LAr_params, hit_e);
        std::cout << "size of hit_e = " << hit_e.size() << std::endl;
    }


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
    int Noise_Vector_Size = (int) 1e6;
    Gaussian_Noise = Qpix::Make_Gaussian_Noise(2, Noise_Vector_Size);

    time_req = clock() - time_req;
    time = (float)time_req/CLOCKS_PER_SEC;
    std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;



    PixFunc.Reset(LAr_params, Gaussian_Noise, Pixel);
    // for (int i=0; i<Pixel.size() ; i++)
    // {
    //     std::cout << "NEW" << std::endl;
    //     //std::cout << Pixel[i].ID    << "\t" ;

    //     for ( int j=0; j<Pixel[i].RESET.size(); j++)
    //     {
    //     std::cout << Pixel[i].ID    << "\t" 
    //                 << Pixel[i].X_Pix << "\t" 
    //                 << Pixel[i].Y_Pix << "\t" 
    //                 << Pixel[i].RESET[j] << std::endl;
    //     }
    
    // }


 

  // std::vector<Qpix::ELECTRON> hit_e;
  // Qpix::ROOT_READ(file_list_, evt, LAr_params, hit_e);

  // std::cout << "size of hit_e = " << hit_e.size() << std::endl;

  // time_req = clock() - time_req;
  // time = (float)time_req/CLOCKS_PER_SEC;
  // std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;



  // std::vector<Pixel_Info> Pixel;
  // Pixelize_Event( hit_e, Pixel );

  // time_req = clock() - time_req;
  // time = (float)time_req/CLOCKS_PER_SEC;
  // std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  // std::cout<< Pixel.size() <<std::endl;
  



  std::cout << "done" << std::endl;

  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
