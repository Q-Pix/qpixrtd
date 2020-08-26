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
#include "Qpix/PixelResponse.h"
#include "Qpix/Random.h"

#include <ctime>



// structure for holding the pixel info

struct Pixel_Info 
{
  int  X_Pix;
  int  Y_Pix;
  int ID;
  std::vector<int>  time;
  std::vector<int>  RESET;

};

void Reset(Qpix::Liquid_Argon_Paramaters * LAr_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
{
  // The number of steps to cover the full buffer
  int End_Time = LAr_params->Buffer_time / LAr_params->Sample_time;

  // geting the size of the vectors for looping
  int Pixels_Hit_Len = Pix_info.size();
  int Noise_Vector_Size = Gaussian_Noise.size();
  int Noise_index = 0;

  // loop over each pixel that was hit
  for (int i = 0; i < Pixels_Hit_Len; i++)
  {
    // seting up some parameters
    int charge = 0;
    int pix_size = Pix_info[i].time.size();
    int pix_dex = 0;
    int current_time = 0;
    int pix_time = Pix_info[i].time[pix_dex];
    std::vector<int>  RESET;

    // for each pixel loop through the buffer time
    while (current_time <= End_Time)
    {
      // setting the "time"
      current_time += LAr_params->Sample_time;

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
          pix_time = Pix_info[i].time[pix_dex];
        }

      }

      // this is the reset 
      if ( charge >= LAr_params->Reset )
      {
        RESET.push_back( current_time );
        charge = 0;

        // this will keep the charge in the loop above
        // just offsets the reset by the dead time
        current_time += LAr_params->Dead_time;

        // condition for charge loss
        // just the main loop without the charge
        if (LAr_params->charge_loss)
        {
          while( current_time > pix_time )
          {
            pix_dex += 1;
            if (pix_dex < pix_size){ pix_time = Pix_info[i].time[pix_dex]; }
          }
        }
      }
    }
    // add it to the pixel info
    Pix_info[i].RESET = RESET;
  }

  return ;
}// Reset








bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs) 
{
  return lhs.time < rhs.time;
}

void ID_Decoder(int const& ID, int& Xcurr, int& Ycurr)
{
  double PixID = ID/10000.0, fractpart, intpart;
  fractpart = modf (PixID , &intpart);
  Xcurr = (int)std::round(intpart);
  Ycurr = (int)std::round(fractpart*10000); 
  return;
}


template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return vec;
}






    


void Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::vector<Pixel_Info>& Pix_info)
{
  int Event_Length = hit_e.size();
  
  std::vector<int> NewID_Index;

  int newID = 0;

  for (int i=0; i<Event_Length ; i++)
  {
    if ( newID != hit_e[i].Pix_ID )
    {
      NewID_Index.push_back( i );
      newID = hit_e[i].Pix_ID;
    }
  
  }
  NewID_Index.push_back( hit_e.size() );


  int N_Index = NewID_Index.size() - 1;
  for (int i=0; i<N_Index ; i++)
  {
    std::vector<Qpix::ELECTRON> sub_vec = slice(hit_e, NewID_Index[i], NewID_Index[i+1] -1 );
    std::sort( sub_vec.begin(), sub_vec.end(), &Pixel_Time_Sorter );
    // if (i<5) std::cout << "NEW" << "\t" << i << std::endl;
    std::vector<int> tmp_time;

    for (int j=0; j<sub_vec.size() ; j++)
    {
      tmp_time.push_back( sub_vec[j].time );
        // if (i<5) std::cout << sub_vec[j].Pix_ID    << "\t" 
        //       << sub_vec[j].time << std::endl;
    }

    int Pix_Xloc, Pix_Yloc ;
    ID_Decoder(sub_vec[0].Pix_ID, Pix_Xloc, Pix_Yloc);
    Pix_info.push_back(Pixel_Info());
    Pix_info[i].ID    = sub_vec[0].Pix_ID;
    Pix_info[i].X_Pix = Pix_Xloc;
    Pix_info[i].Y_Pix = Pix_Yloc;
    Pix_info[i].time  = tmp_time;
  }
  return;
}




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

  int evt=1;

  std::vector<Qpix::ELECTRON> hit_e;
  Qpix::ROOT_READ(file_list_, evt, LAr_params, hit_e);

  std::cout << "size of hit_e = " << hit_e.size() << std::endl;

  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;



  std::vector<Pixel_Info> Pixel;
  Pixelize_Event( hit_e, Pixel );

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

  // double nn=0;
  // for (int j=0; j<Gaussian_Noise.size(); j++)
  // {
  //   //nn+=Gaussian_Noise[j];
  //   if (j % 100)
  //     std::cout << Gaussian_Noise[j] << std::endl;
  // }

  // std::cout << nn << std::endl;

  // std::cout << "*********************************************" << std::endl;
  // std::cout << "finding unique pixels" << std::endl;
  // std::vector<Unique_Pix> Pix_info;
  // AAA_Find_Unique_Pixels(LAr_params, hit_e, Pix_info);

  // time_req = clock() - time_req;
  // time = (float)time_req/CLOCKS_PER_SEC;
  // std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;

  
  std::cout << "*********************************************" << std::endl;
  std::cout << "Starting the Qpix response" << std::endl;
  // std::vector<std::vector<double>> RTD;
  // RTD = Reset(LAr_params, Gaussian_Noise, Pix_info, data2d, hit_e);
  Reset(LAr_params, Gaussian_Noise, Pixel);
  
  
  for (int i=0; i<Pixel.size() ; i++)
  {
    std::cout << "NEW" << std::endl;
    //std::cout << Pixel[i].ID    << "\t" ;

    for ( int j=0; j<Pixel[i].RESET.size(); j++)
    {
      std::cout << Pixel[i].ID    << "\t" 
                << Pixel[i].X_Pix << "\t" 
                << Pixel[i].Y_Pix << "\t" 
                << Pixel[i].RESET[j] << std::endl;
    }
  
  }


  std::cout << "done" << std::endl;

  time_req = clock() - time_req;
  time = (float)time_req/CLOCKS_PER_SEC;
  std::cout<< "The operation took "<<time<<" Seconds"<<std::endl;
  return 0;

} // main()
