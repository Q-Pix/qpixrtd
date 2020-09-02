#include <iostream>
#include <string>
#include <vector>
#include <math.h>

// #include "Qpix/Random.h"
#include "Qpix/Structures.h"
#include "Qpix/PixelResponse.h"


namespace Qpix
{


    // Decodes the pixel ID into x and y pix
    void Pixel_Functions::ID_Decoder(int const& ID, int& Xcurr, int& Ycurr)
    {
        double PixID = ID/10000.0, fractpart, intpart;
        fractpart = modf (PixID , &intpart);
        Xcurr = (int) round(intpart);
        Ycurr = (int) round(fractpart*10000); 
        return;
    }//ID_Decoder


    // sorts the electrons in to a pixel structure 
    void Pixel_Functions::Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::vector<Pixel_Info>& Pix_info)
    {
        int Event_Length = hit_e.size();
        
        std::vector<int> NewID_Index;
        int newID = 0;

        // this loop finds the index where the pixels change
        for (int i=0; i<Event_Length ; i++)
        {
            if ( newID != hit_e[i].Pix_ID )
            {
                NewID_Index.push_back( i );
                newID = hit_e[i].Pix_ID;
            }
        
        }
        NewID_Index.push_back( hit_e.size() );

        // Now loop though thoues to make a pixel opject that is X, Y and a vect of times
        int N_Index = NewID_Index.size() - 1;
        for (int i=0; i<N_Index ; i++)
        {
            std::vector<Qpix::ELECTRON> sub_vec = slice(hit_e, NewID_Index[i], NewID_Index[i+1] -1 );
            std::sort( sub_vec.begin(), sub_vec.end(), &Pixel_Time_Sorter );
            std::vector<int> tmp_time;

            for (int j=0; j<sub_vec.size() ; j++) { tmp_time.push_back( sub_vec[j].time ); }

            int Pix_Xloc, Pix_Yloc ;
            ID_Decoder(sub_vec[0].Pix_ID, Pix_Xloc, Pix_Yloc);
            Pix_info.push_back(Pixel_Info());
            Pix_info[i].ID    = sub_vec[0].Pix_ID;
            Pix_info[i].X_Pix = Pix_Xloc;
            Pix_info[i].Y_Pix = Pix_Yloc;
            Pix_info[i].time  = tmp_time;
        }
        return;
    }//Pixelize_Event



    // function performs the resets 
    void Pixel_Functions::Reset(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        // The number of steps to cover the full buffer
        int End_Time = Qpix_params->Buffer_time / Qpix_params->Sample_time;

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
            int tslr_ = 0;
            std::vector<int>  TSLR;

            // skip if it wont reset
            if (pix_size < (Qpix_params->Reset)*0.5){continue;}

            // for each pixel loop through the buffer time
            while (current_time <= End_Time)
            {
                // setting the "time"
                current_time += Qpix_params->Sample_time;

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
                if ( charge >= Qpix_params->Reset )
                {

                    TSLR.push_back(current_time - tslr_);
                    tslr_ = current_time;

                    RESET.push_back( current_time );
                    charge -= Qpix_params->Reset;

                    // this will keep the charge in the loop above
                    // just offsets the reset by the dead time
                    current_time += Qpix_params->Dead_time;

                    // condition for charge loss
                    // just the main loop without the charge
                    if (Qpix_params->charge_loss)
                    {
                        while( current_time > pix_time )
                        {
                            pix_dex += 1;
                            // if (pix_dex < pix_size){ pix_time = Pix_info[i].time[pix_dex]; }

                            if (pix_dex >= pix_size){break; }
                            pix_time = Pix_info[i].time[pix_dex];
                        }
                    }
                }
            }
            // add it to the pixel info
            Pix_info[i].RESET = RESET;
            Pix_info[i].TSLR  = TSLR;
        }

        return ;
    }// Reset

}