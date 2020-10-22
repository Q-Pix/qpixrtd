#include "Qpix/Structures.h"
#include "Qpix/Electronics.h"


namespace Qpix
{

    void Current_Profile::Get_Hot_Current(Qpix::Qpix_Paramaters * Qpix_params, std::vector<Qpix::Pixel_Info>& Pixel, std::vector<double>& Gaussian_Noise, std::string Current_F, std::string Reset_F)
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
        int Icharge= 0;
        int pix_size = Pixel[Hot_index].time.size();
        int pix_dex = 0;
        int current_time = 0;
        int Noise_index = 0;
        int Noise_Vector_Size = Gaussian_Noise.size();


        int pix_time = Pixel[Hot_index].time[pix_dex];

        int End_Time = Qpix_params->Buffer_time / Qpix_params->Sample_time;
        bool End_Reached = false;

        std::ofstream Current_File;
        Current_File.open(Current_F);

        // for each pixel loop through the buffer time
        while (current_time <= End_Time)
        {
        // setting the "time"
        current_time += Qpix_params->Sample_time;

        // adding noise from the noise vector
        Icharge = Gaussian_Noise[Noise_index];

        Noise_index += 1;
        if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

        // main loop to add electrons to the counter
        if ( current_time > pix_time && pix_dex < pix_size)
        {
            // this adds the electrons that are in the step
            while( current_time > pix_time )
            {
            Icharge += 1;
            pix_dex += 1;
            if (pix_dex >= pix_size){break; }
            pix_time = Pixel[Hot_index].time[pix_dex];
            }
            charge += Icharge;

        }
        if (pix_dex >= pix_size && !End_Reached)
        {
            End_Reached = true;
            End_Time = pix_time +10000;
        }

        // write the instanuoous and cummlitive currents
        Current_File  << current_time << "," << ((Icharge*ElectronCharge_/10e-9)*1e9) << "," << ((charge*ElectronCharge_/10e-9)*1e9) << "\n";

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
        Reset_File << ct << "," << Pixel[Hot_index].RESET[pixx] << "\n";
        }

        Reset_File << ct << "," << End_Time ;
        Reset_File.close();
        Reset_File.clear();

    }// Get_Hot_Current



}