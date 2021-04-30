#include "Structures.h"
#include "Electronics.h"


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
        double current_time = 0;
        int Noise_index = 0;
        int Noise_Vector_Size = Gaussian_Noise.size();


        double pix_time = Pixel[Hot_index].time[pix_dex];

        // int End_Time = Qpix_params->Buffer_time / Qpix_params->Sample_time;
        bool End_Reached = false;

        std::ofstream Current_File;
        Current_File.open(Current_F);

        // for each pixel loop through the buffer time
        while (current_time <= Qpix_params->Buffer_time)
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
                //End_Time = pix_time +10000;
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

        Reset_File << ct << "," << Qpix_params->Buffer_time ;
        Reset_File.close();
        Reset_File.clear();

    }// Get_Hot_Current



    void Snip::Snipped_RTD( std::vector<Qpix::Pixel_Info> Pixel,  std::string File_Name)
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
        std::cout << "Hot_index = " << Hot_index << std::endl;
        std::cout << "Hot Pixel info = " << Pixel[Hot_index].X_Pix << " , "<< Pixel[Hot_index].Y_Pix << std::endl;

        std::cout << "X Pix   " << "Y Pix" << std::endl;

        std::ofstream Reset_File;
        Reset_File.open(File_Name);

        int X_dex = Pixel[Hot_index].X_Pix - 5;
        int Y_dex = Pixel[Hot_index].Y_Pix - 5;

        for (int row = 0; row < 11; row++)
        {
        for (int col = 0; col < 11; col++)
        {
            for (int pixx = 0; pixx < Pixel.size(); pixx++)
            {
            if ( Y_dex == Pixel[pixx].Y_Pix && X_dex == Pixel[pixx].X_Pix )
            {
                std::cout << "Hot Pixel info = " << Pixel[pixx].X_Pix << " , "<< Pixel[pixx].Y_Pix << std::endl;
                std::cout << "Size  = " << Pixel[pixx].RESET.size() << std::endl;
                for (int T = 0; T < Pixel[pixx].RESET.size(); T++)
                {
                Reset_File << X_dex << "," << Y_dex << "," << Pixel[pixx].RESET[T] << "\n";
                }

            }
            }
            X_dex += 1;

        }
        Y_dex += 1;
        X_dex = Pixel[Hot_index].X_Pix - 5;

        }

        Reset_File.close();
        Reset_File.clear();

    }//Snipped_RTD


}
