// C++ includes
#include <iostream>
#include <string>
#include <vector>
#include <map>

// see this include for the actiual structs...
#include "Structures.h"

namespace Qpix
{
    // sort the elctrons by pixel index
    bool Electron_Pix_Sort(ELECTRON one, ELECTRON two)
    {
        return (one.Pix_ID < two.Pix_ID);
    }//Electron_Pix_Sort



    //sort the electrons in a pixel by time
    bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs) 
    {
        return lhs.time < rhs.time;
    }//Pixel_Time_Sorter



    void Get_Frequencies(std::vector<int> vec, std::vector<int>& TrkIDs, std::vector<int>& weight )
    {   // Define an map
        std::map<int, int> M;
    
        // check if current element is present
        for (int i = 0; i < vec.size(); i++)
        {   // If the current element is not found then insert
            // current element with frequency 1
            if (M.find(vec[i]) == M.end()) { M[vec[i]] = 1; }
            // Else update the frequency
            else { M[vec[i]]++; }
        }
        // Traverse the map to print the frequency
        for (auto& it : M) 
        {
            TrkIDs.push_back(it.first );
            weight.push_back(it.second);
        }
    }//Get_Frequencies




    // setup the default Qpix paramaters
    void set_Qpix_Paramaters(Qpix_Paramaters * Qpix_params)
    {
        Qpix_params->Wvalue = 23.6; // in eV
        Qpix_params->E_vel = 164800.0; // cm/s
        Qpix_params->DiffusionL = 6.8223  ;  //cm**2/s
        Qpix_params->DiffusionT = 13.1586 ; //cm**2/s
        Qpix_params->Life_Time = 0.1; // in s

        // Read out plane size in cm
        Qpix_params->Readout_Dim = 100;
        Qpix_params->Pix_Size = 0.4;

        // Number of electrons for reset
        Qpix_params->Reset = 6250;

        // Set the downsampling factor (1 means no downsampling/full statistics)
        // Must be less than or equal to 1
        Qpix_params->Sampling = 1;
        
        Qpix_params->Sample_time = 10e-9; // in s 
        Qpix_params->Buffer_time = 1; // in s 
        Qpix_params->Dead_time = 0; // in s 
        Qpix_params->Charge_loss = false;
        Qpix_params->Recombination = true;
        Qpix_params->Noise = true;
        Qpix_params->TimeWindow = true;
    }//set_Qpix_Paramaters



    // A nice printing function 
    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params)
    {
        std::cout << "**********************************************************" << std::endl;
        std::cout << "Liquid Argon Paramaters" << std::endl;
        std::cout << "**********************************************************" << std::endl;
        std::cout   << "W value                       = " << Qpix_params->Wvalue << " [eV] \n"
                    << "Drift velocity                = " << Qpix_params->E_vel<< " [cm/s] \n"
                    << "Longitidunal diffusion        = " << Qpix_params->DiffusionL<< " [cm^2/s] \n"
                    << "Transverse diffusion          = " << Qpix_params->DiffusionT<< " [cm^2/s] \n"
                    << "Electron life time            = " << Qpix_params->Life_Time<< " [s] \n"
                    << "Readout dimensions            = " << Qpix_params->Readout_Dim<< " [cm] \n"
                    << "Pixel size                    = " << Qpix_params->Pix_Size<< " [cm] \n"
                    << "Reset threshold               = " << Qpix_params->Reset<< " [electrons] \n"
                    << "Sample time                   = " << Qpix_params->Sample_time<< " [s] \n"
                    << "Buffer window                 = " << Qpix_params->Buffer_time<< " [s] \n"
                    << "Dead time                     = " << Qpix_params->Dead_time<< " [s] \n"
                    << "Downsampling (1 = full stats) = " << Qpix_params->Sampling
                    << std::endl;
        if (Qpix_params->Charge_loss)
        {std::cout << "Charge loss                   = " << "YES" << " [yes/no] " << std::endl;}
        else{std::cout << "Charge loss                   = " << "NO" << " [yes/no] " << std::endl;}
        if (Qpix_params->Recombination)
        {std::cout << "Recombination                 = " << "YES" << " [yes/no] " << std::endl;}
        else{std::cout << "Recombination                 = " << "NO" << " [yes/no] " << std::endl;}
        if (Qpix_params->Noise)
        {std::cout << "Noise                         = " << "YES" << " [yes/no] " << std::endl;}
        else{std::cout << "Noise                         = " << "NO" << " [yes/no] " << std::endl;}
        if (Qpix_params->TimeWindow)
        {std::cout << "TimeWindow                    = " << "YES" << " [yes/no] " << std::endl;}
        else{std::cout << "TimeWindow                    = " << "NO" << " [yes/no] " << std::endl;}       
 
        std::cout << "**********************************************************" << std::endl;
    }//print_Qpix_Paramaters
    
}
