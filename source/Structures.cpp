// C++ includes
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <nlohmann/json.hpp>


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

        std::ifstream param_file("params.json");
        nlohmann::json QPixParamsFile = nlohmann::json::parse(param_file);

        Qpix_params->Wvalue = QPixParamsFile["QPix"]["Wvalue"]; // in eV
        Qpix_params->E_vel = QPixParamsFile["QPix"]["E_vel"]; // cm/s
        Qpix_params->DiffusionL = QPixParamsFile["QPix"]["DiffusionL"];  //cm**2/s
        Qpix_params->DiffusionT = QPixParamsFile["QPix"]["DiffusionT"]; //cm**2/s
        Qpix_params->Life_Time = QPixParamsFile["QPix"]["Life_Time"]; // in s

        // Read out plane size in cm
        Qpix_params->Readout_Dim = QPixParamsFile["QPix"]["Readout_Dim"];
        Qpix_params->Pix_Size = QPixParamsFile["QPix"]["Pix_Size"];

        // Number of electrons for reset
        Qpix_params->Reset = QPixParamsFile["QPix"]["Reset"];
        // time in ns

        Qpix_params->Sample_time = QPixParamsFile["QPix"]["Sample_time"]; // in s 
        Qpix_params->Buffer_time = QPixParamsFile["QPix"]["Buffer_time"]; // in s 
        Qpix_params->Dead_time = QPixParamsFile["QPix"]["Dead_time"]; // in s 
        Qpix_params->Charge_loss = QPixParamsFile["QPix"]["Charge_loss"];
        Qpix_params->Recombination = QPixParamsFile["QPix"]["Recombination"];
    }//set_Qpix_Paramaters

    // A nice printing function 
    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params)
    {
        std::cout << "*******************************************************" << std::endl;
        std::cout << "Liquid Argon Paramaters" << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout   << "W value                    = " << Qpix_params->Wvalue << " [eV] \n"
                    << "Dirft velocity             = " << Qpix_params->E_vel<< " [cm/s] \n"
                    << "Longitidunal diffusion     = " << Qpix_params->DiffusionL<< " [cm^2/s] \n"
                    << "Transverse diffusion       = " << Qpix_params->DiffusionT<< " [cm^2/s] \n"
                    << "Electron life time         = " << Qpix_params->Life_Time<< " [s] \n"
                    << "Readout dimensions         = " << Qpix_params->Readout_Dim<< " [cm] \n"
                    << "Pixel size                 = " << Qpix_params->Pix_Size<< " [cm] \n"
                    << "Reset threshold            = " << Qpix_params->Reset<< " [electrons] \n"
                    << "Sample time                = " << Qpix_params->Sample_time<< " [s] \n"
                    << "Buffer window              = " << Qpix_params->Buffer_time<< " [s] \n"
                    << "Dead time                  = " << Qpix_params->Dead_time<< " [s] "
                    << std::endl;
        if (Qpix_params->Charge_loss)
        {std::cout << "Charge loss                = " << "YES" << " [yes/no] " << std::endl;}
        else{std::cout << "Charge loss                = " << "NO" << " [yes/no] " << std::endl;}
        if (Qpix_params->Recombination)
        {std::cout << "Recombination              = " << "YES" << " [yes/no] " << std::endl;}
        else{std::cout << "Recombination                = " << "NO" << " [yes/no] " << std::endl;}
        
        std::cout << "*******************************************************" << std::endl;
    }//print_Qpix_Paramaters
    
}
