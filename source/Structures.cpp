// C++ includes
#include <iostream>
#include <string>
#include <vector>

#include "Qpix/Structures.h"

namespace Qpix
{
    bool Electron_Pix_Sort(ELECTRON one, ELECTRON two)
    {
        return (one.Pix_ID < two.Pix_ID);
    }

    bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs) 
    {
        return lhs.time < rhs.time;
    }


    void set_Qpix_Paramaters(Qpix_Paramaters * Qpix_params)
    {
        Qpix_params->Wvalue = 23.6; // in eV
        //E_vel = 0.1648; //cm/mus
        Qpix_params->E_vel = 1.648e-4; // cm/ns
        Qpix_params->DiffusionL = 6.8223 * 1e-9;  //cm**2/ns
        Qpix_params->DiffusionT = 13.1586 * 1e-9; //cm**2/ns
        Qpix_params->Life_Time = 100000000; // in ns
        // Read out plane size in cm
        Qpix_params->Readout_Dim = 100;
        Qpix_params->Pix_Size = 0.4;
        // Number of electrons for reset
        Qpix_params->Reset = 6250;
        // time in ns
        Qpix_params->Sample_time = 10;
        Qpix_params->Buffer_time = 1e8;
        Qpix_params->Dead_time = 0;
        Qpix_params->charge_loss = false;
    }

    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params)
    {
        std::cout << "*******************************************************" << std::endl;
        std::cout << "Liquid Argon Paramaters" << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout   << "W value                    = " << Qpix_params->Wvalue << " [eV] \n"
                    << "Dirft velocity             = " << Qpix_params->E_vel<< " [cm/ns] \n"
                    << "Longitidunal diffusion     = " << Qpix_params->DiffusionL<< " [cm^2/ns] \n"
                    << "Transverse diffusion       = " << Qpix_params->DiffusionT<< " [cm^2/ns] \n"
                    << "Electron life time         = " << Qpix_params->Life_Time<< " [ns] \n"
                    << "Readout dimensions         = " << Qpix_params->Readout_Dim<< " [cm] \n"
                    << "Pixel size                 = " << Qpix_params->Pix_Size<< " [cm] \n"
                    << "Reset threshold            = " << Qpix_params->Reset<< " [electrons] \n"
                    << "Sample time                = " << Qpix_params->Sample_time<< " [ns] \n"
                    << "Buffer window              = " << Qpix_params->Buffer_time<< " [ns] \n"
                    << "Dead time                  = " << Qpix_params->Dead_time<< " [ns] \n"
                    << "Charge loss                = " << Qpix_params->charge_loss<< " [yes/no] \n"
                    << std::endl;
        std::cout << "*******************************************************" << std::endl;
    }






}