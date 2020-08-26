// C++ includes
#include <iostream>
#include <string>
#include <vector>


#include "Qpix/Random.h"
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



    // struct Liquid_Argon_Paramaters 
    // {
    //     double Wvalue = 0;
    //     double E_vel = 0;
    //     double DiffusionL = 0;
    //     double DiffusionT = 0;
    //     double Life_Time = 0;
    //     double Readout_Dim = 0;
    //     double Pix_Size = 0;
    //     int Reset = 0;

    //     int Sample_time = 0;
    //     int Buffer_time = 0;
    //     int Dead_time = 0;
    //     bool charge_loss = false;
    // };

    void set_Liquid_Argon_Paramaters(Liquid_Argon_Paramaters * LAr_params)
    {
        LAr_params->Wvalue = 23.6; // in eV
        //E_vel = 0.1648; //cm/mus
        LAr_params->E_vel = 1.648e-4; // cm/ns
        LAr_params->DiffusionL = 6.8223 * 1e-9;  //cm**2/ns
        LAr_params->DiffusionT = 13.1586 * 1e-9; //cm**2/ns
        LAr_params->Life_Time = 100000000; // in ns
        // Read out plane size in cm
        LAr_params->Readout_Dim = 100;
        LAr_params->Pix_Size = 0.4;
        // Number of electrons for reset
        LAr_params->Reset = 6250;
        // time in ns
        LAr_params->Sample_time = 10;
        LAr_params->Buffer_time = 1e8;
        LAr_params->Dead_time = 0;
        LAr_params->charge_loss = false;
    }

    void print_Liquid_Argon_Paramaters(Liquid_Argon_Paramaters * LAr_params)
    {
        std::cout << "*******************************************************" << std::endl;
        std::cout << "Liquid Argon Paramaters" << std::endl;
        std::cout << "*******************************************************" << std::endl;
        std::cout << "W value                    = " << LAr_params->Wvalue << " [eV] \n"
                    << "Dirft velocity             = " << LAr_params->E_vel<< " [cm/ns] \n"
                    << "Longitidunal diffusion     = " << LAr_params->DiffusionL<< " [cm^2/ns] \n"
                    << "Transverse diffusion       = " << LAr_params->DiffusionT<< " [cm^2/ns] \n"
                    << "Electron life time         = " << LAr_params->Life_Time<< " [ns] \n"
                    << "Readout dimensions         = " << LAr_params->Readout_Dim<< " [cm] \n"
                    << "Pixel size                 = " << LAr_params->Pix_Size<< " [cm] \n"
                    << "Reset threshold            = " << LAr_params->Reset<< " [electrons] \n"
                    << "Sample time                = " << LAr_params->Sample_time<< " [ns] \n"
                    << "Buffer window              = " << LAr_params->Buffer_time<< " [ns] \n"
                    << "Dead time                  = " << LAr_params->Dead_time<< " [ns] \n"
                    << "Charge loss                = " << LAr_params->charge_loss<< " [yes/no] \n"
                    << std::endl;
        std::cout << "*******************************************************" << std::endl;
    }






}