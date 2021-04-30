#ifndef ELECTRONICS_H_
#define ELECTRONICS_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>


namespace Qpix
{
    class Current_Profile
    {
        private:
        
        double const ElectronCharge_ = 1.60217662e-19;

        public:

        void Get_Hot_Current(Qpix::Qpix_Paramaters * Qpix_params, std::vector<Qpix::Pixel_Info>& Pixel, std::vector<double>& Gaussian_Noise, std::string Current_F, std::string Reset_F);
    };

    class Snip
    {
    public:

    void Snipped_RTD( std::vector<Qpix::Pixel_Info> Pixel,  std::string File_Name);

    };




}
#endif