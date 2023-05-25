#ifndef PIXELRESPONSE_H_
#define PIXELRESPONSE_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>

namespace Qpix
{
    class Pixel_Functions
    {
        public:

        void ID_Decoder(int const& ID, int& Xcurr, int& Ycurr);

        void Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::vector<Pixel_Info>& Pix_info);

        void Reset(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);

        void Reset_Fast(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);

        void Reset_THnSparse(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);

        void reset_th1_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);
        void reset_thnsparse_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);
        void reset_thnsparse_noise_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);
        void reset_map_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);

    };

}
#endif