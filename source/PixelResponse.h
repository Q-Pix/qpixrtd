#ifndef PIXELRESPONSE_H_
#define PIXELRESPONSE_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
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

        // overload with unordered_map to handle all searching
        std::set<int> Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::unordered_map<int, Pixel_Info>& mPix_info);
        std::set<int> Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::unordered_map<int, Pixel_Info>& mPix_info, std::set<int> good_pixels);

        void Reset(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);
        
        void Reset_Fast(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info);

        void Reset_Fast(Qpix::Qpix_Paramaters * Qpix_params, const std::set<int>& mPixIds, std::unordered_map<int, Pixel_Info>& mPix_info);
    };

    // helper functions

    // take the number of samples we're skipping and see if there are any
    // electrons in this window, pass back the time of the added electron, if so
    bool DriftCurrentElectrons(const int&, double&);

}
#endif