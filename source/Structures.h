#ifndef QPIXSTRUCTURES_H_
#define QPIXSTRUCTURES_H_

#include <iostream>
#include <string>
#include <vector>

namespace Qpix
{

    struct ELECTRON 
    {
        int    Pix_ID;
        double   time;
        int    Trk_ID;
    };



    struct Pixel_Info 
    {
        int X_Pix;
        int Y_Pix;
        int ID;
        std::vector<double>  time;
        std::vector<int> Trk_ID;

        std::vector<double>  RESET;
        std::vector<double>  TSLR;
        std::vector<std::vector<int>> RESET_TRUTH_ID;
        std::vector<std::vector<int>> RESET_TRUTH_W;
    };



    template<typename T>
    std::vector<T> slice(std::vector<T> const &v, int m, int n)
    {
        auto first = v.cbegin() + m;
        auto last = v.cbegin() + n + 1;

        std::vector<T> vec(first, last);
        return vec;
    }


    struct Qpix_Paramaters 
    {
        double Wvalue;
        double E_vel;
        double DiffusionL;
        double DiffusionT;
        double Life_Time;
        double Readout_Dim;
        double Pix_Size;
        int Reset;
        double Sample_time;
        double Buffer_time;
        double Dead_time;
        bool Charge_loss;
        bool Recombination;
        bool Noise;
        bool TimeWindow;
        double Sampling;
    };

    void set_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);

    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);
    
    void Get_Frequencies(std::vector<int> vec, std::vector<int>& freq, std::vector<int>& weig );
    
    bool Electron_Pix_Sort(ELECTRON one, ELECTRON two);

    bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs);

}
#endif
