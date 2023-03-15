#ifndef QPIXSTRUCTURES_H_
#define QPIXSTRUCTURES_H_


#include <iostream>
#include <string>
#include <vector>
#include <map>

namespace Qpix
{

    struct ELECTRON 
    {
        int    Pix_ID;
        double   time;
        int    Trk_ID;
    };


    // matches with PixelResponse Decoder..
    int ID_Encoder(const int&, const int&);

    struct Pixel_Info 
    {
        Pixel_Info() = default;
        Pixel_Info(const short& x, const short& y) : X_Pix(x), Y_Pix(y), ID(Qpix::ID_Encoder(x, y)) {};
        short X_Pix;
        short Y_Pix;
        int ID;
        std::vector<double>  time = std::vector<double>();
        std::vector<short> Trk_ID = std::vector<short>();

        // keep track of how much charge we've added
        float charge = 0;
        short nElectrons = 0;
        std::map<u_int16_t, short> mPids;
        double tslr=0;

        std::vector<double>  RESET = std::vector<double>();
        std::vector<std::vector<int>> RESET_TRUTH_ID = std::vector<std::vector<int>>();
        std::vector<std::vector<int>> RESET_TRUTH_W = std::vector<std::vector<int>>();
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
    };

    void set_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);

    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);
    
    void Get_Frequencies(std::vector<int> vec, std::vector<int>& freq, std::vector<int>& weig );
    
    bool Electron_Pix_Sort(ELECTRON one, ELECTRON two);

    bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs);

}
#endif