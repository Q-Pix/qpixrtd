#ifndef QPIXSTRUCTURES_H_
#define QPIXSTRUCTURES_H_

#include <iostream>
#include <string>
#include <vector>


namespace Qpix
{

    struct ELECTRON 
    {
        int  Pix_ID;
        int  time;
    };

    struct Pixel_Info 
    {
        int  X_Pix;
        int  Y_Pix;
        int ID;
        std::vector<int>  time;
        std::vector<int>  RESET;
        std::vector<int>  TSLR;
    };

    template<typename T>
    std::vector<T> slice(std::vector<T> const &v, int m, int n)
    {
        auto first = v.cbegin() + m;
        auto last = v.cbegin() + n + 1;

        std::vector<T> vec(first, last);
        return vec;
    }

    bool Electron_Pix_Sort(ELECTRON one, ELECTRON two);

    bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs);

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
        int Sample_time;
        int Buffer_time;
        int Dead_time;
        bool charge_loss;
    };

    void set_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);

    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);




}



#endif