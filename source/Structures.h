#ifndef QPIXSTRUCTURES_H_
#define QPIXSTRUCTURES_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <math.h>

namespace Qpix
{


    struct ELECTRON 
    {
        int    Pix_ID;
        double   time;
        int    Trk_ID;
    };

    struct ION : public ELECTRON
    {
        // ionization electrons created from the hits that we're going to group into a single image
        double x, y, z, t;
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

        // should only be querried in Reset_Fast, calculating the amount
        // of time that has occured since the last time electrons have 
        // been pulled off of the time vector
        double GetDriftTime() {
            if(time.size() == 0) 
                return 0; 
            else{
                double prevtime = drift_start;
                drift_start = time.back();
                return time.back() - prevtime;
            } 
        }
        double GetDriftStart() const {return drift_start;};

        std::vector<double>  RESET = std::vector<double>();
        std::vector<std::vector<int>> RESET_TRUTH_ID = std::vector<std::vector<int>>();
        std::vector<std::vector<int>> RESET_TRUTH_W = std::vector<std::vector<int>>();

        private:
            double drift_start = 0;
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
        double Wvalue = 23.6; // in eV
        double E_vel = 164800.0; // cm/s
        double DiffusionL = 6.8223  ;  //cm**2/s
        double DiffusionT = 13.1586 ; //cm**2/s
        double Life_Time = 0.1; // in s

        // Read out plane size in cm
        double Readout_Dim = 100;
        double Pix_Size = 0.4;

        // Number of electrons for reset
        int Reset = 6250;

        // time in ns
        double Sample_time = 1/30e6;// in s 
        double Buffer_time = 1; // in s 
        double Dead_time = 0; // in s 
        double Charge_loss = false;
        double Recombination = true;
    };

    void print_Qpix_Paramaters(Qpix_Paramaters * Qpix_params);
    
    void Get_Frequencies(std::vector<int> vec, std::vector<int>& freq, std::vector<int>& weig );
    
    bool Electron_Pix_Sort(ELECTRON one, ELECTRON two);

    bool Pixel_Time_Sorter(Qpix::ELECTRON const& lhs, Qpix::ELECTRON const& rhs);

}
#endif