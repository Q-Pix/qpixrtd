#include <math.h>
#include <vector>

#include "Qpix/Xoshiro_Full.h"

namespace Qpix
{

    XoshiroCpp::Xoshiro256Plus rng(XoshiroCpp::DefaultSeed);
    double RandomUniform() 
    {
        double out = XoshiroCpp::DoubleFromBits(rng());
        return out; 
    }

    void Random_Set_Seed(std::uint64_t Seed)
    {
        rng = XoshiroCpp::Xoshiro256Plus(Seed);
    }


    double RandomNormal(double m, double s)	/* normal random variate generator */
    {				        /* mean m, standard deviation s */
        double x1, x2, w, y1;
        static double y2;
        static int use_last = 0;

        if (use_last)		        /* use value from previous call */
        {
            y1 = y2;
            use_last = 0;
        }
        else
        {
            do {
                x1 = 2.0 * RandomUniform()  - 1.0;
                x2 = 2.0 * RandomUniform()  - 1.0;
                w = x1 * x1 + x2 * x2;
            } while ( w >= 1.0 );

            w = sqrt( (-2.0 * log( w ) ) / w );
            y1 = x1 * w;
            y2 = x2 * w;
            use_last = 1;
        }

        return( m + y1 * s );
    }



    double lngamma(const double xx) 
    {   // Implementation from CLHEP.
        constexpr double cof[6] = {76.18009172947146,-86.50532032941677,
                                    24.01409824083091, -1.231739572450155,
                                    0.1208650973866179e-2, -0.5395239384953e-5};
        double x = xx - 1.0;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j <= 5; j++) {
            x += 1.0;
            ser += cof[j] / x;
        }
        return -tmp + log(2.5066282746310005 * ser);
    }

    int RandomPoisson(const double mean) 
    {   // Implementation from CLHEP (RandPoisson) and ROOT.
        if (mean <= 0) return 0;
        if (mean < 25) {
            const double expmean = exp(-mean);
            double pir = 1.;
            int n = -1;
            while (1) {
            n++;
            pir *= RandomUniform();
            if (pir <= expmean) break;
            }
            return n;
        } else {
            // Use inversion method for large values.
            const double sq = sqrt(2. * mean);
            const double alxm = log(mean);
            const double g = mean * alxm - lngamma(mean + 1.);
            double y = 0., t = 0.;
            double em = -1.;
            do {
            do {
                y = tan(M_PI * RandomUniform());
                em = sq * y + mean;
            } while (em < 0.0);
            em = floor(em);
            t = 0.9 * (1. + y * y) * exp(em * alxm - lngamma(em + 1.) - g);
            } while (RandomUniform() > t);
            return static_cast<int>(em);
        }
        
    }


    std::vector<double> Make_Gaussian_Noise(double sigma, int Noise_Vector_Size)
    {
    std::vector<double> Noise;
    for (int i = 0; i < Noise_Vector_Size; i++)
    {  // should be 200 and 20 per mus making it a gaussian of 20 every 100ns
        int Leakage_Current, Electronics;
        //if ( i % 10 == 0 ){Electronics = (int)Qpix::RandomNormal(0, sigma);}
        //else{Electronics=0; }
        Electronics = (int) round(Qpix::RandomNormal(0, sigma));

        // 100 atto amps is 625 electrons a second
        if (RandomUniform() < 625/1e8 ){Leakage_Current = 1;}
        else {Leakage_Current = 0;}

        Noise.push_back( Electronics + Leakage_Current );
    }
    return Noise;
    }
    

}