#ifndef RANDOM_H_
#define RANDOM_H_

#include "XorShift256.h"
#include <vector>


namespace Qpix
{
    double RandomUniform();

    double RandomNormal(double mean, double sigma) ;

    double lngamma(const double xx) ;
    int RandomPoisson(const double mean) ;

    std::vector<double> Make_Gaussian_Noise(double sigma, int Noise_Vector_Size);

}

#endif