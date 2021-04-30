#ifndef RANDOM_H_
#define RANDOM_H_

#include <vector>
#include <cstdint>

namespace Qpix
{

    double RandomUniform();
    
    void Random_Set_Seed(std::uint64_t Seed);

    double RandomNormal(double mean, double sigma) ;

    double lngamma(const double xx) ;

    int RandomPoisson(const double mean) ;

    std::vector<double> Make_Gaussian_Noise(double sigma, int Noise_Vector_Size);

}

#endif
