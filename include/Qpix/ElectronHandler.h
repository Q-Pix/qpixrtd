#ifndef ELECTRONHANDLER_H_
#define ELECTRONHANDLER_H_

#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

#include "Qpix/Random.h"



namespace Qpix
{

    struct HIT 
    {
        double  x_pos;
        double  y_pos;
        double  z_pos;
    }; 

    //bool sortcol( const std::vector<double>& v1,const std::vector<double>& v2 );

    //std::vector< std::vector<double>> DiffuserTest(double Wvalue,double E_vel,double DiffusionL,double DiffusionT,
    //                                                                    const std::vector<std::vector<double>>& eventt2);


    std::vector<HIT> Diffuser(double Wvalue,double E_vel, double LifeTime,double DiffusionL,double DiffusionT,
                                    const std::vector<std::vector<double>>& eventt2);  

                                                                                                                                        
    bool compare(HIT one, HIT two);
}

#endif