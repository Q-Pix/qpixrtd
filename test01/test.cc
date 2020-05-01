#include <iostream>
#include "Qpix/XorShift256.h"
#include "Qpix/Random.h"
#include <fstream>
#include <ctime>

//using namespace std;
//using namespace Qpix;

int main()
{
    std::ofstream myfile;    
    int Nloop = 100000;
    myfile.open ("RandomUniform.txt");
    for(int i=0;i<Nloop;++i)
    {
        //std::cout<< Qpix::RandomUniform()  <<std::endl;
        myfile << Qpix::RandomUniform() <<std::endl;
    }
    myfile.close();


    myfile.open ("RandomNormal.txt");
    double mean  = 10;
    double sigma = 2;
    for(int i=0;i<Nloop;++i)
    {
        //std::cout<< Qpix::RandomNormal(mean,sigma)  <<std::endl;
        myfile << Qpix::RandomNormal(mean,sigma) <<std::endl;
    }
    myfile.close();

    myfile.open ("RandomPoisson.txt");
    for(int i=0;i<Nloop;++i)
    {
        //std::cout<< Qpix::RandomNormal(mean,sigma)  <<std::endl;
        myfile << Qpix::RandomPoisson(mean) <<std::endl;
    }
    myfile.close();

    clock_t time_req;

    time_req = clock();
    for(int i=0;i<100000000;++i){double a  = Qpix::RandomUniform();}
    time_req = clock() - time_req;
    double time = (float)time_req/CLOCKS_PER_SEC;
    std::cout<< "100 000 000 random numbers (RandomUniform) took "<<time<<" Seconds"<<std::endl;
    std::cout<< "Per loop is  "<<(time/100000000)*1e9<<" nano seconds"<<std::endl;

    std::cout<< "\n "<<std::endl;

    time_req = clock();
    for(int i=0;i<100000000;++i){double a  = Qpix::RandomNormal(mean,sigma);}
    time_req = clock() - time_req;
    time = (float)time_req/CLOCKS_PER_SEC;
    std::cout<< "100 000 000 random numbers (RandomNormal) took "<<time<<" Seconds"<<std::endl;
    std::cout<< "Per loop is  "<<(time/100000000)*1e9<<" nano seconds"<<std::endl;

    std::cout<< "\n "<<std::endl;
    
    time_req = clock();
    for(int i=0;i<100000000;++i){double a  = Qpix::RandomPoisson(mean);}
    time_req = clock() - time_req;
    time = (float)time_req/CLOCKS_PER_SEC;
    std::cout<< "100 000 000 random numbers (RandomPoisson) took "<<time<<" Seconds"<<std::endl;
    std::cout<< "Per loop is  "<<(time/100000000)*1e9<<" nano seconds"<<std::endl;


    return 0;
}