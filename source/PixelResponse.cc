#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>
#include "Qpix/ElectronHandler.h"
#include "Qpix/Random.h"

namespace Qpix
{

    
    std::vector<std::vector<int>> Make_Readout_plane(int Readout_Dim, int Pix_Size, int Reset)
    {
        int N_Pix = Readout_Dim/Pix_Size;
        // check if the pixel is a whole number
        if( N_Pix*Pix_Size == Readout_Dim )
        {
            //std::cout << "Making a " << N_Pix << " by " << N_Pix << " readout" << std::endl;
            //std::cout << "with a " << Pix_Size << " mm pixel pitch " << std::endl;
        }
        else
        {
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            std::cout << "you have failed" << std::endl;
            std::cout << "readout and pixel dimensions do not match" << std::endl;
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<std::vector<int>> readout(N_Pix, std::vector<int>(N_Pix));
        for (int i = 0; i < N_Pix; i++)
            for (int j = 0; j < N_Pix; j++)
                readout[i][j] = RandomUniform()*Reset;
        return readout;
    }

    std::vector<std::vector<int>> Make_True_Readout_plane(int Readout_Dim, int Pix_Size, int Reset)
    {
        int N_Pix = Readout_Dim/Pix_Size;
        // check if the pixel is a whole number
        if( N_Pix*Pix_Size == Readout_Dim )
        {
            //std::cout << "Making a " << N_Pix << " by " << N_Pix << " readout" << std::endl;
            //std::cout << "with a " << Pix_Size << " mm pixel pitch " << std::endl;
        }
        else
        {
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            std::cout << "you have failed" << std::endl;
            std::cout << "readout and pixel dimensions do not match" << std::endl;
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<std::vector<int>> readout(N_Pix, std::vector<int>(N_Pix));
        for (int i = 0; i < N_Pix; i++)
            for (int j = 0; j < N_Pix; j++)
                readout[i][j] = 0;
        return readout;
    }


    std::vector<std::vector<int>> Find_Unique_Pixels(int Pix_Size, int Event_Length, std::vector<HIT> Electron_Event_Vector)
    {
        std::vector<std::vector<int>> Pixels_Hit;
        std::vector<int> tmp;
        for (int i=0; i<Event_Length; i++)
        {
            int Pix_Xloc, Pix_Yloc;
            Pix_Xloc = (int) ceil(Electron_Event_Vector[i].x_pos/Pix_Size);
            Pix_Yloc = (int) ceil(Electron_Event_Vector[i].y_pos/Pix_Size);
            tmp.push_back((int)(Pix_Xloc*10000+Pix_Yloc));
        }
        
        std::sort( tmp.begin(), tmp.end() );
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        
        for (int i=0; i<tmp.size(); i++)
        {
            double PixID = tmp[i]/10000.0, fractpart, intpart;
            fractpart = modf (PixID , &intpart);
            std::vector<int> tmp = {(int)(intpart), (int)(fractpart*10000)};
            Pixels_Hit.push_back( tmp );
        }
        return Pixels_Hit;
    }
        


    std::vector<std::vector<double>> Make_Reset_Response(int Reset, int Pix_Size, double E_vel, int Event_Length, 
                                                        int Pixels_Hit_Len, int Noise_Vector_Size, int Start_Time, int End_Time,
                                                        std::vector<double>& Gaussian_Noise, std::vector<std::vector<int>>& Pixels_Hit,
                                                        std::vector<std::vector<int>>& data2d, std::vector<Qpix::HIT>& Electron_Event_Vector)
    {
        std::vector<std::vector<double>> RTD;
        int Noise_index = 0;
        int GlobalTime = Start_Time;

        for (int i = 0; i < Event_Length; i++)
        {
            int Pix_Xloc, Pix_Yloc;
            Pix_Xloc = (int) ceil(Electron_Event_Vector[i].x_pos/Pix_Size);
            Pix_Yloc = (int) ceil(Electron_Event_Vector[i].y_pos/Pix_Size);
            double Pix_time = Electron_Event_Vector[i].z_pos/E_vel;

            while (GlobalTime < Pix_time)
            {
                for (int curr = 0; curr < Pixels_Hit_Len; curr++) 
                { 
                    int X_curr = Pixels_Hit[curr][0];
                    int Y_curr = Pixels_Hit[curr][1];
                    data2d[X_curr][Y_curr]+=Gaussian_Noise[Noise_index];
                    Noise_index += 1;
                    if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}
                    if (data2d[X_curr][Y_curr] >= Reset)
                    {
                    data2d[X_curr][Y_curr] = 0;
                    std::vector<double> tmp = {(double)X_curr, (double)Y_curr, (double)GlobalTime};
                    RTD.push_back(tmp);
                    //std::cout << "before" << std::endl;
                    //std::cout << (double)X_curr << " " << (double)Y_curr << " " << (double)GlobalTime << std::endl;
                    }
                } 
                GlobalTime+=1;
            }

            data2d[Pix_Xloc][Pix_Yloc]+=1;
            if (data2d[Pix_Xloc][Pix_Yloc] >= Reset)
            {
                data2d[Pix_Xloc][Pix_Yloc] = 0;
                std::vector<double> tmp = {(double)Pix_Xloc, (double)Pix_Yloc, Pix_time};
                RTD.push_back(tmp);
                //std::cout << "during" << std::endl;
                //std::cout << (double)Pix_Xloc << " " << (double)Pix_Yloc << " " << Pix_time << std::endl;
            }

        }

        while (GlobalTime < End_Time)
        {
            for (int curr = 0; curr < Pixels_Hit_Len; curr++) 
            { 
                int X_curr = Pixels_Hit[curr][0];
                int Y_curr = Pixels_Hit[curr][1];
                data2d[X_curr][Y_curr]+=Gaussian_Noise[Noise_index];
                Noise_index += 1;
                if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}
                if (data2d[X_curr][Y_curr] >= Reset)
                {
                    data2d[X_curr][Y_curr] = 0;
                    std::vector<double> tmp = {(double)X_curr, (double)Y_curr, (double)GlobalTime};
                    RTD.push_back(tmp);
                    //std::cout << "after" << std::endl;
                    //std::cout << (double)X_curr << " " << (double)Y_curr << " " << (double)GlobalTime << std::endl;
                }
            } 
            GlobalTime+=1;
        }
        return RTD;

    }



    std::vector<std::vector<double>> Make_Truth_Reset_Response(int Reset, int Pix_Size, double E_vel, int Event_Length, 
                                                        std::vector<std::vector<int>>& data2d, std::vector<Qpix::HIT>& Electron_Event_Vector)
    {
        std::vector<std::vector<double>> RTD;

        for (int i = 0; i < Event_Length; i++)
        {
            int Pix_Xloc, Pix_Yloc;
            Pix_Xloc = (int) ceil(Electron_Event_Vector[i].x_pos/Pix_Size);
            Pix_Yloc = (int) ceil(Electron_Event_Vector[i].y_pos/Pix_Size);
            double Pix_time = Electron_Event_Vector[i].z_pos/E_vel;

            data2d[Pix_Xloc][Pix_Yloc]+=1;
            if (data2d[Pix_Xloc][Pix_Yloc] >= Reset)
            {
                data2d[Pix_Xloc][Pix_Yloc] = 0;
                std::vector<double> tmp = {(double)Pix_Xloc, (double)Pix_Yloc, Pix_time};
                RTD.push_back(tmp);
            }

        }
        return RTD;
    }




    void Write_Reset_Data(std::string Output_File, int eventnum, std::vector<std::vector<double>>& RTD)
    {
        int RTD_len = RTD.size();
        std::ofstream data_out;
        data_out.open (Output_File , std::ios::ate);

        for (int i = 0; i < RTD_len; i++)
        {
            //if (RTD[i][3]<0.1){std::cout<<"DT too small"<<std::endl;}
            data_out << eventnum << ' ' << RTD[i][0] << ' ' << RTD[i][1] << ' ' << RTD[i][2] << std::endl;
        }
        data_out.close();
        data_out.clear();
    }


    /* void Write_Reset_Data(std::string Output_File, int eventnum, int Pixels_Hit_Len, std::vector<std::vector<int>>& Pixels_Hit, std::vector<std::vector<double>>& RTD)
    {
        int RTD_len = RTD.size();
        std::ofstream data_out;
        data_out.open (Output_File , std::ios::ate);

        for (int curr = 0; curr < Pixels_Hit_Len; curr++) 
        { 
            int X_curr = Pixels_Hit[curr][0];
            int Y_curr = Pixels_Hit[curr][1];

            double Delta_T = 0;
            double resetold=0;
            for (int i = 0; i < RTD_len; i++)
            {
                if ( (RTD[i][0] == X_curr) && (RTD[i][1] == Y_curr) )
                {
                    double reset = RTD[i][2];
                    Delta_T = reset - resetold;
                    if (RTD[i][3]<0.1){std::cout<<"DT too small"<<std::endl;}
                    data_out << eventnum << ' ' << X_curr << ' ' << Y_curr << ' ' << RTD[i][2] << ' ' << RTD[i][3] << std::endl;
                    resetold = RTD[i][2];

                }
            }
        }
        data_out.close();
        data_out.clear();
    } */

















    std::vector<std::vector<double>> Make_Dead_Time(int Readout_Dim, int Pix_Size, int Reset)
    {
        int N_Pix = Readout_Dim/Pix_Size;        

        std::vector<std::vector<double>> readout(N_Pix, std::vector<double>(N_Pix));
        for (int i = 0; i < N_Pix; i++)
            for (int j = 0; j < N_Pix; j++)
                readout[i][j] = 0.0;
        return readout;
    }





    std::vector<std::vector<double>> Make_Reset_Response_DeadTime(int Reset, int Pix_Size, double E_vel, int Event_Length, 
                                                        int Pixels_Hit_Len, int Noise_Vector_Size, int Start_Time, int End_Time,
                                                        std::vector<double>& Gaussian_Noise, std::vector<std::vector<int>>& Pixels_Hit,
                                                        std::vector<std::vector<int>>& data2d, std::vector<Qpix::HIT>& Electron_Event_Vector,std::vector<std::vector<double>>& Dead_Time)
    {
        std::vector<std::vector<double>> RTD;
        int Noise_index = 0;
        int GlobalTime = Start_Time;

        for (int i = 0; i < Event_Length; i++)
        {
            int Pix_Xloc, Pix_Yloc;
            Pix_Xloc = (int) ceil(Electron_Event_Vector[i].x_pos/Pix_Size);
            Pix_Yloc = (int) ceil(Electron_Event_Vector[i].y_pos/Pix_Size);
            double Pix_time = Electron_Event_Vector[i].z_pos/E_vel;

            while (GlobalTime < Pix_time)
            {
                for (int curr = 0; curr < Pixels_Hit_Len; curr++) 
                { 
                    int X_curr = Pixels_Hit[curr][0];
                    int Y_curr = Pixels_Hit[curr][1];
                    data2d[X_curr][Y_curr]+=Gaussian_Noise[Noise_index];
                    Noise_index += 1;
                    if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}
                    if ( (data2d[X_curr][Y_curr] >= Reset) && (GlobalTime >= Dead_Time[X_curr][Y_curr]+0.1) )
                    {
                        double DT = GlobalTime - Dead_Time[X_curr][Y_curr];
                        Dead_Time[X_curr][Y_curr] = GlobalTime;
                        data2d[X_curr][Y_curr] = 0;
                        std::vector<double> tmp = {(double)X_curr, (double)Y_curr, (double)GlobalTime, DT};
                        RTD.push_back(tmp);
                    }
                } 
                GlobalTime+=1;
            }

            data2d[Pix_Xloc][Pix_Yloc]+=1;
            if ( (data2d[Pix_Xloc][Pix_Yloc] >= Reset) && (Pix_time >= (Dead_Time[Pix_Xloc][Pix_Yloc]+0.1)) )
            {
                //std::cout << Pix_time - Dead_Time[Pix_Xloc][Pix_Yloc] << std::endl;
                double DT = Pix_time - Dead_Time[Pix_Xloc][Pix_Yloc];
                Dead_Time[Pix_Xloc][Pix_Yloc] = Pix_time;
                data2d[Pix_Xloc][Pix_Yloc] = 0;
                std::vector<double> tmp = {(double)Pix_Xloc, (double)Pix_Yloc, Pix_time, DT};
                RTD.push_back(tmp);
            }

        }

        while (GlobalTime < End_Time)
        {
            for (int curr = 0; curr < Pixels_Hit_Len; curr++) 
            { 
                int X_curr = Pixels_Hit[curr][0];
                int Y_curr = Pixels_Hit[curr][1];
                data2d[X_curr][Y_curr]+=Gaussian_Noise[Noise_index];
                Noise_index += 1;
                if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}
                if ( (data2d[X_curr][Y_curr] >= Reset) && (GlobalTime >= Dead_Time[X_curr][Y_curr]+0.1) )
                {
                    double DT = GlobalTime - Dead_Time[X_curr][Y_curr];
                    Dead_Time[X_curr][Y_curr] = GlobalTime;
                    data2d[X_curr][Y_curr] = 0;
                    std::vector<double> tmp = {(double)X_curr, (double)Y_curr, (double)GlobalTime, DT};
                    RTD.push_back(tmp);
                }
            } 
            GlobalTime+=1;
        }
        return RTD;

    }

}