#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <set>

// sorting things
#include <algorithm>
#include <numeric>

#include "Random.h"
#include "Structures.h"
#include "PixelResponse.h"


namespace Qpix
{
    // Decodes the pixel ID into x and y pix
    void Pixel_Functions::ID_Decoder(int const& ID, int& Xcurr, int& Ycurr)
    {
        double PixID = ID/10000.0, fractpart, intpart;
        fractpart = modf (PixID , &intpart);
        Xcurr = (int) round(intpart);
        Ycurr = (int) round(fractpart*10000); 
        return;
    }//ID_Decoder


    // sorts the electrons in to a pixel structure 
    void Pixel_Functions::Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::vector<Pixel_Info>& Pix_info)
    {
        int Event_Length = hit_e.size();
        
        std::vector<int> NewID_Index;
        int newID = 0;

        // this loop finds the index where the pixels change
        for (int i=0; i<Event_Length ; i++)
        {
            if ( newID != hit_e[i].Pix_ID )
            {
                NewID_Index.push_back( i );
                newID = hit_e[i].Pix_ID;
            }
        }
        NewID_Index.push_back( hit_e.size() );

        // Now loop though thoues to make a pixel opject that is X, Y and a vect of times
        int N_Index = NewID_Index.size() - 1;
        for (int i=0; i<N_Index ; i++)
        {
            std::vector<Qpix::ELECTRON> sub_vec = slice(hit_e, NewID_Index[i], NewID_Index[i+1] -1 );
            std::sort( sub_vec.begin(), sub_vec.end(), &Pixel_Time_Sorter );

            std::vector<int> tmp_trk_id;
            std::vector<double> tmp_time;
            for (int j=0; j<sub_vec.size() ; j++) 
            { 
                tmp_time.push_back( sub_vec[j].time ); 
                tmp_trk_id.push_back( sub_vec[j].Trk_ID ); 
            }

            int Pix_Xloc, Pix_Yloc ;
            ID_Decoder(sub_vec[0].Pix_ID, Pix_Xloc, Pix_Yloc);
            Pix_info.push_back(Pixel_Info());
            Pix_info[i].ID      = sub_vec[0].Pix_ID;
            Pix_info[i].X_Pix   = Pix_Xloc;
            Pix_info[i].Y_Pix   = Pix_Yloc;
            Pix_info[i].time    = tmp_time;
            Pix_info[i].Trk_ID  = tmp_trk_id;
        }
        return;
    }//Pixelize_Event


    // determine which IDs had electrons collide with them, and look at those times
    std::set<int> Pixel_Functions::Pixelize_Event(std::vector<Qpix::ELECTRON>& hit_e, std::map<int, Pixel_Info>& mPix_info)
    {
        std::set<int> pixel_ids;

        // go through all of the hits in the electron cloud
        for(auto hit : hit_e){
            pixel_ids.insert(hit.Pix_ID);
            mPix_info[hit.Pix_ID].time.push_back(hit.time);
            mPix_info[hit.Pix_ID].Trk_ID.push_back(hit.Trk_ID);
        }

        return pixel_ids;
    }

    // function performs the resets 
    void Pixel_Functions::Reset(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        // geting the size of the vectors for looping
        int Pixels_Hit_Len = Pix_info.size();
        int Noise_Vector_Size = Gaussian_Noise.size();
        int Noise_index = 0;

        // loop over each pixel that was hit
        for (int i = 0; i < Pixels_Hit_Len; i++)
        {
            // for truth matching 
            std::vector<int> trk_id_holder;
            std::vector<std::vector<int>> RESET_TRUTH_ID;
            std::vector<std::vector<int>> RESET_TRUTH_W;

            // seting up some parameters
            int charge = 0;
            int pix_size = Pix_info[i].time.size();
            int pix_dex = 0;
            double current_time = 0;
            double pix_time = Pix_info[i].time[pix_dex];
            std::vector<double>  RESET;
            double tslr_ = 0;
            std::vector<double>  TSLR;

            // skip if it wont reset
            if (pix_size < (Qpix_params->Reset)*0.5){continue;}

            // for each pixel loop through the buffer time
            // while (current_time <= End_Time)
            while (current_time <= Qpix_params->Buffer_time)
            {
                // setting the "time"
                current_time += Qpix_params->Sample_time;

                // adding noise from the noise vector
                charge += Gaussian_Noise[Noise_index];
                Noise_index += 1;
                if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

                // main loop to add electrons to the counter
                if ( current_time > pix_time && pix_dex < pix_size)
                {   // this adds the electrons that are in the step
                    while( current_time > pix_time )
                    {
                        trk_id_holder.push_back(Pix_info[i].Trk_ID[pix_dex]);
                        charge += 1;
                        pix_dex += 1;
                        if (pix_dex >= pix_size){break; }
                        pix_time = Pix_info[i].time[pix_dex];
                    }
                }

                // this is the reset 
                if ( charge >= Qpix_params->Reset )
                {
                    std::vector<int> trk_TrkIDs_holder;
                    std::vector<int> trk_weight_holder;
                    Get_Frequencies(trk_id_holder, trk_TrkIDs_holder, trk_weight_holder);
                    RESET_TRUTH_ID.push_back(trk_TrkIDs_holder);
                    RESET_TRUTH_W.push_back(trk_weight_holder);

                    
                    TSLR.push_back(current_time - tslr_);
                    tslr_ = current_time;

                    RESET.push_back( current_time );
                    charge -= Qpix_params->Reset;

                    if (charge < Qpix_params->Reset) { trk_id_holder.clear(); }

                    // this will keep the charge in the loop above
                    // just offsets the reset by the dead time
                    current_time += Qpix_params->Dead_time;

                    // condition for charge loss
                    // just the main loop without the charge
                    if (Qpix_params->Charge_loss)
                    {
                        while( current_time > pix_time )
                        {
                            pix_dex += 1;
                            // if (pix_dex < pix_size){ pix_time = Pix_info[i].time[pix_dex]; }

                            if (pix_dex >= pix_size){break; }
                            pix_time = Pix_info[i].time[pix_dex];
                        }
                    }
                }
            }
            // add it to the pixel info
            Pix_info[i].RESET = RESET;
            Pix_info[i].TSLR  = TSLR;
            Pix_info[i].RESET_TRUTH_ID  = RESET_TRUTH_ID;
            Pix_info[i].RESET_TRUTH_W   = RESET_TRUTH_W;
        }

        return ;
    }// Reset


    // function performs the resets 
    void Pixel_Functions::Reset_Fast(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        // time window before and after event
        double Window = 1e-6;

        // geting the size of the vectors for looping
        int Pixels_Hit_Len = Pix_info.size();
        int Noise_Vector_Size = Gaussian_Noise.size();
        int Noise_index = 0;

        // loop over each pixel that was hit
        for (int i = 0; i < Pixels_Hit_Len; i++)
        {
            // for truth matching 
            std::vector<int> trk_id_holder;
            std::vector<std::vector<int>> RESET_TRUTH_ID;
            std::vector<std::vector<int>> RESET_TRUTH_W;

            // seting up some parameters
            int charge = 0;
            int pix_size = Pix_info[i].time.size();
            int pix_dex = 0;
            // int current_time = 0;
            double pix_time = Pix_info[i].time[pix_dex];

            double current_time = pix_time - Window;
            if (current_time < 0){current_time = 0;}

            double End_Time = Pix_info[i].time[pix_size-1] + Window;

            // 100 atto amps is 625 electrons a second
            // approximate the leakage charge given "curretn_time"
            // charge = 625/1e9 * current_time;
            // mcharge = (int)ceil(625 * current_time);
            // Make sure it dose not start with a bunch of resets
            // while ( charge >= Qpix_params->Reset ){ charge -= Qpix_params->Reset; }


            std::vector<double>  RESET;
            double tslr_ = 0;
            std::vector<double>  TSLR;

            // skip if it wont reset
            if (pix_size < (Qpix_params->Reset)*0.5){continue;}

            // for each pixel loop through the buffer time
            while (current_time <= End_Time)
            {
                // setting the "time"
                current_time += Qpix_params->Sample_time;

                // adding noise from the noise vector
                charge += Gaussian_Noise[Noise_index];
                Noise_index += 1;
                if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

                // main loop to add electrons to the counter
                if ( current_time > pix_time && pix_dex < pix_size)
                {   // this adds the electrons that are in the step
                    while( current_time > pix_time )
                    {
                        trk_id_holder.push_back(Pix_info[i].Trk_ID[pix_dex]);
                        charge += 1;
                        pix_dex += 1;
                        if (pix_dex >= pix_size){break; }
                        pix_time = Pix_info[i].time[pix_dex];
                    }
                }

                // this is the reset 
                if ( charge >= Qpix_params->Reset )
                {
                    std::vector<int> trk_TrkIDs_holder;
                    std::vector<int> trk_weight_holder;
                    Get_Frequencies(trk_id_holder, trk_TrkIDs_holder, trk_weight_holder);
                    RESET_TRUTH_ID.push_back(trk_TrkIDs_holder);
                    RESET_TRUTH_W.push_back(trk_weight_holder);


                    TSLR.push_back(current_time - tslr_);
                    tslr_ = current_time;

                    RESET.push_back( current_time );
                    charge -= Qpix_params->Reset;

                    if (charge < Qpix_params->Reset) { trk_id_holder.clear(); }

                    // this will keep the charge in the loop above
                    // just offsets the reset by the dead time
                    current_time += Qpix_params->Dead_time;

                    // condition for charge loss
                    // just the main loop without the charge
                    if (Qpix_params->Charge_loss)
                    {
                        while( current_time > pix_time )
                        {
                            pix_dex += 1;
                            // if (pix_dex < pix_size){ pix_time = Pix_info[i].time[pix_dex]; }

                            if (pix_dex >= pix_size){break; }
                            pix_time = Pix_info[i].time[pix_dex];
                        }
                    }
                }
            }
            // add it to the pixel info
            Pix_info[i].RESET = RESET;
            Pix_info[i].TSLR  = TSLR;
            Pix_info[i].RESET_TRUTH_ID  = RESET_TRUTH_ID;
            Pix_info[i].RESET_TRUTH_W   = RESET_TRUTH_W;
        }

        return ;
    }// Reset

    // reset fast overload to use pixel map
    void Pixel_Functions::Reset_Fast(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, 
                            const std::set<int>& mPixIds, std::map<int, Pixel_Info>& mPix_info)
    {
        // time window before and after event
        double Window = 1e-6;

        // look at every pixel that was hit
        for(auto hit_id : mPixIds)
        {
            auto hit_pixel = mPix_info[hit_id];
            mPix_info[hit_id].RESET_TRUTH_ID.clear();
            mPix_info[hit_id].RESET_TRUTH_W.clear();
            mPix_info[hit_id].RESET.clear();
            mPix_info[hit_id].TSLR.clear();

            // skip if won't reset
            const int& nElectrons = hit_pixel.time.size();
            if(nElectrons < Qpix_params->Reset*0.5) continue;

            // sort through the electrons using their timing as the indices we care about
            std::vector<int> elec_index (nElectrons);
            std::iota(elec_index.begin(),elec_index.end(),0);
            std::stable_sort(elec_index.begin(), elec_index.end(), [&hit_pixel](size_t i1, size_t i2) 
                            {return hit_pixel.time[i1] <hit_pixel.time[i2];});

            // for truth matching 
            std::vector<std::vector<int>> RESET_TRUTH_ID;
            std::vector<std::vector<int>> RESET_TRUTH_W;
            std::vector<double>  RESET;
            double tslr_ = 0;
            std::vector<double>  TSLR;

            // build up the charge for each reset now
            double charge = 0;
            double curr_time = hit_pixel.time[elec_index.front()];
            double stop_time = hit_pixel.time[elec_index.back()];
            int i=0, j=0, lastResetElectron=0;
            while(curr_time < stop_time)
            {
                curr_time += Qpix_params->Sample_time;
                charge += Gaussian_Noise[i++];
                if(i > Gaussian_Noise.size()) i = 0;

                // check to pop any electrons
                if(curr_time > hit_pixel.time[elec_index[j]] && j < nElectrons)
                    while(curr_time > hit_pixel.time[elec_index[j++]] && j < nElectrons)
                        charge += 1;

                // reset occurs here
                if(charge >= Qpix_params->Reset)
                {
                    std::vector<int> trk_id_holder;
                    for(int i=lastResetElectron; i<j; ++i)
                        trk_id_holder.push_back(hit_pixel.Trk_ID[elec_index[i]]);

                    // calculation information
                    std::vector<int> trk_TrkIDs_holder;
                    std::vector<int> trk_weight_holder;
                    Get_Frequencies(trk_id_holder, trk_TrkIDs_holder, trk_weight_holder);
                    mPix_info[hit_id].RESET_TRUTH_ID.push_back(trk_TrkIDs_holder);
                    mPix_info[hit_id].RESET_TRUTH_W.push_back(trk_weight_holder);

                    // which electron caused this reset
                    lastResetElectron = j;

                    mPix_info[hit_id].RESET.push_back(curr_time);
                    mPix_info[hit_id].TSLR.push_back(curr_time - tslr_);
                    tslr_ = curr_time;

                    charge -= Qpix_params->Reset;
                    if (charge < Qpix_params->Reset) trk_id_holder.clear();

                    // this will keep the charge in the loop above
                    // just offsets the reset by the dead time
                    curr_time += Qpix_params->Dead_time;

                    // condition for charge loss
                    // just the main loop without the charge
                    if (Qpix_params->Charge_loss)
                        while(curr_time > hit_pixel.time[elec_index[j++]] && j < nElectrons)
                            if (j >= nElectrons){break;}

                }

            }

            // save the charge that we didn't use
            std::vector<int> hit_ids(elec_index.size()-lastResetElectron);
            std::vector<double> hit_times(elec_index.size()-lastResetElectron);
            for(auto i=elec_index.begin()+lastResetElectron; i<elec_index.end(); ++i)
            {
                hit_times.push_back(hit_pixel.time.at(*i));
                hit_ids.push_back(hit_pixel.Trk_ID.at(*i));
            }
            mPix_info[hit_id].time = hit_times;
            mPix_info[hit_id].Trk_ID = hit_ids;
        }

    }
}
