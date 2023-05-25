#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <math.h>

#include <boost/format.hpp>
#include <boost/histogram.hpp>
#include "TH1F.h"
#include "THnSparse.h"

#include "Random.h"
#include "Structures.h"
#include "PixelResponse.h"

namespace bh = boost::histogram;

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
            std::cout << "pixel size: " << pix_size << std::endl;

            // for each pixel loop through the buffer time
            // while (current_time <= End_Time)
            while (current_time <= Qpix_params->Buffer_time)
            {
                // setting the "time"
                current_time += Qpix_params->Sample_time;

                // adding noise from the noise vector
                // if (Gaussian_Noise[Noise_index] != 0) charge += Gaussian_Noise[Noise_index];
                charge += Gaussian_Noise[Noise_index];
                if (Gaussian_Noise[Noise_index] != 0) std::cout << Noise_index << ", " << current_time << ", " << charge << std::endl;
                Noise_index += 1;
                if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}
                // if (Noise_index >= Noise_Vector_Size)
                // {
                //     std::cout << "Noise_index, Noise_Vector_Size: " << Noise_index << ", " << Noise_Vector_Size << std::endl;
                //     Noise_index = 0;
                // }

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
                        // std::cout << "current_time: " << current_time << "; charge: " << charge << std::endl;
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
                    double tslr_tmp = current_time - tslr_;
                    tslr_ = current_time;

                    std::cout << charge << ", ";
                    RESET.push_back( current_time );
                    charge -= Qpix_params->Reset;
                    std::cout << charge << ", ";
                    std::cout << current_time << ", ";
                    std::cout << tslr_tmp << std::endl;

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
            // std::cout << "Noise_index: " << Noise_index << std::endl;
            // std::cout << "time, Noise_index: " << current_time << ", " << Noise_index << std::endl;
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
                    double tslr_tmp = current_time - tslr_;
                    tslr_ = current_time;

                    RESET.push_back( current_time );
                    std::cout << charge << ", ";
                    charge -= Qpix_params->Reset;
                    std::cout << charge << ", ";
                    std::cout << current_time << ", ";
                    std::cout << tslr_tmp << std::endl;

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
    }// Reset_Fast

    void Pixel_Functions::reset_th1_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        int Noise_Vector_Size = Gaussian_Noise.size();
        int Noise_index = 0;

        auto h1 = TH1F("h1", "h1", Qpix_params->Buffer_time/Qpix_params->Sample_time, 0.0, Qpix_params->Buffer_time);

        for (auto& pixel : Pix_info)
        {
            double tslr_ = 0;
            int charge = 0;

            std::vector< double > reset;
            std::vector< double > tslr;

            std::vector< int > trk_id_holder;
            std::vector< std::vector< int > > RESET_TRUTH_ID;
            std::vector< std::vector< int > > RESET_TRUTH_W;

            // skip if it won't reset
            if (pixel.time.size() < (Qpix_params->Reset)*0.5) continue;

            for (int idx = 0; idx < pixel.time.size(); ++idx)
            {
                h1.Fill(pixel.time.at(idx));
            }

            for (int bin_idx = 0; bin_idx < h1.GetNbinsX(); ++bin_idx)
            {
                // double current_time = h1.GetBinLowEdge(bin_idx+1);
                double current_time = h1.GetXaxis()->GetBinUpEdge(bin_idx);
                if (current_time > Qpix_params->Buffer_time) break;

                // adding noise from the noise vector
                if (Gaussian_Noise[Noise_index] > 0)
                {
                    charge += Gaussian_Noise[Noise_index];
                    // std::cout << "Gaussian_Noise[" << Noise_index << "]: " << Gaussian_Noise[Noise_index] << std::endl;
                }
                Noise_index += 1;
                if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

                charge += h1.GetBinContent(bin_idx);
                if (charge >= Qpix_params->Reset)
                {
                    std::cout << charge << ", ";
                    tslr.push_back(current_time - tslr_);
                    double tslr_tmp = current_time - tslr_;
                    tslr_ = current_time;
                    reset.push_back(current_time);
                    charge -= Qpix_params->Reset;
                    std::cout << charge << ", ";
                    std::cout << current_time << ", ";
                    std::cout << tslr_tmp << std::endl;
                }
            }

            h1.Reset("ICESM");

            pixel.RESET = reset;
            pixel.TSLR = tslr;
            pixel.RESET_TRUTH_ID = RESET_TRUTH_ID;
            pixel.RESET_TRUTH_W = RESET_TRUTH_W;

        }

        return;
    } // reset_th1_test

    void Pixel_Functions::reset_thnsparse_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        // boost::histogram
        // auto h = bh::make_histogram(bh::axis::regular<>(Qpix_params->Buffer_time/Qpix_params->Sample_time, 0.0, Qpix_params->Buffer_time, "x"));

        // TH1F histogram
        // auto h = TH1F("h", "h", Qpix_params->Buffer_time/Qpix_params->Sample_time, 0.0, Qpix_params->Buffer_time);
        // std::cout << h.GetNbinsX() << std::endl;
        // for (int idx = 0; idx < h.GetNbinsX(); ++idx)
        // {
        //     std::cout << h.GetBinLowEdge(idx) << std::endl;
        // }

        // THnSparseF histograms
        // https://root.cern/doc/master/classTHnSparse.html
        // https://root.cern/doc/master/sparsehist_8C.html
        // https://root.cern/doc/master/drawsparse_8C.html
        // https://github.com/alisw/AliPhysics/blob/master/CORRFW/AliCFGridSparse.cxx ?
        // int dimensions = 1;
        int bins[1] = { static_cast<int>(Qpix_params->Buffer_time/Qpix_params->Sample_time) };
        double xmin[1] = { 0.0 };
        double xmax[1] = { Qpix_params->Buffer_time };
        auto h = THnSparseD("h", "h", 1, bins, xmin, xmax);
        auto hproj = h.Projection(0);
        auto xaxis = hproj->GetXaxis();

        // std::cout << "bins[0]: " << bins[0] << std::endl;
        // std::cout << "xmin[0]: " << xmin[0] << std::endl;
        // std::cout << "xmax[0]: " << xmax[0] << std::endl;

        // int bins2[] = { bins[0], 100001 };
        // double xmin2[] = { xmin[0], -1 };
        // double xmax2[] = { xmax[0], 100000 };
        // auto h2 = THnSparseF("h2", "h2", 2, bins2, xmin2, xmax2);

        // int index = 0;

        for (auto& pixel : Pix_info)
        {
            double tslr_ = 0;
            int charge = 0;

            std::vector< double > reset;
            std::vector< double > tslr;

            std::vector< int > trk_id_holder;
            std::vector< std::vector< int > > RESET_TRUTH_ID;
            std::vector< std::vector< int > > RESET_TRUTH_W;

            // std::for_each(pixel.time.begin(), pixel.time.end(), std::ref(h));
            // h.reset();

            // skip if it won't reset
            if (pixel.time.size() < (Qpix_params->Reset)*0.5) continue;

            for (int idx = 0; idx < pixel.time.size(); ++idx)
            {
                // double time_array[1] = { pixel.time.at(idx) };
                // h.Fill(time_array);
                h.Fill(pixel.time.at(idx));
                // h2.Fill(pixel.time.at(idx), pixel.Trk_ID.at(idx));
            }

            // std::cout << "h.GetSparseFractionBins(): " << h.GetSparseFractionBins() << std::endl;
            // std::cout << "h.GenNbins(): " << h.GetNbins() << std::endl;

            for (int bin_idx = 0; bin_idx < h.GetNbins(); ++bin_idx)
            {
                int bin_coordinates[1];
                int counts = h.GetBinContent(bin_idx, bin_coordinates);
                double current_time = xaxis->GetBinUpEdge(bin_coordinates[0]);
                // std::cout << "bin_coodinate[0]: " << bin_coordinates[0] << std::endl;
                // double current_time = h.GetAxis(0)->GetBinLowEdge(bin_idx);
                if (current_time > Qpix_params->Buffer_time) break;
                // // int tmp[] = { 0 };
                // // std::cout << "  h.GetBinContent(" << bin_idx <<  "): " << h.GetBinContent(bin_idx, tmp) << std::endl;
                // // std::cout << "  h.GetBinContent(" << bin_idx <<  "): " << h.GetBinContent(bin_idx) << std::endl;
                // // std::cout << "  h.GetBin(" << bin_idx <<  "): " << h.GetBin(bin_idx) << std::endl;
                // // std::cout << "  h.GetAxis(0)->GetBinLowEdge(" << bin_idx <<  "): " << h.GetAxis(0)->GetBinLowEdge(bin_idx) << std::endl;
                // // std::cout << "  h2.GetAxis(0)->GetBinLowEdge(" << bin_idx <<  "): " << h2.GetAxis(0)->GetBinLowEdge(bin_idx) << std::endl;
                // // if (h.GetBinContent(bin_idx) < Qpix_params->Reset) continue;
                // std::cout << "  h.GetBinContent(" << bin_idx <<  "): " << h.GetBinContent(bin_idx) << std::endl;
                // // std::cout << "  h2.GetBinContent(" << bin_idx <<  "): " << h2.GetBinContent(bin_idx) << std::endl;
                // if (bin_idx > 0)
                // {
                //     std::cout << "  " << h.GetAxis(0)->GetBinLowEdge(bin_idx) - h.GetAxis(0)->GetBinLowEdge(bin_idx-1) << std::endl;
                // }

                // if (Gaussian_Noise[Noise_index] > 0)
                // {
                //     charge += Gaussian_Noise[Noise_index];
                //     // std::cout << "Gaussian_Noise[" << Noise_index << "]: " << Gaussian_Noise[Noise_index] << std::endl;
                // }
                // Noise_index += 1;
                // if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

                // charge += h.GetBinContent(bin_idx);
                charge += counts;
                if (charge >= Qpix_params->Reset)
                {
                    std::cout << charge << ", ";
                    tslr.push_back(current_time - tslr_);
                    double tslr_tmp = current_time - tslr_;
                    tslr_ = current_time;
                    reset.push_back(current_time);
                    charge -= Qpix_params->Reset;
                    std::cout << charge << ", ";
                    std::cout << current_time << ", ";
                    std::cout << tslr_tmp << std::endl;
                }

            }

            h.Reset("ICESM");

            // add it to the pixel info
            // Pix_info[i].RESET = RESET;
            // Pix_info[i].TSLR  = TSLR;
            // Pix_info[i].RESET_TRUTH_ID  = RESET_TRUTH_ID;
            // Pix_info[i].RESET_TRUTH_W   = RESET_TRUTH_W;
            pixel.RESET = reset;
            pixel.TSLR = tslr;
            pixel.RESET_TRUTH_ID = RESET_TRUTH_ID;
            pixel.RESET_TRUTH_W = RESET_TRUTH_W;

            // std::cout << index << std::endl;
            // index++;
        }

        delete hproj;

        return;
    } // reset_thnsparse_test

    void Pixel_Functions::reset_thnsparse_noise_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        int Noise_Vector_Size = Gaussian_Noise.size();
        int Noise_index = 0;

        // boost::histogram
        // auto h = bh::make_histogram(bh::axis::regular<>(Qpix_params->Buffer_time/Qpix_params->Sample_time, 0.0, Qpix_params->Buffer_time, "x"));

        // TH1F histogram
        // auto h = TH1F("h", "h", Qpix_params->Buffer_time/Qpix_params->Sample_time, 0.0, Qpix_params->Buffer_time);
        // std::cout << h.GetNbinsX() << std::endl;
        // for (int idx = 0; idx < h.GetNbinsX(); ++idx)
        // {
        //     std::cout << h.GetBinLowEdge(idx) << std::endl;
        // }

        // THnSparseF histograms
        // https://root.cern/doc/master/classTHnSparse.html
        // https://root.cern/doc/master/sparsehist_8C.html
        // https://root.cern/doc/master/drawsparse_8C.html
        // https://github.com/alisw/AliPhysics/blob/master/CORRFW/AliCFGridSparse.cxx ?
        // int dimensions = 1;
        int bins[1] = { static_cast<int>(Qpix_params->Buffer_time/Qpix_params->Sample_time) };
        double xmin[1] = { 0.0 };
        double xmax[1] = { Qpix_params->Buffer_time };
        auto h = THnSparseD("h", "h", 1, bins, xmin, xmax);
        auto hproj = h.Projection(0);
        auto xaxis = hproj->GetXaxis();

        // std::cout << "bins[0]: " << bins[0] << std::endl;
        // std::cout << "xmin[0]: " << xmin[0] << std::endl;
        // std::cout << "xmax[0]: " << xmax[0] << std::endl;

        // int bins2[] = { bins[0], 100001 };
        // double xmin2[] = { xmin[0], -1 };
        // double xmax2[] = { xmax[0], 100000 };
        // auto h2 = THnSparseF("h2", "h2", 2, bins2, xmin2, xmax2);

        // int index = 0;

        for (auto& pixel : Pix_info)
        {
            double tslr_ = 0;
            int charge = 0;
            // int noise = 0;
            int prev_bin_coordinate = 0;
            int prev_reset_bin_coordinate = 0;

            std::vector< double > reset;
            std::vector< double > tslr;

            std::vector< int > trk_id_holder;
            std::vector< std::vector< int > > RESET_TRUTH_ID;
            std::vector< std::vector< int > > RESET_TRUTH_W;

            // std::for_each(pixel.time.begin(), pixel.time.end(), std::ref(h));
            // h.reset();

            // skip if it won't reset
            if (pixel.time.size() < (Qpix_params->Reset)*0.5) continue;
            // if (pix_size < (Qpix_params->Reset)*0.5){continue;}
            std::cout << "pixel size: " << pixel.time.size() << std::endl;

            for (int idx = 0; idx < pixel.time.size(); ++idx)
            {
                // double time_array[1] = { pixel.time.at(idx) };
                // h.Fill(time_array);
                h.Fill(pixel.time.at(idx));
                // h2.Fill(pixel.time.at(idx), pixel.Trk_ID.at(idx));
            }

            // std::cout << "h.GetSparseFractionBins(): " << h.GetSparseFractionBins() << std::endl;
            // std::cout << "h.GenNbins(): " << h.GetNbins() << std::endl;
            // std::cout << "hproj->GetNbinsX(): " << hproj->GetNbinsX() << std::endl;

            for (int bin_idx = 0; bin_idx < h.GetNbins(); ++bin_idx)
            {
                int bin_coordinates[1];
                int counts = h.GetBinContent(bin_idx, bin_coordinates);
                double current_time = xaxis->GetBinUpEdge(bin_coordinates[0]);
                // std::cout << "bin_coodinate[0]: " << bin_coordinates[0] << std::endl;
                // double current_time = h.GetAxis(0)->GetBinLowEdge(bin_idx);
                if (current_time > Qpix_params->Buffer_time) break;
                // // int tmp[] = { 0 };
                // // std::cout << "  h.GetBinContent(" << bin_idx <<  "): " << h.GetBinContent(bin_idx, tmp) << std::endl;
                // // std::cout << "  h.GetBinContent(" << bin_idx <<  "): " << h.GetBinContent(bin_idx) << std::endl;
                // // std::cout << "  h.GetBin(" << bin_idx <<  "): " << h.GetBin(bin_idx) << std::endl;
                // // std::cout << "  h.GetAxis(0)->GetBinLowEdge(" << bin_idx <<  "): " << h.GetAxis(0)->GetBinLowEdge(bin_idx) << std::endl;
                // // std::cout << "  h2.GetAxis(0)->GetBinLowEdge(" << bin_idx <<  "): " << h2.GetAxis(0)->GetBinLowEdge(bin_idx) << std::endl;
                // // if (h.GetBinContent(bin_idx) < Qpix_params->Reset) continue;
                // std::cout << "  h.GetBinContent(" << bin_idx <<  "): " << h.GetBinContent(bin_idx) << std::endl;
                // // std::cout << "  h2.GetBinContent(" << bin_idx <<  "): " << h2.GetBinContent(bin_idx) << std::endl;
                // if (bin_idx > 0)
                // {
                //     std::cout << "  " << h.GetAxis(0)->GetBinLowEdge(bin_idx) - h.GetAxis(0)->GetBinLowEdge(bin_idx-1) << std::endl;
                // }

                // adding noise from the noise vector
                int noise_start = Noise_index;
                int noise_stop = noise_start + (bin_coordinates[0] - prev_bin_coordinate) + 1;
                // std::cout << "noise_start, noise_stop: " << noise_start << ", " << noise_stop << std::endl;

                while (Noise_index < noise_stop)
                {
                    // if (Gaussian_Noise[Noise_index] > 0) charge += Gaussian_Noise[Noise_index];
                    if (Gaussian_Noise[Noise_index] != 0) charge += Gaussian_Noise[Noise_index];
                    // charge += Gaussian_Noise[Noise_index];
                    if (Gaussian_Noise[Noise_index] != 0) std::cout << Noise_index << ", " << xaxis->GetBinUpEdge(prev_bin_coordinate) << ", " << charge << std::endl;

                    if (charge >= Qpix_params->Reset)
                    {
                        double noise_time = xaxis->GetBinUpEdge(prev_bin_coordinate);
                        // // std::cout << noise << ", ";
                        // std::cout << charge << ", ";
                        tslr.push_back(noise_time - tslr_);
                        double tslr_tmp = noise_time - tslr_;
                        tslr_ = noise_time;
                        reset.push_back(noise_time);
                        // noise -= Qpix_params->Reset;
                        // std::cout << noise << ", ";
                        charge -= Qpix_params->Reset;
                        // std::cout << charge << ", ";
                        // std::cout << noise_time << ", ";
                        // // std::cout << tslr_tmp << std::endl;
                        // std::cout << tslr_tmp << ", " << "NOISE RESET" << std::endl;
                        // prev_reset_bin_coordinate = prev_bin_coordinate;
                    }

                    prev_bin_coordinate += 1;

                    Noise_index += 1;
                    if (Noise_index >= Noise_Vector_Size)
                    {
                        // std::cout << "Noise_index, Noise_Vector_Size, noise_stop: " << Noise_index << ", " << Noise_Vector_Size << ", " << noise_stop << std::endl;
                        Noise_index = 0;
                        noise_stop -= Noise_Vector_Size;
                    }
                    // if (Noise_index >= noise_stop) std::cout << "AYY " << bin_idx << ", " << current_time << ", " << Noise_index << ", " << noise_stop << std::endl;
                }

/*

                // adding noise from the noise vector
                auto noise_start = Gaussian_Noise.begin() + Noise_index;
                auto noise_stop = noise_start + (bin_coordinates[0] - prev_bin_coordinate);
                int noise_sum = std::accumulate(noise_start, noise_stop, 0);
                // std::cout << "DEBUG: " << Noise_index << ", " << bin_coordinates[0] << " - " << prev_bin_coordinate << " = " << bin_coordinates[0]-prev_bin_coordinate <<  ", " << noise_sum << std::endl;
                if (noise_sum > 0)
                {
                    // if (noise+noise_sum < Qpix_params->Reset)
                    if (charge+noise_sum < Qpix_params->Reset)
                    {
                        // noise += noise_sum;
                        charge += noise_sum;
                        Noise_index += (bin_coordinates[0] - prev_bin_coordinate);
                    }
                    // else if (noise+noise_sum >= Qpix_params->Reset)
                    else if (charge+noise_sum >= Qpix_params->Reset)
                    {
                        // int noise_resets = (noise+noise_sum) / Qpix_params->Reset;
                        int noise_resets = (charge+noise_sum) / Qpix_params->Reset;
                        for (int noise_reset_idx = 0; noise_reset_idx < noise_resets; ++noise_reset_idx)
                        {
                            // noise += Gaussian_Noise[Noise_index];
                            charge += Gaussian_Noise[Noise_index];
                            // Noise_index += 1;
                            // prev_bin_coordinate += 1;
                            ++Noise_index;
                            if (Noise_index >= Noise_Vector_Size) Noise_index = 0;
                            ++prev_bin_coordinate;

                            // if (noise >= Qpix_params->Reset)
                            if (charge >= Qpix_params->Reset)
                            {
                                double noise_time = xaxis->GetBinUpEdge(prev_bin_coordinate);
                                // std::cout << noise << ", ";
                                std::cout << charge << ", ";
                                tslr.push_back(noise_time - tslr_);
                                double tslr_tmp = noise_time - tslr_;
                                tslr_ = noise_time;
                                reset.push_back(noise_time);
                                // noise -= Qpix_params->Reset;
                                // std::cout << noise << ", ";
                                charge -= Qpix_params->Reset;
                                std::cout << charge << ", ";
                                std::cout << noise_time << ", ";
                                std::cout << tslr_tmp << std::endl;
                                prev_reset_bin_coordinate = prev_bin_coordinate;
                            }
                        }

                        noise_start = Gaussian_Noise.begin() + Noise_index;
                        noise_stop = noise_start + (bin_coordinates[0] - prev_bin_coordinate);
                        noise_sum = std::accumulate(noise_start, noise_stop, 0);
                        noise += noise_sum;
                        Noise_index += (bin_coordinates[0] - prev_bin_coordinate);
                    }
                }
                else
                {
                    Noise_index += (bin_coordinates[0] - prev_bin_coordinate);
                }

*/

                // if (noise > 0) std::cout << "noise: " << noise << std::endl;
                // charge += noise;
                // noise = 0;

                // if (Gaussian_Noise[Noise_index] > 0)
                // {
                //     charge += Gaussian_Noise[Noise_index];
                //     // std::cout << "Gaussian_Noise[" << Noise_index << "]: " << Gaussian_Noise[Noise_index] << std::endl;
                // }
                // Noise_index += 1;
                // if (Noise_index >= Noise_Vector_Size){Noise_index = 0;}

                // charge += h.GetBinContent(bin_idx);
                charge += counts;
                // std::cout << "current_time: " << current_time << "; charge: " << charge << std::endl;
                if (charge >= Qpix_params->Reset)
                {
                    std::cout << charge << ", ";
                    tslr.push_back(current_time - tslr_);
                    double tslr_tmp = current_time - tslr_;
                    tslr_ = current_time;
                    reset.push_back(current_time);
                    charge -= Qpix_params->Reset;
                    std::cout << charge << ", ";
                    std::cout << current_time << ", ";
                    std::cout << tslr_tmp << std::endl;
                    prev_reset_bin_coordinate = bin_coordinates[0];
                    // if (bin_coordinates[0] - prev_bin_coordinate != 1)
                    // std::cout << "HELLO! " << prev_bin_coordinate << ", " << bin_coordinates[0] << ", " << bin_coordinates[0] - prev_bin_coordinate << std::endl;
                }

                // prev_bin_coordinate = bin_coordinates[0];
                prev_bin_coordinate = bin_coordinates[0]+1;

                if (bin_idx == h.GetNbins()-1)
                {
                    // h.GetNbins() - bin_coordinates[0] - 1;
                    // std::cout << "bin_idx, h.GetNbins(): " << bin_idx << ", " << h.GetNbins() << std::endl;

                    // adding noise from the noise vector
                    noise_start = Noise_index;
                    noise_stop = noise_start + (hproj->GetNbinsX() - prev_bin_coordinate) + 1;
                    // std::cout << "noise_start, noise_stop, hproj->GetNbinsX(), prev_bin_coordinate, hproj->GetNbinsX()-prev_bin_coordinate: " << noise_start << ", " << noise_stop << ", " << hproj->GetNbinsX() << ", " << prev_bin_coordinate << ", " << hproj->GetNbinsX()-prev_bin_coordinate << std::endl;

                    while (Noise_index < noise_stop)
                    {
                        // if (Gaussian_Noise[Noise_index] > 0) charge += Gaussian_Noise[Noise_index];
                        if (Gaussian_Noise[Noise_index] != 0) charge += Gaussian_Noise[Noise_index];
                        // charge += Gaussian_Noise[Noise_index];
                        if (Gaussian_Noise[Noise_index] != 0) std::cout << Noise_index << ", " << xaxis->GetBinUpEdge(prev_bin_coordinate+1) << ", " << charge << std::endl;

                        if (charge >= Qpix_params->Reset)
                        {
                            double noise_time = xaxis->GetBinUpEdge(prev_bin_coordinate+1);
                            // std::cout << noise << ", ";
                            std::cout << charge << ", ";
                            tslr.push_back(noise_time - tslr_);
                            double tslr_tmp = noise_time - tslr_;
                            tslr_ = noise_time;
                            reset.push_back(noise_time);
                            // noise -= Qpix_params->Reset;
                            // std::cout << noise << ", ";
                            charge -= Qpix_params->Reset;
                            std::cout << charge << ", ";
                            std::cout << noise_time << ", ";
                            std::cout << tslr_tmp << std::endl;
                            // std::cout << tslr_tmp << ", " << "NOISE RESET" << std::endl;
                            // prev_reset_bin_coordinate = prev_bin_coordinate;
                        }

                        prev_bin_coordinate += 1;

                        Noise_index += 1;
                        if (Noise_index >= Noise_Vector_Size)
                        {
                            // std::cout << "Noise_index, Noise_Vector_Size, noise_stop, noise_stop-Noise_Vector_Size: " << Noise_index << ", " << Noise_Vector_Size << ", " << noise_stop << ", " << noise_stop-Noise_Vector_Size << std::endl;
                            Noise_index = 0;
                            noise_stop -= Noise_Vector_Size;
                        }
                        // if (Noise_index >= noise_stop) std::cout << "AYY " << bin_idx << ", " << current_time << ", " << Noise_index << ", " << noise_stop << std::endl;
                    }
                }
            }

            // std::cout << "Noise_index: " << Noise_index << std::endl;
            // std::cout << "time, Noise_index: " << xaxis->GetBinUpEdge(prev_bin_coordinate) << ", " << Noise_index << std::endl;

            h.Reset("ICESM");

            // add it to the pixel info
            // Pix_info[i].RESET = RESET;
            // Pix_info[i].TSLR  = TSLR;
            // Pix_info[i].RESET_TRUTH_ID  = RESET_TRUTH_ID;
            // Pix_info[i].RESET_TRUTH_W   = RESET_TRUTH_W;
            pixel.RESET = reset;
            pixel.TSLR = tslr;
            pixel.RESET_TRUTH_ID = RESET_TRUTH_ID;
            pixel.RESET_TRUTH_W = RESET_TRUTH_W;

            // std::cout << index << std::endl;
            // index++;
        }

        delete hproj;

        return;
    } // reset_thnsparse_test

    void Pixel_Functions::Reset_THnSparse(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {
        // getting the size of the noise vector
        int Noise_Vector_Size = Gaussian_Noise.size();
        int Noise_index = 0;

        // THnSparseF histograms
        // https://root.cern/doc/master/classTHnSparse.html
        // https://root.cern/doc/master/sparsehist_8C.html
        // https://root.cern/doc/master/drawsparse_8C.html
        int bins[2] = { static_cast<int>(Qpix_params->Buffer_time/Qpix_params->Sample_time), 100001 };
        double xmin[2] = { 0.0, -1 };
        double xmax[2] = { Qpix_params->Buffer_time, 100000 };
        auto h = THnSparseF("h", "h", 2, bins, xmin, xmax);
        auto hproj0 = h.Projection(0);
        auto xaxis0 = hproj0->GetXaxis();
        auto hproj1 = h.Projection(1);
        auto xaxis1 = hproj1->GetXaxis();

        for (auto& pixel : Pix_info)
        {
            double tslr_ = 0;
            int charge = 0;
            int prev_bin_coordinate = 0;

            std::vector< double > reset;
            std::vector< double > tslr;

            std::vector< int > trk_id_holder;
            std::vector< std::vector< int > > RESET_TRUTH_ID;
            std::vector< std::vector< int > > RESET_TRUTH_W;

            std::map<int, int> track_id_map;

            // skip if it won't reset
            if (pixel.time.size() < (Qpix_params->Reset)*0.5) continue;
            std::cout << "pixel size: " << pixel.time.size() << std::endl;

            for (int idx = 0; idx < pixel.time.size(); ++idx)
            {
                // double data[2] = { pixel.time.at(idx), pixel.Trck_ID.at(idx) };
                // h.Fill(data);
                h.Fill(pixel.time.at(idx), pixel.Trk_ID.at(idx));
            }

            for (int bin_idx = 0; bin_idx < h.GetNbins(); ++bin_idx)
            {
                int bin_coordinates[2];
                int counts = h.GetBinContent(bin_idx, bin_coordinates);
                double current_time = xaxis0->GetBinUpEdge(bin_coordinates[0]);
                int current_track_id = xaxis1->GetBinUpEdge(bin_coordinates[1]);

                if (current_time > Qpix_params->Buffer_time) break;

                // adding noise from the noise vector
                int noise_start = Noise_index;
                int noise_stop = noise_start + (bin_coordinates[0] - prev_bin_coordinate) + 1;
                // std::cout << "noise_start, noise_stop: " << noise_start << ", " << noise_stop << std::endl;

                while (Noise_index < noise_stop)
                {
                    // if (Gaussian_Noise[Noise_index] > 0) charge += Gaussian_Noise[Noise_index];
                    if (Gaussian_Noise[Noise_index] != 0) charge += Gaussian_Noise[Noise_index];
                    // charge += Gaussian_Noise[Noise_index];
                    if (Gaussian_Noise[Noise_index] != 0) std::cout << Noise_index << ", " << xaxis0->GetBinUpEdge(prev_bin_coordinate) << ", " << charge << std::endl;

                    if (charge >= Qpix_params->Reset)
                    {
                        double noise_time = xaxis0->GetBinUpEdge(prev_bin_coordinate);
                        // // std::cout << noise << ", ";
                        // std::cout << charge << ", ";
                        tslr.push_back(noise_time - tslr_);
                        double tslr_tmp = noise_time - tslr_;
                        tslr_ = noise_time;
                        reset.push_back(noise_time);
                        // noise -= Qpix_params->Reset;
                        // std::cout << noise << ", ";
                        charge -= Qpix_params->Reset;
                        // std::cout << charge << ", ";
                        // std::cout << noise_time << ", ";
                        // // std::cout << tslr_tmp << std::endl;
                        // std::cout << tslr_tmp << ", " << "NOISE RESET" << std::endl;
                        // prev_reset_bin_coordinate = prev_bin_coordinate;
                        std::vector< int > trk_TrkIDs_holder;
                        std::vector< int > trk_weight_holder;
                        for (auto [trk_id, weight] : track_id_map)
                        {
                            trk_TrkIDs_holder.push_back(trk_id);
                            trk_weight_holder.push_back(weight);
                        }
                        RESET_TRUTH_ID.push_back(trk_TrkIDs_holder);
                        RESET_TRUTH_W.push_back(trk_weight_holder);
                        if (charge < Qpix_params->Reset) track_id_map.clear();
                    }

                    prev_bin_coordinate += 1;

                    Noise_index += 1;
                    if (Noise_index >= Noise_Vector_Size)
                    {
                        // std::cout << "Noise_index, Noise_Vector_Size, noise_stop: " << Noise_index << ", " << Noise_Vector_Size << ", " << noise_stop << std::endl;
                        Noise_index = 0;
                        noise_stop -= Noise_Vector_Size;
                    }
                    // if (Noise_index >= noise_stop) std::cout << "AYY " << bin_idx << ", " << current_time << ", " << Noise_index << ", " << noise_stop << std::endl;
                }

                if (track_id_map.find(current_track_id) != track_id_map.end()) track_id_map[current_track_id] += 1;
                else track_id_map[current_track_id] = 1;

                charge += counts;

                int next_bin_coordinates[2];
                int next_counts = h.GetBinContent(bin_idx+1, next_bin_coordinates);
                double next_time = xaxis0->GetBinUpEdge(next_bin_coordinates[0]);

                if (bin_coordinates[0] == next_bin_coordinates[0]) continue;

                if (charge >= Qpix_params->Reset)
                {
                    std::cout << charge << ", ";
                    tslr.push_back(current_time - tslr_);
                    double tslr_tmp = current_time - tslr_;
                    tslr_ = current_time;
                    reset.push_back(current_time);
                    charge -= Qpix_params->Reset;
                    std::cout << charge << ", ";
                    std::cout << current_time << ", ";
                    std::cout << tslr_tmp << std::endl;
                    std::vector< int > trk_TrkIDs_holder;
                    std::vector< int > trk_weight_holder;
                    for (auto [trk_id, weight] : track_id_map)
                    {
                        trk_TrkIDs_holder.push_back(trk_id);
                        trk_weight_holder.push_back(weight);
                    }
                    RESET_TRUTH_ID.push_back(trk_TrkIDs_holder);
                    RESET_TRUTH_W.push_back(trk_weight_holder);
                    if (charge < Qpix_params->Reset) track_id_map.clear();
                }

                prev_bin_coordinate = bin_coordinates[0]+1;

                if (bin_idx == h.GetNbins()-1)
                {
                    // h.GetNbins() - bin_coordinates[0] - 1;
                    // std::cout << "bin_idx, h.GetNbins(): " << bin_idx << ", " << h.GetNbins() << std::endl;

                    // adding noise from the noise vector
                    noise_start = Noise_index;
                    noise_stop = noise_start + (hproj0->GetNbinsX() - prev_bin_coordinate) + 1;
                    // std::cout << "noise_start, noise_stop, hproj->GetNbinsX(), prev_bin_coordinate, hproj->GetNbinsX()-prev_bin_coordinate: " << noise_start << ", " << noise_stop << ", " << hproj->GetNbinsX() << ", " << prev_bin_coordinate << ", " << hproj->GetNbinsX()-prev_bin_coordinate << std::endl;

                    while (Noise_index < noise_stop)
                    {
                        // if (Gaussian_Noise[Noise_index] > 0) charge += Gaussian_Noise[Noise_index];
                        if (Gaussian_Noise[Noise_index] != 0) charge += Gaussian_Noise[Noise_index];
                        // charge += Gaussian_Noise[Noise_index];
                        if (Gaussian_Noise[Noise_index] != 0) std::cout << Noise_index << ", " << xaxis0->GetBinUpEdge(prev_bin_coordinate+1) << ", " << charge << std::endl;

                        if (charge >= Qpix_params->Reset)
                        {
                            double noise_time = xaxis0->GetBinUpEdge(prev_bin_coordinate+1);
                            // std::cout << noise << ", ";
                            std::cout << charge << ", ";
                            tslr.push_back(noise_time - tslr_);
                            double tslr_tmp = noise_time - tslr_;
                            tslr_ = noise_time;
                            reset.push_back(noise_time);
                            // noise -= Qpix_params->Reset;
                            // std::cout << noise << ", ";
                            charge -= Qpix_params->Reset;
                            std::cout << charge << ", ";
                            std::cout << noise_time << ", ";
                            std::cout << tslr_tmp << std::endl;
                            // std::cout << tslr_tmp << ", " << "NOISE RESET" << std::endl;
                            // prev_reset_bin_coordinate = prev_bin_coordinate;
                            std::vector< int > trk_TrkIDs_holder;
                            std::vector< int > trk_weight_holder;
                            for (auto [trk_id, weight] : track_id_map)
                            {
                                trk_TrkIDs_holder.push_back(trk_id);
                                trk_weight_holder.push_back(weight);
                            }
                            RESET_TRUTH_ID.push_back(trk_TrkIDs_holder);
                            RESET_TRUTH_W.push_back(trk_weight_holder);
                            if (charge < Qpix_params->Reset) track_id_map.clear();
                        }

                        prev_bin_coordinate += 1;

                        Noise_index += 1;
                        if (Noise_index >= Noise_Vector_Size)
                        {
                            // std::cout << "Noise_index, Noise_Vector_Size, noise_stop, noise_stop-Noise_Vector_Size: " << Noise_index << ", " << Noise_Vector_Size << ", " << noise_stop << ", " << noise_stop-Noise_Vector_Size << std::endl;
                            Noise_index = 0;
                            noise_stop -= Noise_Vector_Size;
                        }
                        // if (Noise_index >= noise_stop) std::cout << "AYY " << bin_idx << ", " << current_time << ", " << Noise_index << ", " << noise_stop << std::endl;
                    }
                }

            }

            h.Reset("ICESM");

            // add it to the pixel info
            pixel.RESET = reset;
            pixel.TSLR = tslr;
            pixel.RESET_TRUTH_ID = RESET_TRUTH_ID;
            pixel.RESET_TRUTH_W = RESET_TRUTH_W;
        }

        delete hproj0;
        delete hproj1;

        return;
    } // Reset_THnSparse

    void Pixel_Functions::reset_map_test(Qpix::Qpix_Paramaters * Qpix_params, std::vector<double>& Gaussian_Noise, std::vector<Pixel_Info>& Pix_info)
    {

        int index = 0;

        for (auto pixel : Pix_info)
        {
            for (int idx = 0; idx < pixel.time.size(); ++idx)
            {
                // pixel.time.at(idx));
            }

            // std::cout << index << std::endl;
            // index++;
        }

        return;
    } // reset_map_test

}
