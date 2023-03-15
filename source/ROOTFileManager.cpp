// -----------------------------------------------------------------------------
//  ROOTFileManager.cpp
//
//  Class definition of the ROOT file manager
//   * Author: Everybody is an author!
//   * Creation date: 31 August 2020
// -----------------------------------------------------------------------------

// Qpix includes
#include "ROOTFileManager.h"
#include "PixelResponse.h"

// Boost includes
#include "boost/filesystem.hpp"

// ROOT includes
#include "TObject.h"

// math includes
#include <math.h>

#include <algorithm>

namespace bfs = boost::filesystem;

namespace Qpix {

    //--------------------------------------------------------------------------
    ROOTFileManager::ROOTFileManager(std::string const& input_file, std::string const& output_file)
    {
        this->initialize(input_file, output_file);
    }

    //--------------------------------------------------------------------------
    ROOTFileManager::~ROOTFileManager()
    {}

    //--------------------------------------------------------------------------
    void ROOTFileManager::initialize(std::string const& input_file, std::string const& output_file)
    {
        // don't copy; just TTree.AddFriend provides this behavior without duplicating data..
        // now we're not dependent on Boost anymore too..
        // bfs::copy_file(input_file, output_file, bfs::copy_option::overwrite_if_exists);
        in_tfile_ = new TFile(input_file.data(), "read");
        in_ttree_ = (TTree*) in_tfile_->Get("event_tree");
        this->set_branch_addresses(in_ttree_);

        out_tfile_ = new TFile(output_file.data(), "recreate");
        out_ttree_ = new TTree("event_tree", "event_tree");
        out_ttree_->Branch("pixel_x", &b_pixel_x_);
        out_ttree_->Branch("pixel_y", &b_pixel_y_);
        out_ttree_->Branch("pixel_reset", &b_pixel_reset_);
        // out_ttree_->Branch("pixel_tslr", &pixel_tslr_);
        out_ttree_->Branch("pixel_reset_truth_track_id", &b_pixel_reset_truth_track_id_);
        out_ttree_->Branch("pixel_reset_truth_weight", &b_pixel_reset_truth_weight_);

        metadata_ = (TTree*) in_tfile_->Get("metadata");
        metadata_->SetBranchAddress("detector_length_x", &detector_length_x_);
        metadata_->SetBranchAddress("detector_length_y", &detector_length_y_);
        metadata_->SetBranchAddress("detector_length_z", &detector_length_z_);

        // meta branches
        tbranch_w_value_ = metadata_->Branch("w_value", &w_value_);
        tbranch_drift_velocity_ = metadata_->Branch("drift_velocity", &drift_velocity_);
        tbranch_longitudinal_diffusion_ = metadata_->Branch("longitudinal_diffusion", &longitudinal_diffusion_);
        tbranch_transverse_diffusion_ = metadata_->Branch("transverse_diffusion", &transverse_diffusion_);
        tbranch_electron_lifetime_ = metadata_->Branch("electron_lifetime", &electron_lifetime_);
        tbranch_readout_dimensions_ = metadata_->Branch("readout_dimensions", &readout_dimensions_);
        tbranch_pixel_size_ = metadata_->Branch("pixel_size", &pixel_size_);
        tbranch_reset_threshold_ = metadata_->Branch("reset_threshold", &reset_threshold_);
        tbranch_sample_time_ = metadata_->Branch("sample_time", &sample_time_);
        tbranch_buffer_window_ = metadata_->Branch("buffer_window", &buffer_window_);
        tbranch_dead_time_ = metadata_->Branch("dead_time", &dead_time_);
        tbranch_charge_loss_ = metadata_->Branch("charge_loss", &charge_loss_);
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::set_branch_addresses(TTree * ttree)
    {
        run_ = -1;
        event_ = -1;
        number_particles_ = -1;
        number_hits_ = -1;
        energy_deposit_ = -1;

        // use friend ttree
        // particle_track_id_ = 0;
        // particle_parent_track_id_ = 0;
        // particle_pdg_code_ = 0;
        // particle_mass_ = 0;
        // particle_charge_ = 0;
        // particle_process_key_ = 0;
        // particle_total_occupancy_ = 0;

        // particle_initial_x_ = 0;
        // particle_initial_y_ = 0;
        // particle_initial_z_ = 0;
        // particle_initial_t_ = 0;

        // particle_initial_px_ = 0;
        // particle_initial_py_ = 0;
        // particle_initial_pz_ = 0;
        // particle_initial_energy_ = 0;

        hit_track_id_ = 0;
        hit_start_x_ = 0;
        hit_start_y_ = 0;
        hit_start_z_ = 0;
        hit_start_t_ = 0;
        hit_end_x_ = 0;
        hit_end_y_ = 0;
        hit_end_z_ = 0;
        hit_end_t_ = 0;
        hit_length_ = 0;
        hit_energy_deposit_ = 0;
        hit_process_key_ = 0;

        // ttree->SetBranchAddress("run", &run_);
        // ttree->SetBranchAddress("event", &event_);
        // ttree->SetBranchAddress("number_particles", &number_particles_);
        // ttree->SetBranchAddress("number_hits", &number_hits_);
        // ttree->SetBranchAddress("energy_deposit", &energy_deposit_);
        ttree->SetBranchAddress("particle_id", &particle_id_);

        // use friend ttree
        // ttree->SetBranchAddress("particle_track_id", &particle_track_id_);
        // ttree->SetBranchAddress("particle_parent_track_id", &particle_parent_track_id_);
        // ttree->SetBranchAddress("particle_pdg_code", &particle_pdg_code_);
        // ttree->SetBranchAddress("particle_mass", &particle_mass_);
        // ttree->SetBranchAddress("particle_charge", &particle_charge_);
        // ttree->SetBranchAddress("particle_process_key", &particle_process_key_);
        // ttree->SetBranchAddress("particle_total_occupancy", &particle_total_occupancy_);

        // ttree->SetBranchAddress("particle_initial_x", &particle_initial_x_);
        // ttree->SetBranchAddress("particle_initial_y", &particle_initial_y_);
        // ttree->SetBranchAddress("particle_initial_z", &particle_initial_z_);
        // ttree->SetBranchAddress("particle_initial_t", &particle_initial_t_);

        // ttree->SetBranchAddress("particle_initial_px", &particle_initial_px_);
        // ttree->SetBranchAddress("particle_initial_py", &particle_initial_py_);
        // ttree->SetBranchAddress("particle_initial_pz", &particle_initial_pz_);
        // ttree->SetBranchAddress("particle_initial_energy", &particle_initial_energy_);

        ttree->SetBranchAddress("hit_track_id", &hit_track_id_);

        ttree->SetBranchAddress("hit_start_x", &hit_start_x_);
        ttree->SetBranchAddress("hit_start_y", &hit_start_y_);
        ttree->SetBranchAddress("hit_start_z", &hit_start_z_);
        ttree->SetBranchAddress("hit_start_t", &hit_start_t_);

        ttree->SetBranchAddress("hit_end_x", &hit_end_x_);
        ttree->SetBranchAddress("hit_end_y", &hit_end_y_);
        ttree->SetBranchAddress("hit_end_z", &hit_end_z_);
        ttree->SetBranchAddress("hit_end_t", &hit_end_t_);

        ttree->SetBranchAddress("hit_energy_deposit", &hit_energy_deposit_);
        ttree->SetBranchAddress("hit_length", &hit_length_);
        ttree->SetBranchAddress("hit_process_key", &hit_process_key_);
    }

    //--------------------------------------------------------------------------
    unsigned int ROOTFileManager::NumberEntries()
    {
        if (in_ttree_) return in_ttree_->GetEntries();
        return -1;
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::EventFill()
    {
        // fill if we got any resets
        if(pixel_x_.size() > 0){
            // for every pixel that has a reset
            for(int i=0; i<pixel_x_.size(); ++i){
                b_pixel_x_ = pixel_x_[i];
                b_pixel_y_ = pixel_y_[i];
                for(int j=0; j<pixel_reset_[i].size(); ++j){
                    b_pixel_reset_ = pixel_reset_[i][j];
                    b_pixel_reset_truth_track_id_ = pixel_reset_truth_track_id_[i][j];
                    b_pixel_reset_truth_weight_ = pixel_reset_truth_weight_[i][j];
                    out_ttree_->Fill();
                }
            }
        }
    }


    //--------------------------------------------------------------------------
    std::unordered_map<int, Qpix::Pixel_Info> ROOTFileManager::MakePixelInfoMap()
    {
        std::unordered_map<int, Qpix::Pixel_Info> pixel_map;
        if(detector_length_z_ < 1)
        {
            std::cout << "WARNING did not read metadata correctly for detector volume!\n";
            return pixel_map;
        }
        // we can figure out what the maximum pixel sizes must be from meta data
        int maxX = ceil(detector_length_x_ / pixel_size_);
        int maxY = ceil(detector_length_y_ / pixel_size_);

        // build the pixel map, start at 0 and end at +1 to include boundary
        for(int i=0; i<maxX+1; ++i){
            for(int j=0; j<maxY+1; ++j){
                pixel_map[Qpix::ID_Encoder(i, j)] = Qpix::Pixel_Info(i, j);
            }
        }

        return pixel_map;
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::EventReset()
    {   
        // only bother clearing if we have anything to clear
        if(pixel_x_.size() > 0){
            pixel_x_.clear();
            pixel_y_.clear();
            pixel_reset_.clear();
            // pixel_tslr_.clear();
            pixel_reset_truth_track_id_.clear();
            pixel_reset_truth_weight_.clear();
        }
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::Save()
    {
        // save only the new version of the tree
        out_ttree_->Write();
        metadata_->Write("", TObject::kOverwrite);
        // close file
        out_tfile_->Close();
    }

    double ROOTFileManager::Modified_Box(double dEdx) 
    {
    // Moddeling the recombination based on dEdx
    // the function was taken form 
    // https://www.phy.bnl.gov/~chao/uboone/docdb/files/LArProperty.pdf slide 10
    // and 
    // https://arxiv.org/pdf/1306.1712.pdf
    double B_by_E = 0.212 / 0.5;
    double A      = 0.930;
    double Recombination;

    Recombination = log(A + B_by_E * dEdx) / (B_by_E * dEdx);
    if (Recombination<0){Recombination=0;}

    return Recombination;
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::AddMetadata(Qpix::Qpix_Paramaters * const Qpix_params)
    {
        w_value_ = Qpix_params->Wvalue;
        drift_velocity_ = Qpix_params->E_vel;
        longitudinal_diffusion_ = Qpix_params->DiffusionL;
        transverse_diffusion_ = Qpix_params->DiffusionT;
        electron_lifetime_ = Qpix_params->Life_Time;
        readout_dimensions_ = Qpix_params->Readout_Dim;
        pixel_size_ = Qpix_params->Pix_Size;
        reset_threshold_ = Qpix_params->Reset;
        sample_time_ = Qpix_params->Sample_time;
        buffer_window_ = Qpix_params->Buffer_time;
        dead_time_ = Qpix_params->Dead_time;
        charge_loss_ = static_cast<int>(Qpix_params->Charge_loss);

        tbranch_w_value_->Fill();
        tbranch_drift_velocity_->Fill();
        tbranch_longitudinal_diffusion_->Fill();
        tbranch_transverse_diffusion_->Fill();
        tbranch_electron_lifetime_->Fill();
        tbranch_readout_dimensions_->Fill();
        tbranch_pixel_size_->Fill();
        tbranch_reset_threshold_->Fill();
        tbranch_sample_time_->Fill();
        tbranch_buffer_window_->Fill();
        tbranch_dead_time_->Fill();
        tbranch_charge_loss_->Fill();

        // populate the detector size values
        metadata_->GetEntry(0);
    }

    //--------------------------------------------------------------------------
    // gets the event from the file and tunrs it into electrons
    void ROOTFileManager::Get_Event(int EVENT, Qpix::Qpix_Paramaters * Qpix_params, std::vector<Qpix::ELECTRON>& hit_e,
                                    bool sort_elec)
    {
        in_ttree_->GetEntry(EVENT);

        int indexer = 0;

        // from PreStepPoint
        double const start_x = hit_start_x_;      // cm
        double const start_y = hit_start_y_;      // cm
        double const start_z = hit_start_z_;      // cm
        double const start_t = hit_start_t_*1e-9; // sec

        // from PostStepPoint
        double const end_x = hit_end_x_;      // cm
        double const end_y = hit_end_y_;      // cm
        double const end_z = hit_end_z_;      // cm
        double const end_t = hit_end_t_*1e-9; // sec

        // follow the track for truth matching
        int const hit_trk_id = hit_track_id_; // track id

        if (start_t < 0.0 || start_t > Qpix_params->Buffer_time){return;}

        // energy deposit
        double const energy_deposit = hit_energy_deposit_;  // MeV

        // hit length
        double const length_of_hit = hit_length_;  // cm

        // Set up the paramaters for the recombiataion 
        double const dEdx = energy_deposit/length_of_hit;
        double const Recombonation = Modified_Box(dEdx);
        int Nelectron;

        // to account for recombination or not
        // calcualte the number of electrons in the hit
        if (Qpix_params->Recombination)
        {
            Nelectron = round(Recombonation * (energy_deposit*1e6/Qpix_params->Wvalue) );
        }else
        {
            Nelectron = round( (energy_deposit*1e6/Qpix_params->Wvalue) );
        }
        
        // if not enough move on
        if (Nelectron == 0){return;}

        // define the electrons start position
        double electron_loc_x = start_x;
        double electron_loc_y = start_y;
        double electron_loc_z = start_z;
        double electron_loc_t = start_t;
        
        // Determin the "step" size (pre to post hit)
        double const step_x = (end_x - start_x) / Nelectron;
        double const step_y = (end_y - start_y) / Nelectron;
        double const step_z = (end_z - start_z) / Nelectron;
        double const step_t = (end_t - start_t) / Nelectron;

        double electron_x, electron_y, electron_z;
        double T_drift, sigma_L, sigma_T;

        // Loop through the electrons 
        for (int i = 0; i < Nelectron; i++) 
        {
            // calculate drift time for diffusion 
            T_drift = electron_loc_z / Qpix_params->E_vel;
            // electron lifetime
            if (Qpix::RandomUniform() >= exp(-T_drift/Qpix_params->Life_Time)){continue;}
            
            // diffuse the electrons position
            sigma_T = sqrt(2*Qpix_params->DiffusionT*T_drift);
            sigma_L = sqrt(2*Qpix_params->DiffusionL*T_drift);
            electron_x = Qpix::RandomNormal(electron_loc_x,sigma_T);
            electron_y = Qpix::RandomNormal(electron_loc_y,sigma_T);
            electron_z = Qpix::RandomNormal(electron_loc_z,sigma_L);
    
            // add the electron to the vector.
            hit_e.push_back(Qpix::ELECTRON());
            
            // convert the electrons x,y to a pixel index
            int Pix_Xloc, Pix_Yloc;
            Pix_Xloc = (int) ceil(electron_x / Qpix_params->Pix_Size);
            Pix_Yloc = (int) ceil(electron_y / Qpix_params->Pix_Size);

            hit_e[indexer].Pix_ID = Qpix::ID_Encoder(Pix_Xloc, Pix_Yloc);
            hit_e[indexer].time = electron_loc_t + ( electron_z / Qpix_params->E_vel );
            // hit_e[indexer].Trk_ID = hit_trk_id;
            // change to tracking source of hit-event to radiogenic particle
            hit_e[indexer].Trk_ID = particle_id_; 
            
            // Move to the next electron
            electron_loc_x += step_x;
            electron_loc_y += step_y;
            electron_loc_z += step_z;
            electron_loc_t += step_t;
            indexer += 1;
        }
        // sorts the electrons in terms of the pixel ID
        if(sort_elec)
            std::sort(hit_e.begin(), hit_e.end(), Qpix::Electron_Pix_Sort);
        else
            std::sort(hit_e.begin(), hit_e.end(), Qpix::Pixel_Time_Sorter);
    }//Get_Event



    //--------------------------------------------------------------------------
    // Adds event that needs to be filled
    void ROOTFileManager::AddEvent(const std::vector<Qpix::Pixel_Info> Pixel)
    {

        for (unsigned int i=0; i<Pixel.size() ; i++)
        {
            // skip pixel if there are no resets
            if (Pixel[i].RESET.size() < 1) continue;

            pixel_x_.push_back(Pixel[i].X_Pix);
            pixel_y_.push_back(Pixel[i].Y_Pix);
            pixel_reset_.push_back(Pixel[i].RESET);
            pixel_reset_truth_track_id_.push_back(Pixel[i].RESET_TRUTH_ID);
            pixel_reset_truth_weight_.push_back(Pixel[i].RESET_TRUTH_W);
        }

    }//AddEvent

    void ROOTFileManager::AddEvent(const std::set<int>& hit_ids, std::unordered_map<int, Qpix::Pixel_Info>& mPixelMap)
    {
        // add all of the pixel vectors we care about
        for(auto i : hit_ids)
        {
            Pixel_Info& pixel_info = mPixelMap[i];
            if(pixel_info.RESET.size() < 1) continue;
            pixel_x_.push_back(pixel_info.X_Pix);
            pixel_y_.push_back(pixel_info.Y_Pix);
            pixel_reset_.push_back(std::move(pixel_info.RESET));
            // pixel_tslr_.push_back(std::move(pixel_info.TSLR));
            pixel_reset_truth_track_id_.push_back(std::move(pixel_info.RESET_TRUTH_ID));
            pixel_reset_truth_weight_.push_back(std::move(pixel_info.RESET_TRUTH_W));
            // empty these vectors and keep their memory small
            std::vector<double>().swap(pixel_info.RESET);
            std::vector<std::vector<int>>().swap(pixel_info.RESET_TRUTH_ID);
            std::vector<std::vector<int>>().swap(pixel_info.RESET_TRUTH_W);
        }
    }
}
