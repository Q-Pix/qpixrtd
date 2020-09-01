// -----------------------------------------------------------------------------
//  ROOTFileManager.cpp
//
//  Class definition of the ROOT file manager
//   * Author: Everybody is an author!
//   * Creation date: 31 August 2020
// -----------------------------------------------------------------------------

#include "Qpix/ROOTFileManager.h"

// Boost includes
#include "boost/filesystem.hpp"

// ROOT includes
#include "TObject.h"

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
        bfs::copy_file(input_file, output_file, bfs::copy_option::overwrite_if_exists);
        tfile_ = new TFile(output_file.data(), "update");
        ttree_ = (TTree*) tfile_->Get("event_tree");

        this->set_branch_addresses(ttree_);

        tbranch_x_ = ttree_->Branch("pixel_x", &pixel_x_);
        tbranch_y_ = ttree_->Branch("pixel_y", &pixel_y_);
        tbranch_reset_ = ttree_->Branch("pixel_reset", &pixel_reset_);
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::set_branch_addresses(TTree * ttree)
    {
        run_ = -1;
        event_ = -1;
        number_particles_ = -1;
        number_hits_ = -1;
        energy_deposit_ = -1;

        particle_track_id_ = 0;
        particle_parent_track_id_ = 0;
        particle_pdg_code_ = 0;
        particle_mass_ = 0;
        particle_charge_ = 0;
        particle_process_key_ = 0;
        particle_total_occupancy_ = 0;

        particle_initial_x_ = 0;
        particle_initial_y_ = 0;
        particle_initial_z_ = 0;
        particle_initial_t_ = 0;

        particle_initial_px_ = 0;
        particle_initial_py_ = 0;
        particle_initial_pz_ = 0;
        particle_initial_energy_ = 0;

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

        ttree->SetBranchAddress("run", &run_);
        ttree->SetBranchAddress("event", &event_);
        ttree->SetBranchAddress("number_particles", &number_particles_);
        ttree->SetBranchAddress("number_hits", &number_hits_);
        ttree->SetBranchAddress("energy_deposit", &energy_deposit_);

        ttree->SetBranchAddress("particle_track_id", &particle_track_id_);
        ttree->SetBranchAddress("particle_parent_track_id", &particle_parent_track_id_);
        ttree->SetBranchAddress("particle_pdg_code", &particle_pdg_code_);
        ttree->SetBranchAddress("particle_mass", &particle_mass_);
        ttree->SetBranchAddress("particle_charge", &particle_charge_);
        ttree->SetBranchAddress("particle_process_key", &particle_process_key_);
        ttree->SetBranchAddress("particle_total_occupancy", &particle_total_occupancy_);

        ttree->SetBranchAddress("particle_initial_x", &particle_initial_x_);
        ttree->SetBranchAddress("particle_initial_y", &particle_initial_y_);
        ttree->SetBranchAddress("particle_initial_z", &particle_initial_z_);
        ttree->SetBranchAddress("particle_initial_t", &particle_initial_t_);

        ttree->SetBranchAddress("particle_initial_px", &particle_initial_px_);
        ttree->SetBranchAddress("particle_initial_py", &particle_initial_py_);
        ttree->SetBranchAddress("particle_initial_pz", &particle_initial_pz_);
        ttree->SetBranchAddress("particle_initial_energy", &particle_initial_energy_);

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
        if (ttree_) return ttree_->GetEntries();
        return -1;
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::EventFill()
    {
        tbranch_x_->Fill();
        tbranch_y_->Fill();
        tbranch_reset_->Fill();
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::EventReset()
    {   
        pixel_x_.clear();
        pixel_y_.clear();
        pixel_reset_.clear();
    }

    // //--------------------------------------------------------------------------
    // void ROOTFileManager::WriteTree()
    // {
    //     // save only the new version of the tree
    //     ttree_->Write("", TObject::kOverwrite);
    // }

    // //--------------------------------------------------------------------------
    // void ROOTFileManager::CloseFile()
    // {
    //     tfile_->Close();
    // }

    //--------------------------------------------------------------------------
    void ROOTFileManager::Save()
    {
        // save only the new version of the tree
        ttree_->Write("", TObject::kOverwrite);
        // close file
        tfile_->Close();
    }

    //--------------------------------------------------------------------------
    // gets the event from the file and tunrs it into electrons
    void ROOTFileManager::Get_Event(int EVENT, Qpix::Qpix_Paramaters * Qpix_params, std::vector<Qpix::ELECTRON>& hit_e)
    {
        ttree_->GetEntry(EVENT);

        int indexer = 0;
        // loop over all hits in the event
        for (int h_idx = 0; h_idx < number_hits_; ++h_idx)
        {
            // from PreStepPoint
            double const start_x = hit_start_x_->at(h_idx);  // cm
            double const start_y = hit_start_y_->at(h_idx);  // cm
            double const start_z = hit_start_z_->at(h_idx);  // cm
            double const start_t = hit_start_t_->at(h_idx);  // ns

            // from PostStepPoint
            double const end_x = hit_end_x_->at(h_idx);  // cm
            double const end_y = hit_end_y_->at(h_idx);  // cm
            double const end_z = hit_end_z_->at(h_idx);  // cm
            double const end_t = hit_end_t_->at(h_idx);  // ns

            // energy deposit
            double const energy_deposit = hit_energy_deposit_->at(h_idx);  // MeV

            // calcualte the number of electrons in the hit
            int Nelectron = round(energy_deposit*1e6/Qpix_params->Wvalue);
            // if not enough move on
            if (Nelectron == 0){continue;}

            // define the electrons start position
            double electron_loc_x = start_x +50;
            double electron_loc_y = start_y +50;
            double electron_loc_z = start_z +250;
            double electron_loc_t = start_t;
            // Determin the "step" size
            double const step_x = (end_x - start_x) / Nelectron;
            double const step_y = (end_y - start_y) / Nelectron;
            double const step_z = (end_z - start_z) / Nelectron;
            double const step_t = (end_t - start_t) / Nelectron;

            double electron_x, electron_y, electron_z;
            double T_drift, sigma_L, sigma_T;

            // Loop through the electrons 
            for (int i = 0; i < Nelectron; i++) 
            {
                // calculate frift time for diffusion 
                T_drift = electron_loc_z / Qpix_params->E_vel;
                // electron lifetime
                if (Qpix::RandomUniform() >= exp(-T_drift/Qpix_params->Life_Time)){continue;}
                
                // diffuse the electrons position
                sigma_T = sqrt(2*Qpix_params->DiffusionT*T_drift);
                sigma_L = sqrt(2*Qpix_params->DiffusionL*T_drift);
                electron_x = Qpix::RandomNormal(electron_loc_x,sigma_T);
                electron_y = Qpix::RandomNormal(electron_loc_y,sigma_T);
                electron_z = Qpix::RandomNormal(electron_loc_z,sigma_L);

                // check event is contained after diffused ( one pixel from the edge)
                if (!(0.4 <= electron_x && electron_x <= 99.6) || !(0.4 <= electron_y && electron_y <= 99.6) || !(0.1 <= electron_z && electron_z <= 499.6)){continue;}

                // add the electron to the vector.
                hit_e.push_back(Qpix::ELECTRON());
                
                // convert the electrons x,y to a pixel index
                int Pix_Xloc, Pix_Yloc;
                Pix_Xloc = (int) ceil(electron_x / Qpix_params->Pix_Size);
                Pix_Yloc = (int) ceil(electron_y / Qpix_params->Pix_Size);

                hit_e[indexer].Pix_ID = (int)(Pix_Xloc*10000+Pix_Yloc);
                hit_e[indexer].time = (int)ceil( electron_loc_t + ( electron_z / Qpix_params->E_vel ) ) ;
                
                // Move to the next electron
                electron_loc_x += step_x;
                electron_loc_y += step_y;
                electron_loc_z += step_z;
                electron_loc_t += step_t;
                indexer += 1;
            }

        }

        // sorts the electrons in terms of the pixel ID
        std::sort(hit_e.begin(), hit_e.end(), Qpix::Electron_Pix_Sort);

    }//Get_Event

    //--------------------------------------------------------------------------
    // Adds event that needs to be filled
    void ROOTFileManager::AddEvent(const std::vector<Qpix::Pixel_Info> Pixel)
    {

        for (unsigned int i=0; i<Pixel.size() ; i++)
        {
            // skip pixel if there are no resets
            if (Pixel[i].RESET.size() < 1) continue;

            // vector of resets as double instead of int for workaround
            std::vector< double > reset_double;

            for (unsigned int j=0; j<Pixel[i].RESET.size(); j++)
            {
                // cast from int to double
                reset_double.push_back(static_cast<double>(Pixel[i].RESET[j]));
            }

            // add to tree vectors if there are resets
            if (reset_double.size() > 0)
            {
                pixel_x_.push_back(Pixel[i].X_Pix);
                pixel_y_.push_back(Pixel[i].Y_Pix);
                pixel_reset_.push_back(reset_double);
            }
        }

    }//AddEvent

}
