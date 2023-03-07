// -----------------------------------------------------------------------------
//  ROOTFileManager.cpp
//
//  Class definition of the ROOT file manager
//   * Author: Everybody is an author!
//   * Creation date: 31 August 2020
// -----------------------------------------------------------------------------

#include "ROOTFileManager.h"

// Boost includes
#include "boost/filesystem.hpp"

// ROOT includes
#include "TObject.h"

// math includes
#include <math.h>
#include <stdlib.h>
#include <iostream>

namespace bfs = boost::filesystem;

namespace Qpix {

    ROOTFileManager * ROOTFileManager::instance_=0;

    //--------------------------------------------------------------------------
    ROOTFileManager::ROOTFileManager()
      : ttree_(0), tfile_(0), inputfile(0), outputfile(0),
      inputeventtree(0), inputmetatree(0), outputeventtree(0), outputmetatree(0), run_(0), event_(0),
      number_particles_(0), number_entries_(0), number_hits_(0), energy_deposit_(0.), evt_initial(0),
      evt_final(-1), entry(0), firstEventInFile(0)
    {}

    //--------------------------------------------------------------------------
    ROOTFileManager::~ROOTFileManager()
    {}

    //-------------------------------------------------------------------------

    ROOTFileManager * ROOTFileManager::Instance()
    {
      if (instance_ == 0) instance_ = new ROOTFileManager();
      return instance_;
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::Initialize(std::string const& input_file, std::string const& output_file, int evt_start, int evt_end)
    {
        //bfs::copy_file(input_file, output_file, bfs::copy_option::overwrite_if_exists);

        // Make a copy of the original root file to a output_file path, only including the specified events 
        inputfile = new TFile(input_file.c_str());
        inputfile->GetObject("event_tree",inputeventtree);
        inputfile->GetObject("metadata",inputmetatree);
        number_entries_ = (Long64_t)inputeventtree->GetEntries();
        inputeventtree->SetBranchAddress("event",         &event_           );

        outputfile = new TFile(output_file.c_str(), "recreate");
        outputeventtree = inputeventtree->CloneTree(0);
        outputmetatree = inputmetatree->CloneTree();

        if (evt_end < evt_start && evt_end != -1)
        {
          std::cout << "Event end number must be greater than the start number" << std::endl;
          abort;
        }
        if (evt_start != -1) { evt_initial = evt_start; }
        if (evt_end != -1 ) 
        {
          evt_final = evt_end; 
        } else {
          evt_final = inputeventtree->GetMaximum("event");
        } 



        for (Long64_t i=0; i<number_entries_; i++){
          inputeventtree->GetEntry(i);
          if (event_>=evt_initial && event_<=evt_final) outputeventtree->Fill();
        }

        outputeventtree->Print();
        outputmetatree->Print();
        outputeventtree->AutoSave();
        outputmetatree->AutoSave();
        delete inputfile;
        delete outputfile;


        tfile_ = new TFile(output_file.c_str(),"update");
        if (tfile_ != 0) {
          tfile_->GetObject("event_tree", ttree_);
        } else {
          std::cout << "tfile is a null pointer" << std::endl;
        }

        if (ttree_ != 0) {
          this->set_branch_addresses(ttree_);
          ttree_->GetEntry(0);
          firstEventInFile = event_;
          
          ttree_->Print();
        } else {
          std::cout << "tree is a null pointer" << std::endl;
          abort;
        }
        

        tbranch_x_ = ttree_->Branch("pixel_x", &pixel_x_);
        tbranch_y_ = ttree_->Branch("pixel_y", &pixel_y_);
        tbranch_reset_ = ttree_->Branch("pixel_reset", &pixel_reset_);
        tbranch_tslr_ = ttree_->Branch("pixel_tslr", &pixel_tslr_);
        tbranch_reset_truth_track_id_ = ttree_->Branch("pixel_reset_truth_track_id", &pixel_reset_truth_track_id_);
        tbranch_reset_truth_weight_ = ttree_->Branch("pixel_reset_truth_weight", &pixel_reset_truth_weight_);


        if (tfile_ == 0)
        {
          std::cout << "tfile_ is a null pointer" << std::endl;
          abort;
        }
        tfile_->GetObject("metadata", metadata_);


        if (metadata_ == 0)
        {
          std::cout << "metadata_ is a null pointer" << std::endl;
          abort;
        }
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
        //event_ = -1;
        number_particles_ = -1;
        //number_hits_ = -1;
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
        std::cout << "Test Point -- Before SetBranchAddress" << std::endl;
        ttree->SetBranchAddress("number_hits", &number_hits_);
        std::cout << "Test Point -- After SetBranchAddress" << std::endl;
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
        tbranch_tslr_->Fill();
        tbranch_reset_truth_track_id_->Fill();
        tbranch_reset_truth_weight_->Fill();
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::EventReset()
    {   
        pixel_x_.clear();
        pixel_y_.clear();
        pixel_reset_.clear();
        pixel_tslr_.clear();
        pixel_reset_truth_track_id_.clear();
        pixel_reset_truth_weight_.clear();
    }

    //--------------------------------------------------------------------------
    void ROOTFileManager::Save()
    {
        // check for null pointers
        if (tfile_ == 0) { std::cout << "ttree_ is a null pointer" << std::endl; }
        if (ttree_ == 0) { std::cout << "ttree_ is a null pointer" << std::endl; }
        if (metadata_ == 0) { std::cout << "ttree_ is a null pointer" << std::endl; }

        // save only the new version of the tree
        ttree_->Write("", TObject::kOverwrite);
        metadata_->Write("", TObject::kOverwrite);

        // close file
        tfile_->Close();
    }

    //-------------------------------------------------------------------------

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
    void ROOTFileManager::AddMetadata(Qpix_Paramaters * const Qpix_params)
    {
        std::cout << "Beginnning of AddMetadata" << std::endl;
   
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

        std::cout << "End of AddMetadata" << std::endl;
    }

    //--------------------------------------------------------------------------
    // gets the event from the file and tunrs it into electrons
    void ROOTFileManager::Get_Event(int EVENT, Qpix_Paramaters * Qpix_params, std::vector<ELECTRON>& hit_e)
    {
        if (ttree_ == 0) {std::cout << "ttree_ is a null pointer" << std::endl;}
        entry = EVENT - firstEventInFile;
        if (entry > ttree_->GetEntries()) { std::cout << "entry is larger than NEntries in TTree" << std::endl;}
        ttree_->GetEntry(entry,1);

        std::cout << "entry: " << entry << std::endl;
        std::cout << "NEntries: " << ttree_->GetEntries() << std::endl;
        std::cout << "event_: " << event_ << std::endl; 
        std::cout << "nHits: " << number_hits_ << std::endl;

        int indexer = 0;
        // loop over all hits in the event
        for (Long64_t h_idx = 0; h_idx < number_hits_; ++h_idx)
        {
            // from PreStepPoint
            double const start_x = hit_start_x_->at(h_idx);      // cm
            double const start_y = hit_start_y_->at(h_idx);      // cm
            double const start_z = hit_start_z_->at(h_idx);      // cm
            double const start_t = hit_start_t_->at(h_idx)*1e-9; // sec

            // from PostStepPoint
            double const end_x = hit_end_x_->at(h_idx);      // cm
            double const end_y = hit_end_y_->at(h_idx);      // cm
            double const end_z = hit_end_z_->at(h_idx);      // cm
            double const end_t = hit_end_t_->at(h_idx)*1e-9; // sec

            // follow the track for truth matching
            int const hit_trk_id = hit_track_id_->at(h_idx); // track id

            if (start_t < 0.0 || start_t > Qpix_params->Buffer_time){continue;}

            // energy deposit
            double const energy_deposit = hit_energy_deposit_->at(h_idx);  // MeV

            // hit length
            double const length_of_hit = hit_length_->at(h_idx);  // cm

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
            if (Nelectron == 0){continue;}
            //std::cout << "h_idx: " << h_idx << "    Nelectron = " << Nelectron << std::endl;

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
                if (RandomUniform() >= exp(-T_drift/Qpix_params->Life_Time)){continue;}
                
                // diffuse the electrons position
                sigma_T = sqrt(2*Qpix_params->DiffusionT*T_drift);
                sigma_L = sqrt(2*Qpix_params->DiffusionL*T_drift);
                electron_x = RandomNormal(electron_loc_x,sigma_T);
                electron_y = RandomNormal(electron_loc_y,sigma_T);
                electron_z = RandomNormal(electron_loc_z,sigma_L);
		
                // add the electron to the vector.
                hit_e.push_back(ELECTRON());
                
                // convert the electrons x,y to a pixel index
                int Pix_Xloc, Pix_Yloc;
                Pix_Xloc = (int) ceil(electron_x / Qpix_params->Pix_Size);
                Pix_Yloc = (int) ceil(electron_y / Qpix_params->Pix_Size);

                hit_e[indexer].Pix_ID = (int)(Pix_Xloc*10000+Pix_Yloc);
                hit_e[indexer].time = electron_loc_t + ( electron_z / Qpix_params->E_vel );
                hit_e[indexer].Trk_ID = hit_trk_id;
                
                // Move to the next electron
                electron_loc_x += step_x;
                electron_loc_y += step_y;
                electron_loc_z += step_z;
                electron_loc_t += step_t;
                indexer += 1;
            }
        }
        // sorts the electrons in terms of the pixel ID
        std::sort(hit_e.begin(), hit_e.end(), Electron_Pix_Sort);
    }//Get_Event



    //--------------------------------------------------------------------------
    // Adds event that needs to be filled
    void ROOTFileManager::AddEvent(const std::vector<Pixel_Info> Pixel)
    {

        for (unsigned int i=0; i<Pixel.size() ; i++)
        {
            // skip pixel if there are no resets
            if (Pixel[i].RESET.size() < 1) continue;

            pixel_x_.push_back(Pixel[i].X_Pix);
            pixel_y_.push_back(Pixel[i].Y_Pix);
            pixel_reset_.push_back(Pixel[i].RESET);
            pixel_tslr_.push_back(Pixel[i].TSLR);
            pixel_reset_truth_track_id_.push_back(Pixel[i].RESET_TRUTH_ID);
            pixel_reset_truth_weight_.push_back(Pixel[i].RESET_TRUTH_W);
        }

    }//AddEvent

}
