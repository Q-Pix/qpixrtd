// -----------------------------------------------------------------------------
//  ROOTFileManager.h
//
//  Class definition of the ROOT file manager
//   * Author: Everybody is an author!
//   * Creation date: 31 August 2020
// -----------------------------------------------------------------------------

#ifndef ROOTFileManager_h
#define ROOTFileManager_h 1

// Q-Pix includes
#include "Random.h"
#include "Structures.h"

// ROOT includes
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

// C++ includes
#include <unordered_map>
#include <set>

namespace Qpix {

    class ROOTFileManager {

        public:

            ROOTFileManager(std::string const&, std::string const&);
            ~ROOTFileManager();

            unsigned int NumberEntries();
            void EventFill();
            void EventReset();
            void Save();

            void AddMetadata(Qpix::Qpix_Paramaters * const);
            double Modified_Box(double dEdx);
            void Get_Event(int, Qpix::Qpix_Paramaters *, std::vector< Qpix::ELECTRON > &, bool sort_elec=true);
            void AddEvent(std::vector<Qpix::Pixel_Info> const);
            void AddEvent(const std::set<int>&, std::unordered_map<int, Qpix::Pixel_Info>&);
            std::unordered_map<int, Qpix::Pixel_Info> MakePixelInfoMap();

        private:

            TFile * in_tfile_;
            TFile * out_tfile_;
            TTree * in_ttree_;
            TTree * out_ttree_;

            //--------------------------------------------------
            // new branch variables
            //--------------------------------------------------

            // containers to flatten branches
            std::vector< int > pixel_x_;
            std::vector< int > pixel_y_;
            std::vector< std::vector < double > > pixel_reset_;
            // std::vector< std::vector < double > > pixel_tslr_; // don't need this..
            std::vector< std::vector< std::vector < int > > > pixel_reset_truth_track_id_;
            std::vector< std::vector< std::vector < int > > > pixel_reset_truth_weight_;

            // branch holders, output TTree entries are resets
            int b_pixel_x_;
            int b_pixel_y_;
            double b_pixel_reset_;
            // double pixel_tslr_;
            std::vector<int> b_pixel_reset_truth_track_id_;
            std::vector<int> b_pixel_reset_truth_weight_;

            TBranch * tbranch_x_;
            TBranch * tbranch_y_;
            TBranch * tbranch_reset_;
            // TBranch * tbranch_tslr_;
            TBranch * tbranch_reset_truth_track_id_;
            TBranch * tbranch_reset_truth_weight_;

            //--------------------------------------------------
            // existing branch variables
            //--------------------------------------------------

            int run_;
            int event_;
            int number_particles_;
            int number_hits_;
            double energy_deposit_;
            int particle_id_;

            // reserve for friend tree
            // std::vector< int >    * particle_track_id_;
            // std::vector< int >    * particle_parent_track_id_;
            // std::vector< int >    * particle_pdg_code_;
            // std::vector< double > * particle_mass_;
            // std::vector< double > * particle_charge_;
            // std::vector< int >    * particle_process_key_;
            // std::vector< int >    * particle_total_occupancy_;

            // std::vector< double > * particle_initial_x_;
            // std::vector< double > * particle_initial_y_;
            // std::vector< double > * particle_initial_z_;
            // std::vector< double > * particle_initial_t_;

            // std::vector< double > * particle_initial_px_;
            // std::vector< double > * particle_initial_py_;
            // std::vector< double > * particle_initial_pz_;
            // std::vector< double > * particle_initial_energy_;

            // update to read flattened hits entry at a time
            int hit_track_id_;
            double hit_start_x_;
            double hit_start_y_;
            double hit_start_z_;
            double hit_start_t_;
            double hit_end_x_;
            double hit_end_y_;
            double hit_end_z_;
            double hit_end_t_;
            double hit_length_;
            double hit_energy_deposit_;
            int hit_process_key_;

            //--------------------------------------------------
            // metadata
            //--------------------------------------------------

            TTree * metadata_;

            double detector_length_x_;
            double detector_length_y_;
            double detector_length_z_;

            double w_value_;
            double drift_velocity_;
            double longitudinal_diffusion_;
            double transverse_diffusion_;
            double electron_lifetime_;
            double readout_dimensions_;
            double pixel_size_;
            int    reset_threshold_;
            int    sample_time_;
            int    buffer_window_;
            int    dead_time_;
            int    charge_loss_;

            TBranch * tbranch_w_value_;
            TBranch * tbranch_drift_velocity_;
            TBranch * tbranch_longitudinal_diffusion_;
            TBranch * tbranch_transverse_diffusion_;
            TBranch * tbranch_electron_lifetime_;
            TBranch * tbranch_readout_dimensions_;
            TBranch * tbranch_pixel_size_;
            TBranch * tbranch_reset_threshold_;
            TBranch * tbranch_sample_time_;
            TBranch * tbranch_buffer_window_;
            TBranch * tbranch_dead_time_;
            TBranch * tbranch_charge_loss_;

            //--------------------------------------------------
            // initialize
            //--------------------------------------------------
            void initialize(std::string const&, std::string const&);

            //--------------------------------------------------
            // set branch addresses
            //--------------------------------------------------
            void set_branch_addresses(TTree *);

    };

}
#endif
