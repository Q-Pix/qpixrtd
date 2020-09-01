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
#include "Qpix/Random.h"
#include "Qpix/Structures.h"

// ROOT includes
#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

// C++ includes
#include <map>
#include <set>

namespace Qpix {

    class ROOTFileManager {

        public:

            ROOTFileManager(std::string const&, std::string const&);
            ~ROOTFileManager();

            unsigned int NumberEntries();
            void EventFill();
            void EventReset();
            // void WriteTree();
            // void CloseFile();
            void Save();

            void Get_Event(int, Qpix::Qpix_Paramaters *, std::vector< Qpix::ELECTRON > &);
            void AddEvent(std::vector<Qpix::Pixel_Info> const);

        private:

            TFile * tfile_;
            TTree * ttree_;

            //--------------------------------------------------
            // new branch variables
            //--------------------------------------------------

            std::vector< int > pixel_x_;
            std::vector< int > pixel_y_;
            std::vector< std::vector < double > > pixel_reset_;

            TBranch * tbranch_x_;
            TBranch * tbranch_y_;
            TBranch * tbranch_reset_;

            //--------------------------------------------------
            // existing branch variables
            //--------------------------------------------------

            int run_;
            int event_;
            int number_particles_;
            int number_hits_;
            double energy_deposit_;

            std::vector< int >    * particle_track_id_;
            std::vector< int >    * particle_parent_track_id_;
            std::vector< int >    * particle_pdg_code_;
            std::vector< double > * particle_mass_;
            std::vector< double > * particle_charge_;
            std::vector< int >    * particle_process_key_;
            std::vector< int >    * particle_total_occupancy_;

            std::vector< double > * particle_initial_x_;
            std::vector< double > * particle_initial_y_;
            std::vector< double > * particle_initial_z_;
            std::vector< double > * particle_initial_t_;

            std::vector< double > * particle_initial_px_;
            std::vector< double > * particle_initial_py_;
            std::vector< double > * particle_initial_pz_;
            std::vector< double > * particle_initial_energy_;

            std::vector< int >    * hit_track_id_;
            std::vector< double > * hit_start_x_;
            std::vector< double > * hit_start_y_;
            std::vector< double > * hit_start_z_;
            std::vector< double > * hit_start_t_;
            std::vector< double > * hit_end_x_;
            std::vector< double > * hit_end_y_;
            std::vector< double > * hit_end_z_;
            std::vector< double > * hit_end_t_;
            std::vector< double > * hit_length_;
            std::vector< double > * hit_energy_deposit_;
            std::vector< int >    * hit_process_key_;

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
